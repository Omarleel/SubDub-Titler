import { useEffect, useMemo, useRef, useState } from 'react'
import { api, type Component, type DubCue, type DubOptions, type Gate, type Job, type Ratings } from './api'
import { clock, usePlayer, type Range } from './media'
import * as ui from './ui'

const SKIP_SECONDS = 3
const LANGUAGES = [{ code: 'es', title: 'Español' }, { code: 'en', title: 'Inglés' },
  { code: 'ja', title: 'Japonés' }, { code: 'zh', title: 'Chino' }]
const PRESETS = [
  { steps: 16, title: 'Equilibrado (16 pasos)' },
  { steps: 32, title: 'Máxima calidad (32 pasos, ~2× más lento)' },
  { steps: 8, title: 'Rápido (8 pasos, menos parecido de voz)' },
]
const STAGE_LABEL: Record<string, string> = {
  inicio: 'Preparando', transcribir: 'Transcribiendo', separar: 'Separando voz y fondo',
  voz: 'Preparando la voz clonada', traducir: 'Traduciendo', sintetizar: 'Sintetizando la voz',
  verificar: 'Verificando que se entiende', mezclar: 'Mezclando y exportando', evaluar: 'Midiendo la calidad', listo: 'Listo',
}
const GATES: Record<string, { label: string; format: (v: number) => string }> = {
  cer_ida_vuelta: { label: 'Inteligibilidad (CER ida y vuelta)', format: v => pct(v) },
  similitud_voz_relativa: { label: 'Parecido de voz frente al original', format: v => `${(v * 100).toFixed(0)} %` },
  utmos_vs_original: { label: 'Naturalidad frente al original (ΔUTMOS)', format: v => signed(v) },
  sincronia: { label: 'Subtítulos dentro de tolerancia', format: v => pct(v) },
  solapes: { label: 'Voces solapadas', format: v => String(v) },
  velocidad_en_rango: { label: 'Velocidad de habla en rango', format: v => pct(v) },
  fuga_voz_fondo_ppm: { label: 'Voz original en el fondo (palabras/min)', format: v => v.toFixed(1) },
  error_lufs: { label: 'Error de sonoridad (LU)', format: v => v.toFixed(1) },
  pico_real_dbtp: { label: 'Pico real (dBTP)', format: v => v.toFixed(1) },
}

const BENCH = `${ui.panelBare} sticky top-[12px] h-[calc(100vh-24px)] min-h-[520px] flex flex-col py-[18px] px-[20px] upto1100:static upto1100:h-auto upto1100:min-h-0`
const BENCH_HEAD = 'flex items-center justify-between mb-[12px] flex-wrap gap-[10px]'
const SELECT = 'bg-field border border-field-line rounded-[7px] py-[9px] px-[10px] text-field-text max-w-[340px]'
const TRANSPORT_BTN = `${ui.ghostBtn} disabled:opacity-45 disabled:cursor-not-allowed`
const COUNT = 'bg-[#172133] rounded-[7px] p-[9px] min-w-0'
const COUNT_LABEL = 'text-[9px] tracking-[.6px] uppercase text-[#7185a1]'
const COUNT_VALUE = 'mt-[5px] mx-0 mb-0 font-semibold text-[15px] text-ink whitespace-nowrap overflow-hidden text-ellipsis'
const TOGGLE = 'py-[7px] px-[12px] text-[12px] font-semibold border-0 cursor-pointer'

const pct = (v: number) => `${(v * 100).toFixed(1)} %`
const signed = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}`
const seconds = (v: number) => `${v.toFixed(2)} s`

type Tone = keyof typeof ui.TONES
type Mode = 'dub' | 'original'

/** Semáforo por subtítulo: lo que conviene revisar primero. */
function cueTone(cue: DubCue, cer: number | null | undefined): Tone {
  if (cue.action === 'shorten' || (cer ?? 0) > 0.15) return 'bad'
  if ((cer ?? 0) > 0.05 || cue.speed > 1.15) return 'warn'
  return 'good'
}

const DOT: Record<Tone, string> = { good: 'bg-ok', warn: 'bg-warn', bad: 'bg-bad' }

function vtt(track: DubCue[]) {
  const stamp = (s: number) => new Date(Math.max(0, s) * 1000).toISOString().slice(11, 23)
  return 'WEBVTT\n\n' + track.map(c =>
    `${stamp(c.start)} --> ${stamp(c.start + Math.max(c.generated_seconds, 0.3))}\n${c.target_text}\n`).join('\n')
}

type Props = { jobs: Job[]; job: Job | undefined; onSelect: (id: string) => void }

export function DubPage({ jobs, job, onSelect }: Props) {
  const sources = jobs.filter(j => j.kind === 'transcribe' && j.status === 'completed' && j.transcript)
  const source = job?.kind === 'dub' ? jobs.find(j => j.id === job.parent_id) : job?.kind === 'transcribe' ? job : sources[0]
  const dubsOfSource = jobs.filter(j => j.kind === 'dub' && j.parent_id === source?.id)
  const dub = job?.kind === 'dub' ? job : dubsOfSource[0]
  const result = dub?.status === 'completed' ? dub.result : null
  const track = result?.track ?? []
  const quality = useMemo(() => new Map((result?.cue_quality ?? []).map(q => [q.index, q])), [result])
  const version = dub?.updated_at ?? ''

  const player = usePlayer()
  const [mode, setMode] = useState<Mode>('dub')
  const resume = useRef<{ time: number; play: boolean; range?: Range } | null>(null)
  const [selected, setSelected] = useState(0)
  const [text, setText] = useState('')
  const [pendingJob, setPendingJob] = useState<string | null>(null)
  const [ratings, setRatings] = useState<Ratings | null>(null)
  const [options, setOptions] = useState<DubOptions>({
    target_language: 'es', num_step: 16, tts_batch: 8, n_best: 4, separator: 'htdemucs',
    evaluate: true, tts_engine: 'omnivoice', translator: 'nllb-600m', diarize: false, max_speakers: 4,
    merge_fragments: true, verify_attempts: 1, reference_candidates: 1,
  })
  const [engines, setEngines] = useState<Component[]>([])
  useEffect(() => { api.dubComponents().then(r => setEngines(r.components)).catch(() => setEngines([])) }, [])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const cue = track[selected] as DubCue | undefined
  const cueQuality = cue ? quality.get(cue.index) : undefined
  const range = cue ? { start: cue.start, end: cue.start + Math.max(cue.slot_seconds, cue.generated_seconds) } : null
  const regenerating = !!pendingJob && jobs.find(j => j.id === pendingJob)?.status !== 'completed'
    && jobs.find(j => j.id === pendingJob)?.status !== 'failed'

  useEffect(() => {
    if (source) setOptions(o => ({ ...o, target_language: source.transcript?.language === 'es' ? 'en' : o.target_language }))
  }, [source?.id])
  useEffect(() => { setSelected(0); setPendingJob(null); setRatings(null)
    if (dub?.status === 'completed') api.ratings(dub.id).then(setRatings).catch(() => setRatings(null))
  }, [dub?.id, dub?.status])
  useEffect(() => { setText(cue?.target_text ?? '') }, [cue?.index, cue?.target_text])
  useEffect(() => {
    const finished = jobs.find(j => j.id === pendingJob)
    if (finished?.status === 'failed') { setError(finished.error); setPendingJob(null) }
    if (finished?.status === 'completed') setPendingJob(null)
  }, [jobs, pendingJob])

  const subtitles = useMemo(() => track.length ? URL.createObjectURL(new Blob([vtt(track)], { type: 'text/vtt' })) : '', [track])
  useEffect(() => () => { if (subtitles) URL.revokeObjectURL(subtitles) }, [subtitles])

  const dubSource = result ? api.dubArtifact(dub!.id, result.outputs.video ? 'video' : 'mix', version) : ''
  const src = mode === 'dub' ? dubSource : source ? api.media(source.id) : ''

  /** Cambia de pista conservando la posición (el elemento recarga su src). */
  function switchMode(next: Mode, range?: Range) {
    const media = player.ref.current
    if (next === mode) { if (range) player.playRange(range); return }
    resume.current = { time: range?.start ?? media?.currentTime ?? 0, play: !!range || !!(media && !media.paused), range }
    setMode(next)
  }
  function onLoaded() {
    const pending = resume.current
    if (!pending) return
    resume.current = null
    if (pending.range) player.playRange(pending.range)
    else player.seek(pending.time, pending.play)
  }

  async function launch() {
    if (!source) return
    setBusy(true); setError(null)
    try { const created = await api.createDub(source.id, options); onSelect(created.id) }
    catch (e) { setError(String(e)) }
    finally { setBusy(false) }
  }

  const canRegenerate = !!cue && !!text.trim() && text.trim() !== cue.target_text && !regenerating
  async function regenerate() {
    if (!dub || !cue || !canRegenerate) return
    setError(null)
    try { setPendingJob((await api.regenerate(dub.id, cue.index, text.trim())).id) }
    catch (e) { setError(String(e)) }
  }
  async function rate(score: number) {
    if (!dub || !cue) return
    try { setRatings(await api.rate(dub.id, cue.index, score)) } catch (e) { setError(String(e)) }
  }

  const shortcuts = useRef({ regenerate, range, player })
  shortcuts.current = { regenerate, range, player }
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const { regenerate, range, player } = shortcuts.current
      if (e.key === 'Escape') { e.preventDefault(); player.toggle(range) }
      else if (e.key === 'F1') { e.preventDefault(); player.skip(-SKIP_SECONDS) }
      else if (e.key === 'F2') { e.preventDefault(); player.skip(SKIP_SECONDS) }
      else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); void regenerate() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const playing = track.findIndex(c => player.time >= c.start && player.time < c.start + c.generated_seconds)
  const listRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const list = listRef.current, row = list?.children[playing] as HTMLElement | undefined
    if (list && row && (row.offsetTop < list.scrollTop || row.offsetTop > list.scrollTop + list.clientHeight - row.offsetHeight)) {
      list.scrollTo({ top: row.offsetTop - list.clientHeight / 3, behavior: 'smooth' })
    }
  }, [playing])

  const speakers = Object.keys(result?.references ?? {}).length || 1
  const gates = result?.gates ?? []
  const judged = gates.filter(g => g.passed !== null)
  const rating = cue ? ratings?.ratings[String(cue.index)]?.score : undefined

  return <>
    <details className={`${ui.panelBare} group py-[14px] px-[20px] mb-[22px]`}>
      <summary className="cursor-pointer text-[13px] font-semibold text-[#c9d4e4] list-none before:content-['ⓘ__'] group-open:mb-[10px]">¿Cómo funciona el doblaje?</summary>
      <p className="text-[13px] leading-[1.7] text-[#aebcd0] m-0">A partir de una transcripción: separa la voz del fondo, <b className={ui.strong}>clona la voz</b> del
        hablante con unos segundos de referencia, traduce cada subtítulo eligiendo la versión que <b className={ui.strong}>cabe en su tiempo</b> y
        sintetiza la voz con la duración exacta. Después mezcla con el fondo original y mide la calidad: se vuelve a transcribir
        el doblaje para comprobar que se entiende, se compara la voz con la original y se verifica la sincronía. Clona solo voces con permiso.</p>
    </details>

    {error && <div role="alert" className={ui.errorBanner}>{error}<button className={ui.errorClose} onClick={() => setError(null)} aria-label="Cerrar error">✕</button></div>}

    {!sources.length ? <section className={ui.panel}><p className={ui.desc}>Aún no hay transcripciones terminadas. Transcribe un video primero: el doblaje parte de sus subtítulos.</p></section> : <>
      <section className={`${ui.panel} mb-[20px]`} aria-labelledby="dub-new">
        <div className={ui.panelHead}><h2 id="dub-new" className={ui.panelTitle}>1 <span className={ui.panelTitleSub}>· Nuevo doblaje</span></h2><span className={ui.step}>OMNIVOICE · NLLB · DEMUCS</span></div>
        <div className="flex gap-[12px] flex-wrap items-end">
          <label className="text-[11px] text-[#96a8be] font-semibold flex flex-col gap-[6px]">Transcripción de origen
            <select className={SELECT} value={source?.id ?? ''} onChange={e => onSelect(e.target.value)}>
              {sources.map(j => <option key={j.id} value={j.id}>{j.original_name} · {j.model_id} · {j.transcript?.language.toUpperCase()}</option>)}</select></label>
          <label className="text-[11px] text-[#96a8be] font-semibold flex flex-col gap-[6px]">Idioma del doblaje
            <select className={SELECT} value={options.target_language} onChange={e => setOptions({ ...options, target_language: e.target.value })}>
              {LANGUAGES.map(l => <option key={l.code} value={l.code}>{l.title}</option>)}</select></label>
          <label className="text-[11px] text-[#96a8be] font-semibold flex flex-col gap-[6px]">Calidad
            <select className={SELECT} value={options.num_step} onChange={e => setOptions({ ...options, num_step: Number(e.target.value) })}>
              {PRESETS.map(p => <option key={p.steps} value={p.steps}>{p.title}</option>)}</select></label>
          <label className="text-[12px] text-[#aebcd0] flex items-center gap-[7px] py-[9px]"><input type="checkbox" checked={!!options.separator}
            onChange={e => setOptions({ ...options, separator: e.target.checked ? 'htdemucs' : null })} /> Separar voz y fondo</label>
          <label className="text-[11px] text-[#96a8be] font-semibold flex flex-col gap-[6px]">Motor de voz
            <select className={SELECT} value={options.tts_engine} onChange={e => setOptions({ ...options, tts_engine: e.target.value })}>
              {(engines.filter(c => c.kind === 'motor TTS').length ? engines.filter(c => c.kind === 'motor TTS') : [{ id: 'omnivoice', kind: '', description: '', license: '' }])
                .map(c => <option key={c.id} value={c.id} title={`${c.description} · ${c.license}`}>{c.id}{c.license ? ` · ${c.license}` : ''}</option>)}</select></label>
          <label className="text-[11px] text-[#96a8be] font-semibold flex flex-col gap-[6px]">Traductor
            <select className={SELECT} value={options.translator} onChange={e => setOptions({ ...options, translator: e.target.value })}>
              {(engines.filter(c => c.kind === 'traductor').length ? engines.filter(c => c.kind === 'traductor') : [{ id: 'nllb-600m', kind: '', description: '', license: '' }])
                .map(c => <option key={c.id} value={c.id} title={c.description}>{c.id}</option>)}</select></label>
          <label className="text-[12px] text-[#aebcd0] flex items-center gap-[7px] py-[9px]"><input type="checkbox" checked={options.evaluate}
            onChange={e => setOptions({ ...options, evaluate: e.target.checked })} /> Medir calidad</label>
          <label className="text-[12px] text-[#aebcd0] flex items-center gap-[7px] py-[9px]" title="Detecta a cada hablante y clona su voz por separado"><input type="checkbox" checked={options.diarize}
            onChange={e => setOptions({ ...options, diarize: e.target.checked })} /> Varios hablantes</label>
          <button className={`${ui.primaryBtn} w-auto ml-auto min-w-[200px]`} disabled={!source || busy} onClick={launch}>{busy ? 'Enviando…' : '◐ Doblar'}</button>
        </div>
        {dubsOfSource.length > 0 && <div className="flex gap-[8px] flex-wrap mt-[16px]" role="list" aria-label="Doblajes de esta transcripción">
          {dubsOfSource.map(d => <button key={d.id} role="listitem" onClick={() => onSelect(d.id)}
            className={`${ui.ghostBtn} ${d.id === dub?.id ? 'border-accent text-white' : ''}`}>
            {String(d.params.target_language ?? '').toUpperCase()} · {String(d.params.num_step)} pasos · {d.status === 'completed' ? new Date(d.created_at).toLocaleTimeString() : d.status === 'failed' ? 'error' : `${Math.round((d.progress ?? 0) * 100)} %`}</button>)}
        </div>}
      </section>

      {dub && dub.status === 'failed' && <section className={ui.panel}><p className={ui.errorText}>{dub.error}</p></section>}
      {dub && (dub.status === 'queued' || dub.status === 'running') && <section className={ui.panel} aria-live="polite">
        <div className="flex justify-between text-[13px] text-[#c9d4e4] mb-[10px]"><span>{STAGE_LABEL[dub.stage ?? 'inicio'] ?? dub.stage ?? 'En cola'}…</span><span>{Math.round((dub.progress ?? 0) * 100)} %</span></div>
        <div className="h-[8px] bg-[#222c40] rounded-[10px] overflow-hidden"><div className="h-full bg-[linear-gradient(90deg,#6d73f9,#5cbbd7)] transition-[width]" style={{ width: `${(dub.progress ?? 0) * 100}%` }} /></div>
        <p className={ui.hint}>La primera vez se descargan los modelos (OmniVoice, NLLB, Demucs y los evaluadores): puede tardar varios minutos.</p>
      </section>}

      {result && dub && source && <>
        <section className={`${ui.panel} mb-[20px]`} aria-labelledby="dub-summary">
          <div className={ui.panelHead}><h2 id="dub-summary" className={ui.panelTitle}>2 <span className={ui.panelTitleSub}>· Resultado y validación</span></h2>
            <span className={`${ui.step} ${judged.length && judged.every(g => g.passed) ? 'text-ok' : ''}`}>{judged.filter(g => g.passed).length} / {judged.length} UMBRALES</span></div>
          <dl className="grid grid-cols-[repeat(6,minmax(0,1fr))] gap-[8px] m-0 upto900:grid-cols-[repeat(3,minmax(0,1fr))]">
            <div className={COUNT}><dt className={COUNT_LABEL}>Subtítulos · hablantes</dt><dd className={COUNT_VALUE}>{result.sync.cues} · {speakers}</dd></div>
            <div className={COUNT}><dt className={COUNT_LABEL}>Error medio de duración</dt><dd className={COUNT_VALUE}>{result.sync.mean_abs_error_ms} ms</dd></div>
            <div className={COUNT}><dt className={COUNT_LABEL}>Velocidad de habla</dt><dd className={COUNT_VALUE}>{result.sync.speed_min}–{result.sync.speed_max}×</dd></div>
            <div className={COUNT}><dt className={COUNT_LABEL}>Sonoridad</dt><dd className={COUNT_VALUE}>{result.quality?.lufs ?? result.outputs.lufs ?? 'N/D'} LUFS</dd></div>
            <div className={COUNT}><dt className={COUNT_LABEL}>UTMOS doblaje / original</dt><dd className={COUNT_VALUE}>{result.quality?.mos?.toFixed(2) ?? 'N/D'} / {result.quality?.mos_original?.toFixed(2) ?? 'N/D'}</dd></div>
            <div className={COUNT}><dt className={COUNT_LABEL}>Valoración humana</dt><dd className={COUNT_VALUE}>{ratings?.mean != null ? `${ratings.mean.toFixed(2)} / 5 (${ratings.count})` : 'Sin valorar'}</dd></div>
          </dl>
          {gates.length > 0 && <ul className="grid grid-cols-[repeat(3,minmax(0,1fr))] gap-[6px] mt-[14px] mb-0 p-0 list-none upto900:grid-cols-[1fr]">
            {gates.map((g: Gate) => <li key={g.name} className="flex justify-between gap-[10px] text-[12px] py-[7px] px-[10px] rounded-[6px] bg-[#131d2c]">
              <span className="text-[#aebcd0]">{GATES[g.name]?.label ?? g.name}</span>
              <span className={g.passed === null ? 'text-[#7185a1]' : g.passed ? 'text-ok' : 'text-bad'}>
                {g.value === null ? 'N/D' : (GATES[g.name]?.format ?? String)(g.value)} {g.passed === null ? '' : g.passed ? '✓' : '✗'}
                <small className="text-[#7185a1] ml-[6px]">{g.higher_is_better ? '≥' : '≤'} {(GATES[g.name]?.format ?? String)(g.threshold)}</small></span></li>)}
          </ul>}
          <div className="flex gap-[14px] flex-wrap mt-[14px] text-[12px]">
            {result.outputs.video && <a className="text-[#b5b6ff] font-bold no-underline" href={api.dubArtifact(dub.id, 'video', version)} download>↓ Video doblado (MP4, 2 pistas)</a>}
            <a className="text-[#b5b6ff] font-bold no-underline" href={api.dubArtifact(dub.id, 'mix', version)} download>↓ Audio mezclado (FLAC)</a>
            <a className="text-[#b5b6ff] font-bold no-underline" href={api.dubArtifact(dub.id, 'subtitles', version)} download>↓ Subtítulos traducidos (SRT)</a>
          </div>
        </section>

        <div className="grid grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)] gap-[20px] items-start upto1100:grid-cols-[1fr]">
          <section className={`${BENCH} overflow-hidden`} aria-labelledby="dub-listen">
            <div className={BENCH_HEAD}><h2 id="dub-listen" className={ui.panelTitle}>3 <span className={ui.panelTitleSub}>· Escucha y compara</span></h2>
              <div className="flex rounded-[7px] overflow-hidden border border-control-line" role="group" aria-label="Pista de audio">
                {(['dub', 'original'] as Mode[]).map(m => <button key={m} type="button" aria-pressed={mode === m} onClick={() => switchMode(m)}
                  className={`${TOGGLE} ${mode === m ? 'bg-accent text-white' : 'bg-control text-control-text'}`}>{m === 'dub' ? 'Doblaje' : 'Original'}</button>)}
              </div></div>
            <div className="bg-[#05080e] border border-panel-line rounded-[10px] overflow-hidden grid place-items-center min-h-[120px]">
              <video ref={player.ref} controls preload="metadata" src={src} onLoadedMetadata={onLoaded}
                className="w-full block bg-black max-h-[38vh] short:max-h-[30vh] [&::cue]:bg-[#000000c0] [&::cue]:text-[15px]">
                {mode === 'dub' && subtitles && <track kind="subtitles" src={subtitles} default srcLang={String(dub.params.target_language)} label="Doblaje" />}
              </video>
            </div>
            <div className="flex-1 min-h-[120px] mt-[12px] overflow-auto border border-[#283347] rounded-[8px] upto1100:max-h-[340px]" ref={listRef} role="listbox" aria-label="Subtítulos doblados">
              {track.map((c, i) => {
                const tone = cueTone(c, quality.get(c.index)?.cer)
                return <button type="button" role="option" aria-selected={i === selected} key={c.index}
                  onClick={() => { setSelected(i); player.seek(c.start) }}
                  className={`block w-full text-left text-inherit cursor-pointer py-[10px] px-[14px] border-0 border-b border-b-[#263144] border-l-[3px] last:border-b-0 ${i === selected ? 'border-l-[#8b8cff] bg-[#25264d]' : i === playing ? 'border-l-ok bg-[#13231f]' : 'border-l-transparent bg-transparent hover:bg-field'}`}>
                  <span className="flex items-center gap-[8px] font-[monospace] text-[10px] text-[#9899ff]">
                    <i className={`inline-block w-[7px] h-[7px] rounded-[50%] ${DOT[tone]}`} aria-label={tone === 'good' ? 'bien' : tone === 'warn' ? 'revisar' : 'problema'} />
                    {clock(c.start)} · {c.generated_seconds.toFixed(1)} s{c.speed > 1.001 && ` · ${c.speed.toFixed(2)}×`}
                    {speakers > 1 && <b className="ml-auto font-sans text-[10px] text-[#c3cdd9] bg-[#1d2839] rounded-[4px] px-[6px] py-[1px]">{c.speaker}</b>}</span>
                  <p className="text-[13px] leading-[1.5] mt-[5px] mx-0 mb-0 text-[#d7dfed]">{c.target_text}</p>
                  <p className="text-[11px] leading-[1.4] mt-[3px] mx-0 mb-0 text-[#7f91aa]">{c.source_text}</p>
                </button>
              })}
            </div>
          </section>

          <section className={`${BENCH} overflow-y-auto`} aria-labelledby="dub-edit">
            <div className={BENCH_HEAD}><h2 id="dub-edit" className={ui.panelTitle}>4 <span className={ui.panelTitleSub}>· Revisa el subtítulo</span></h2>
              {cue && <span className={ui.step}>#{cue.index + 1} · {clock(cue.start)}</span>}</div>
            {!cue ? <p className={ui.desc}>Selecciona un subtítulo de la lista.</p> : <>
              <div className="flex gap-[8px] flex-wrap mb-[12px]" role="group" aria-label="Escuchar">
                <button type="button" className={`${TRANSPORT_BTN} flex-1`} onClick={() => switchMode('original', range!)} title="Fragmento original">▶ Original</button>
                <button type="button" className={`${TRANSPORT_BTN} flex-1`} onClick={() => switchMode('dub', range!)} title="Fragmento doblado (Esc)">▶ Doblaje</button>
                <button type="button" className={TRANSPORT_BTN} onClick={() => void new Audio(api.dubCueAudio(dub.id, cue.index, version)).play()} title="Solo la voz sintetizada, sin fondo">Solo voz</button>
              </div>
              <p className="text-[12px] text-[#9fb0c6] m-0 mb-[8px] leading-[1.5]"><span className="text-[10px] tracking-[1px] text-[#7185a1] block mb-[3px]">ORIGINAL</span>{cue.source_text}</p>
              <label htmlFor="dub-text" className="text-[10px] tracking-[1px] text-[#7185a1] block mb-[4px]">TRADUCCIÓN (EDITABLE)</label>
              <textarea id="dub-text" value={text} onChange={e => setText(e.target.value)} rows={3}
                className="w-full m-0 p-[12px] text-[15px] leading-[1.6] border border-[#33425a] rounded-[9px] bg-[#0d1523] text-[#f0f3fa] resize-y outline-none focus:border-accent" />
              {cue.candidates.length > 1 && <div className="flex gap-[6px] flex-wrap mt-[8px]" aria-label="Otras traducciones">
                {cue.candidates.map(c => <button key={c} type="button" onClick={() => setText(c)} title="Usar esta traducción"
                  className={`text-[11px] py-[5px] px-[8px] rounded-[6px] border cursor-pointer text-left ${c === text ? 'border-accent bg-[#25264d] text-white' : 'border-[#2c3b50] bg-[#131d2c] text-[#aebcd0]'}`}>{c}</button>)}
              </div>}
              <button className={`${ui.primaryBtn} mt-[12px]`} disabled={!canRegenerate} onClick={regenerate}>
                {regenerating ? 'Regenerando en la GPU…' : '↻ Regenerar este subtítulo'} <span className="ml-auto text-[10px] font-medium opacity-85"><kbd className={ui.kbdOnPrimary}>Ctrl</kbd>+<kbd className={ui.kbdOnPrimary}>Enter</kbd></span></button>

              <dl className="grid grid-cols-[repeat(4,minmax(0,1fr))] gap-[8px] mt-[14px] mx-0 mb-0">
                <div className={COUNT}><dt className={COUNT_LABEL}>Ranura</dt><dd className={COUNT_VALUE}>{seconds(cue.slot_seconds)}</dd></div>
                <div className={COUNT}><dt className={COUNT_LABEL}>Natural</dt><dd className={COUNT_VALUE}>{seconds(cue.natural_seconds)}</dd></div>
                <div className={COUNT}><dt className={COUNT_LABEL}>Generada</dt><dd className={COUNT_VALUE}>{seconds(cue.generated_seconds)}</dd></div>
                <div className={COUNT}><dt className={COUNT_LABEL}>Velocidad</dt><dd className={`${COUNT_VALUE} ${cue.action === 'shorten' ? 'text-bad' : ''}`}>{cue.speed.toFixed(2)}×</dd></div>
              </dl>
              {!!cue.attempts && <p className="text-[12px] text-[#9fb0c6] mt-[8px] mb-0">Regenerado automáticamente {cue.attempts} {cue.attempts === 1 ? 'vez' : 'veces'} porque la primera versión no se entendía bien.</p>}
              {cue.action === 'shorten' && <p className="text-[12px] text-warn mt-[8px] mb-0">No cabe ni a la velocidad máxima: prueba una traducción más corta.</p>}
              {cueQuality && <dl className="grid grid-cols-[repeat(3,minmax(0,1fr))] gap-[8px] mt-[8px] mx-0 mb-0">
                <div className={COUNT}><dt className={COUNT_LABEL}>CER ida y vuelta</dt><dd className={COUNT_VALUE}>{cueQuality.cer == null ? 'N/D' : pct(cueQuality.cer)}</dd></div>
                <div className={COUNT}><dt className={COUNT_LABEL}>Parecido con el original</dt><dd className={COUNT_VALUE}>{cueQuality.similarity_slot?.toFixed(2) ?? cueQuality.similarity?.toFixed(2) ?? 'N/D'}</dd></div>
                <div className={COUNT}><dt className={COUNT_LABEL}>UTMOS</dt><dd className={COUNT_VALUE}>{cueQuality.mos?.toFixed(2) ?? 'N/D'}</dd></div>
              </dl>}
              {cueQuality && <p className="text-[12px] text-[#9fb0c6] mt-[8px] mb-0 leading-[1.5]"><span className="text-[10px] tracking-[1px] text-[#7185a1] block mb-[3px]">LO QUE ENTIENDE EL ASR</span>{cueQuality.asr_text || '—'}</p>}

              <div className="flex items-center gap-[6px] mt-[14px] text-[12px] text-[#aebcd0]" role="group" aria-label="Valoración humana de 1 a 5">Tu valoración:
                {[1, 2, 3, 4, 5].map(n => <button key={n} type="button" aria-pressed={rating === n} onClick={() => void rate(n)}
                  className={`w-[30px] h-[30px] rounded-[6px] border cursor-pointer ${rating === n ? 'bg-accent border-accent text-white' : 'bg-control border-control-line text-control-text'}`}>{n}</button>)}
              </div>
            </>}
            <p className="mt-auto mx-0 mb-0 pt-[12px] text-[11px] text-[#7f91aa]" aria-label="Atajos de teclado"><kbd className={ui.kbd}>Esc</kbd> reproducir/pausar · <kbd className={ui.kbd}>F1</kbd> −3 s · <kbd className={ui.kbd}>F2</kbd> +3 s · <kbd className={ui.kbd}>Ctrl</kbd>+<kbd className={ui.kbd}>Enter</kbd> regenerar</p>
          </section>
        </div>
      </>}
    </>}
  </>
}
