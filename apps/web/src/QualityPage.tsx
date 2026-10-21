import { useEffect, useRef, useState } from 'react'
import { api, type Evaluation, type Job } from './api'
import { clock, CueList, MediaPlayer, usePlayer } from './media'
import * as ui from './ui'

const MAX_CHARS = 6000
const SKIP_SECONDS = 3
const RATES = [0.5, 0.75, 1]
const LANGUAGES = [{ code: 'es', title: 'Español' }, { code: 'en', title: 'Inglés' },
  { code: 'ja', title: 'Japonés' }, { code: 'zh', title: 'Chino' }]

const BENCH = `${ui.panelBare} sticky top-[12px] h-[calc(100vh-24px)] min-h-[520px] flex flex-col py-[18px] px-[20px] upto1100:static upto1100:h-auto upto1100:min-h-0`
const BENCH_HEAD = 'flex items-center justify-between mb-[12px] flex-wrap gap-[10px]'
const INLINE_SELECT = 'bg-field border border-field-line rounded-[7px] py-[9px] px-[10px] text-field-text max-w-[340px]'
const TRANSPORT_BTN = `${ui.ghostBtn} disabled:opacity-45 disabled:cursor-not-allowed`
const COUNT = 'bg-[#172133] rounded-[7px] p-[9px]'
const COUNT_LABEL = 'text-[9px] tracking-[.6px] uppercase text-[#7185a1]'
const COUNT_VALUE = 'mt-[5px] mx-0 mb-0 font-score font-semibold text-[18px] text-ink'

const percent = (rate: number) => `${(rate * 100).toFixed(1)} %`

function verdict(rate: number): { label: string; tone: keyof typeof ui.TONES } {
  if (rate <= 0.05) return { label: 'Excelente', tone: 'good' }
  if (rate <= 0.10) return { label: 'Muy buena', tone: 'good' }
  if (rate <= 0.20) return { label: 'Aceptable: revisa antes de publicar', tone: 'warn' }
  return { label: 'Baja: necesita corrección manual', tone: 'bad' }
}

type Props = { jobs: Job[]; job: Job | undefined; onSelect: (id: string) => void }

export function QualityPage({ jobs, job, onSelect }: Props) {
  const completed = jobs.filter(j => j.status === 'completed')
  const transcript = job?.status === 'completed' ? job.transcript : null
  const segments = transcript?.segments ?? []
  const player = usePlayer()
  const editor = useRef<HTMLTextAreaElement>(null)
  const prefilled = useRef('')
  // anchor: inicio del rango; pending: falta el clic del último subtítulo.
  const [anchor, setAnchor] = useState<number | null>(null)
  const [pending, setPending] = useState(false)
  const [selection, setSelection] = useState<[number, number] | null>(null)
  const [language, setLanguage] = useState('es')
  const [reference, setReference] = useState('')
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null)
  const [busy, setBusy] = useState(false)
  const [focusTick, setFocusTick] = useState(0)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setEvaluation(null); setReference(''); setError(null)
    setAnchor(null); setPending(false); setSelection(null); prefilled.current = ''
    if (transcript) setLanguage(LANGUAGES.some(l => l.code === transcript.language) ? transcript.language : 'es')
  }, [job?.id])

  const chosen = selection ? segments.slice(selection[0], selection[1] + 1) : []
  const hypothesis = chosen.map(s => s.text.trim()).join(' ')
  const range = chosen.length ? { start: chosen[0].start, end: chosen[chosen.length - 1].end } : null
  const ready = !!range && !pending

  // Al cerrar la selección: precargar el texto automático (si el usuario no escribió el suyo) y enfocar.
  useEffect(() => {
    if (!ready) return
    if (!reference.trim() || reference === prefilled.current) {
      setReference(hypothesis); prefilled.current = hypothesis
    }
    player.seek(range!.start)  // Esc / Reproducir empiezan desde el inicio del fragmento
    setFocusTick(t => t + 1)  // mismo lote que setReference: el efecto de foco ve el texto ya aplicado
  }, [ready, hypothesis])

  // Tras aplicar el texto al DOM: cursor al principio, listo para corregir desde la primera palabra.
  useEffect(() => {
    if (!focusTick || !editor.current) return
    editor.current.focus()
    editor.current.setSelectionRange(0, 0)
    editor.current.scrollTop = 0
  }, [focusTick])

  function pick(index: number, event: React.MouseEvent) {
    player.seek(segments[index].start)
    setEvaluation(null)
    // Clic 1 = inicio, clic 2 = final; Mayús+clic ajusta el final; un clic más empieza otro rango.
    if (anchor != null && (pending || event.shiftKey) && index !== anchor) {
      setSelection([Math.min(anchor, index), Math.max(anchor, index)])
      setPending(false)
    } else {
      setAnchor(index); setSelection([index, index]); setPending(true)
    }
  }

  const tooLong = reference.length > MAX_CHARS || hypothesis.length > MAX_CHARS
  const canRun = !!reference.trim() && !!hypothesis && !tooLong && !busy

  async function run() {
    if (!canRun) return
    setBusy(true); setError(null)
    try { setEvaluation(await api.evaluate(reference, hypothesis, language)) }
    catch (e) { setError(String(e)) }
    finally { setBusy(false) }
  }

  // Atajos al estilo de las herramientas de transcripción: funcionan mientras se escribe.
  const shortcuts = useRef({ run, range, player })
  shortcuts.current = { run, range, player }
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const { run, range, player } = shortcuts.current
      if (e.key === 'Escape') { e.preventDefault(); player.toggle(range) }
      else if (e.key === 'F1') { e.preventDefault(); player.skip(-SKIP_SECONDS) }
      else if (e.key === 'F2') { e.preventDefault(); player.skip(SKIP_SECONDS) }
      else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); void run() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const units = language === 'ja' || language === 'zh' ? 'caracteres' : 'palabras'
  const result = evaluation?.normalized[evaluation.primary_metric]
  const status = !range ? 'Haz clic en el primer subtítulo del fragmento.'
    : pending ? `Inicio en ${clock(range.start)}. Ahora haz clic en el último subtítulo.`
    : `Fragmento ${clock(range.start)} – ${clock(range.end)}: ${chosen.length} subtítulos, ${hypothesis.length} caracteres.`

  return <>
    <details className={`${ui.panelBare} group py-[14px] px-[20px] mb-[22px]`}>
      <summary className="cursor-pointer text-[13px] font-semibold text-[#c9d4e4] list-none before:content-['ⓘ__'] group-open:mb-[10px]">¿Qué es esto y cómo se usa?</summary>
      <p className="text-[13px] leading-[1.7] text-[#aebcd0] m-0">Mide <b className={ui.strong}>qué tan exacta</b> fue la transcripción automática comparándola {units === 'palabras' ? 'palabra por palabra' : 'carácter por carácter'} con
        lo que <b className={ui.strong}>realmente</b> se dice. 1) Marca un fragmento de 1–3 min con dos clics en la lista de subtítulos.
        2) El texto automático aparece en el editor: escucha el fragmento y corrige los errores. 3) Pulsa «Medir precisión».
        Úsalo para decidir qué modelo o configuración funciona mejor con tus videos.</p>
    </details>

    {error && <div role="alert" className={ui.errorBanner}>{error}<button className={ui.errorClose} onClick={() => setError(null)} aria-label="Cerrar error">✕</button></div>}

    {!completed.length ? <section className={ui.panel}><p className={ui.desc}>Aún no hay transcripciones terminadas. Transcribe un video primero.</p></section> :
    <div className="grid grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)] gap-[20px] items-start upto1100:grid-cols-[1fr]">
      <section className={`${BENCH} overflow-hidden`} aria-labelledby="step1">
        <div className={BENCH_HEAD}><h2 id="step1" className={ui.panelTitle}>1 <span className={ui.panelTitleSub}>· Marca el fragmento</span></h2>
          <div className="flex gap-[10px]">
            <select className={INLINE_SELECT} value={transcript ? job!.id : ''} onChange={e => onSelect(e.target.value)} aria-label="Transcripción a evaluar">
              {!transcript && <option value="" disabled>Selecciona una transcripción…</option>}
              {completed.map(j => <option key={j.id} value={j.id}>{j.original_name} · {j.model_id}</option>)}
            </select>
            <select className={INLINE_SELECT} value={language} onChange={e => setLanguage(e.target.value)} aria-label="Idioma del audio">
              {LANGUAGES.map(l => <option key={l.code} value={l.code}>{l.title}</option>)}</select>
          </div>
        </div>
        {!transcript || !job ? <p className={ui.desc}>Selecciona una transcripción para empezar.</p> : <>
          <MediaPlayer job={job} player={player} variant="bench" />
          <p className={`flex justify-between items-center gap-[10px] my-[12px] mx-0 text-[12px] py-[9px] px-[12px] bg-field border rounded-[8px] ${pending ? 'border-[#8a7330] text-warn' : 'border-[#2b3949] text-[#aebcd0]'}`} aria-live="polite">{status}
            {range && <button type="button" className={ui.linkBtn} onClick={() => { setSelection(null); setAnchor(null); setPending(false) }}>Limpiar</button>}</p>
          <CueList segments={segments} time={player.time} selection={selection} onCueClick={pick} variant="bench" />
        </>}
      </section>

      <section className={`${BENCH} overflow-y-auto`} aria-labelledby="step2">
        <div className={BENCH_HEAD}><h2 id="step2" className={ui.panelTitle}>2 <span className={ui.panelTitleSub}>· Corrige el texto</span></h2>
          {range && <span className={ui.step}>{clock(range.start)} – {clock(range.end)}</span>}</div>

        <div className="flex gap-[8px] items-center flex-wrap mb-[12px]" role="group" aria-label="Controles de reproducción">
          <button type="button" className={TRANSPORT_BTN} disabled={!range} onClick={() => player.skip(-SKIP_SECONDS)} title="Retroceder 3 s (F1)">⟲ 3 s</button>
          <button type="button" className={`${TRANSPORT_BTN} flex-1 min-w-[150px]`} disabled={!range} onClick={() => player.toggle(range)} title="Reproducir o pausar el fragmento (Esc)">
            {player.playing ? '❚❚ Pausa' : '▶ Reproducir fragmento'}</button>
          <button type="button" className={TRANSPORT_BTN} disabled={!range} onClick={() => player.skip(SKIP_SECONDS)} title="Avanzar 3 s (F2)">3 s ⟳</button>
          <label className="flex items-center gap-[6px] text-[11px] text-[#96a8be] font-semibold">Velocidad
            <select className="bg-field border border-field-line rounded-[6px] py-[7px] px-[6px] text-field-text" value={player.rate} onChange={e => player.setRate(Number(e.target.value))}>
              {RATES.map(r => <option key={r} value={r}>{r}×</option>)}</select></label>
        </div>

        <label htmlFor="reference" className="sr-only">Texto correcto del fragmento</label>
        <textarea id="reference" ref={editor} value={reference} onChange={e => { setReference(e.target.value); setEvaluation(null) }}
          className="flex-1 min-h-[120px] short:min-h-[100px] w-full m-0 p-[15px] text-[15px] leading-[1.75] border border-[#33425a] rounded-[9px] bg-[#0d1523] text-[#f0f3fa] resize-none outline-none focus:border-accent disabled:opacity-60"
          disabled={!ready} aria-describedby="editor-help"
          placeholder={ready ? 'Escribe exactamente lo que dice el hablante…' : 'Marca un fragmento en el paso 1: su texto automático aparecerá aquí para que lo corrijas.'} />

        <div className="flex justify-between items-center gap-[8px] flex-wrap text-[11px] text-desc mt-[8px] mx-0 mb-[12px]" id="editor-help">
          <span className={tooLong ? ui.errorText : ''}>{reference.length} / {MAX_CHARS} caracteres{tooLong && ' · reduce el fragmento'}</span>
          <span>
            <button type="button" className={ui.linkBtn} disabled={!hypothesis || reference === hypothesis}
              onClick={() => { setReference(hypothesis); prefilled.current = hypothesis; editor.current?.focus() }}>Restaurar texto automático</button>
            <button type="button" className={ui.linkBtn} disabled={!reference}
              onClick={() => { setReference(''); editor.current?.focus() }}>Vaciar</button>
          </span>
        </div>

        <button className={ui.primaryBtn} disabled={!canRun} onClick={run}>
          {busy ? 'Comparando…' : '◎ Medir precisión'} <span className="ml-auto text-[10px] font-medium opacity-85"><kbd className={ui.kbdOnPrimary}>Ctrl</kbd>+<kbd className={ui.kbdOnPrimary}>Enter</kbd></span></button>

        <div aria-live="polite">
          {evaluation && result && <div className="mt-[14px] p-[14px] border border-[#2b3949] rounded-[10px] bg-[#131d2c]">
            <div className="flex items-center justify-between gap-[12px] flex-wrap">
              <div className="font-score font-semibold text-[38px] short:text-[30px] m-0 text-[#bdc0ff]">{percent(Math.max(0, 1 - (result.rate ?? 0)))}<span className="font-dm font-normal text-[12px] text-[#9badc2] inline mt-[7px] ml-[8px]">de precisión</span></div>
              <span className={`inline-block text-[12px] font-bold py-[7px] px-[11px] rounded-[6px] ${ui.TONES[verdict(result.rate ?? 0).tone]}`}>{verdict(result.rate ?? 0).label}</span>
            </div>
            <dl className="grid grid-cols-[repeat(4,1fr)] gap-[8px] mt-[12px] mx-0 mb-0">
              <div className={COUNT}><dt className={COUNT_LABEL}>{units} del texto correcto</dt><dd className={COUNT_VALUE}>{result.reference_units}</dd></div>
              <div className={COUNT}><dt className={COUNT_LABEL}>Mal reconocidas</dt><dd className={COUNT_VALUE}>{result.substitutions}</dd></div>
              <div className={COUNT}><dt className={COUNT_LABEL}>Omitidas</dt><dd className={COUNT_VALUE}>{result.deletions}</dd></div>
              <div className={COUNT}><dt className={COUNT_LABEL}>Añadidas de más</dt><dd className={COUNT_VALUE}>{result.insertions}</dd></div>
            </dl>
            <p className="text-[11px] text-desc leading-[1.6] mt-[10px]">Error {evaluation.primary_metric.toUpperCase()} {percent(result.rate ?? 0)} ignorando mayúsculas y puntuación
              ({percent(evaluation.raw[evaluation.primary_metric]?.rate ?? 0)} contándolas). Menos error es mejor.</p>
          </div>}
        </div>

        <p className="mt-auto mx-0 mb-0 pt-[12px] text-[11px] text-[#7f91aa]" aria-label="Atajos de teclado"><kbd className={ui.kbd}>Esc</kbd> reproducir/pausar · <kbd className={ui.kbd}>F1</kbd> −3 s · <kbd className={ui.kbd}>F2</kbd> +3 s · <kbd className={ui.kbd}>Ctrl</kbd>+<kbd className={ui.kbd}>Enter</kbd> medir</p>
      </section>
    </div>}
  </>
}
