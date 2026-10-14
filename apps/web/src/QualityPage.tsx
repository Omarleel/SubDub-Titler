import { useEffect, useRef, useState } from 'react'
import { api, type Evaluation, type Job } from './api'
import { clock, CueList, MediaPlayer, usePlayer } from './media'

const MAX_CHARS = 6000
const SKIP_SECONDS = 3
const RATES = [0.5, 0.75, 1]
const LANGUAGES = [{ code: 'es', title: 'Español' }, { code: 'en', title: 'Inglés' },
  { code: 'ja', title: 'Japonés' }, { code: 'zh', title: 'Chino' }]

const percent = (rate: number) => `${(rate * 100).toFixed(1)} %`

function verdict(rate: number) {
  if (rate <= 0.05) return { label: 'Excelente', tone: 'good' }
  if (rate <= 0.10) return { label: 'Muy buena', tone: 'good' }
  if (rate <= 0.20) return { label: 'Aceptable: revisa antes de publicar', tone: 'warn' }
  return { label: 'Baja: necesita corrección manual', tone: 'bad' }
}

type Props = { jobs: Job[]; job: Job | undefined; onSelect: (id: string) => void }

export function QualityPage({ jobs, job, onSelect }: Props) {
  const completed = jobs.filter(j => j.status === 'completed' && j.transcript)
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
    <details className="panel explain">
      <summary>¿Qué es esto y cómo se usa?</summary>
      <p>Mide <b>qué tan exacta</b> fue la transcripción automática comparándola {units === 'palabras' ? 'palabra por palabra' : 'carácter por carácter'} con
        lo que <b>realmente</b> se dice. 1) Marca un fragmento de 1–3 min con dos clics en la lista de subtítulos.
        2) El texto automático aparece en el editor: escucha el fragmento y corrige los errores. 3) Pulsa «Medir precisión».
        Úsalo para decidir qué modelo o configuración funciona mejor con tus videos.</p>
    </details>

    {error && <div role="alert" className="error">{error}<button onClick={() => setError(null)} aria-label="Cerrar error">✕</button></div>}

    {!completed.length ? <section className="panel"><p className="panel-desc">Aún no hay transcripciones terminadas. Transcribe un video primero.</p></section> :
    <div className="workbench">
      <section className="panel bench-media" aria-labelledby="step1">
        <div className="panel-head"><h2 id="step1">1 <span>· Marca el fragmento</span></h2>
          <div className="inline-fields">
            <select value={transcript ? job!.id : ''} onChange={e => onSelect(e.target.value)} aria-label="Transcripción a evaluar">
              {!transcript && <option value="" disabled>Selecciona una transcripción…</option>}
              {completed.map(j => <option key={j.id} value={j.id}>{j.original_name} · {j.model_id}</option>)}
            </select>
            <select value={language} onChange={e => setLanguage(e.target.value)} aria-label="Idioma del audio">
              {LANGUAGES.map(l => <option key={l.code} value={l.code}>{l.title}</option>)}</select>
          </div>
        </div>
        {!transcript || !job ? <p className="panel-desc">Selecciona una transcripción para empezar.</p> : <>
          <MediaPlayer job={job} player={player} />
          <p className={`selection-status ${pending ? 'waiting' : ''}`} aria-live="polite">{status}
            {range && <button type="button" className="link-btn" onClick={() => { setSelection(null); setAnchor(null); setPending(false) }}>Limpiar</button>}</p>
          <CueList segments={segments} time={player.time} selection={selection} onCueClick={pick} />
        </>}
      </section>

      <section className="panel bench-editor" aria-labelledby="step2">
        <div className="panel-head"><h2 id="step2">2 <span>· Corrige el texto</span></h2>
          {range && <span className="step">{clock(range.start)} – {clock(range.end)}</span>}</div>

        <div className="transport" role="group" aria-label="Controles de reproducción">
          <button type="button" className="ghost-btn" disabled={!range} onClick={() => player.skip(-SKIP_SECONDS)} title="Retroceder 3 s (F1)">⟲ 3 s</button>
          <button type="button" className="ghost-btn play" disabled={!range} onClick={() => player.toggle(range)} title="Reproducir o pausar el fragmento (Esc)">
            {player.playing ? '❚❚ Pausa' : '▶ Reproducir fragmento'}</button>
          <button type="button" className="ghost-btn" disabled={!range} onClick={() => player.skip(SKIP_SECONDS)} title="Avanzar 3 s (F2)">3 s ⟳</button>
          <label className="rate">Velocidad
            <select value={player.rate} onChange={e => player.setRate(Number(e.target.value))}>
              {RATES.map(r => <option key={r} value={r}>{r}×</option>)}</select></label>
        </div>

        <label htmlFor="reference" className="sr-only">Texto correcto del fragmento</label>
        <textarea id="reference" ref={editor} value={reference} onChange={e => { setReference(e.target.value); setEvaluation(null) }}
          disabled={!ready} aria-describedby="editor-help"
          placeholder={ready ? 'Escribe exactamente lo que dice el hablante…' : 'Marca un fragmento en el paso 1: su texto automático aparecerá aquí para que lo corrijas.'} />

        <div className="editor-meta" id="editor-help">
          <span className={tooLong ? 'error-text' : ''}>{reference.length} / {MAX_CHARS} caracteres{tooLong && ' · reduce el fragmento'}</span>
          <span>
            <button type="button" className="link-btn" disabled={!hypothesis || reference === hypothesis}
              onClick={() => { setReference(hypothesis); prefilled.current = hypothesis; editor.current?.focus() }}>Restaurar texto automático</button>
            <button type="button" className="link-btn" disabled={!reference}
              onClick={() => { setReference(''); editor.current?.focus() }}>Vaciar</button>
          </span>
        </div>

        <button className="primary-btn" disabled={!canRun} onClick={run}>
          {busy ? 'Comparando…' : '◎ Medir precisión'} <span><kbd>Ctrl</kbd>+<kbd>Enter</kbd></span></button>

        <div aria-live="polite">
          {evaluation && result && <div className="result-compact">
            <div className="result-score">
              <div className="quality-number">{percent(Math.max(0, 1 - (result.rate ?? 0)))}<span>de precisión</span></div>
              <span className={`verdict ${verdict(result.rate ?? 0).tone}`}>{verdict(result.rate ?? 0).label}</span>
            </div>
            <dl className="result-counts">
              <div><dt>{units} del texto correcto</dt><dd>{result.reference_units}</dd></div>
              <div><dt>Mal reconocidas</dt><dd>{result.substitutions}</dd></div>
              <div><dt>Omitidas</dt><dd>{result.deletions}</dd></div>
              <div><dt>Añadidas de más</dt><dd>{result.insertions}</dd></div>
            </dl>
            <p className="hint">Error {evaluation.primary_metric.toUpperCase()} {percent(result.rate ?? 0)} ignorando mayúsculas y puntuación
              ({percent(evaluation.raw[evaluation.primary_metric]?.rate ?? 0)} contándolas). Menos error es mejor.</p>
          </div>}
        </div>

        <p className="shortcuts" aria-label="Atajos de teclado"><kbd>Esc</kbd> reproducir/pausar · <kbd>F1</kbd> −3 s · <kbd>F2</kbd> +3 s · <kbd>Ctrl</kbd>+<kbd>Enter</kbd> medir</p>
      </section>
    </div>}
  </>
}
