import { useEffect, useState } from 'react'
import { api, type Job, type Model, type ProfileEvent } from './api'
import { CueList, MediaPlayer, usePlayer } from './media'
import { QualityPage } from './QualityPage'
import { ROUTES, useLocation, type Route } from './router'

const PAGES: Record<Route, { icon: string; title: string; subtitle: string }> = {
  transcribir: { icon: '▣', title: 'Estudio de transcripción', subtitle: 'Convierte audio y video en subtítulos con modelos ejecutados en tu equipo.' },
  precision: { icon: '◎', title: 'Medir precisión', subtitle: 'Compara la transcripción automática con el texto correcto y calcula el porcentaje de acierto.' },
  rendimiento: { icon: '▥', title: 'Rendimiento', subtitle: 'Cuánto tardó cada etapa del proceso y qué tan rápido fue frente al tiempo real.' },
}
const STATUS: Record<Job['status'], string> = {
  queued: 'En cola', running: 'Procesando', completed: 'Completado', failed: 'Error',
}
const STAGES: Record<string, string> = {
  'registry.engine_for': 'Preparar motor',
  'asr.transcribe_complete': 'Reconocimiento de voz (total)',
  'asr.model_load': 'Cargar modelo en la GPU',
  'audio.decode': 'Extraer audio del archivo',
  'asr.prepare_decode': 'Detectar voz (VAD) y preparar',
  'asr.consume_generator': 'Transcribir en la GPU',
  'subtitles.resegment': 'Dividir en subtítulos',
  'subtitles.write_srt': 'Guardar archivo SRT',
}
const options = [{ code: 'auto', title: 'Automático' }, { code: 'es', title: 'Español' },
  { code: 'en', title: 'Inglés' }, { code: 'ja', title: 'Japonés' }, { code: 'zh', title: 'Chino' }]

/** Los spans se guardan al cerrar (hijos antes que padre); mostrar padre → hijos. */
function stagesInOrder(events: ProfileEvent[]) {
  const ordered: ProfileEvent[] = []
  let children: ProfileEvent[] = []
  for (const e of events) {
    if (e.depth === 0) { ordered.push(e, ...children); children = [] } else children.push(e)
  }
  return [...ordered, ...children]
}


export function App() {
  const { route, job: jobId, navigate } = useLocation()
  const player = usePlayer()
  const [models, setModels] = useState<Model[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [file, setFile] = useState<File | null>(null)
  const [model, setModel] = useState('small')
  const [language, setLanguage] = useState('auto')
  const [device, setDevice] = useState('cuda')
  const [compute, setCompute] = useState('float16')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.models().then(setModels).catch(e => setError(String(e)))
    const poll = () => api.jobs().then(setJobs).catch(e => setError(String(e)))
    poll()
    const id = window.setInterval(poll, 1800)
    return () => window.clearInterval(id)
  }, [])

  const job = jobs.find(j => j.id === jobId) || jobs[0]
  const page = PAGES[route]
  const transcript = job?.transcript

  async function submit() {
    if (!file) return
    setBusy(true); setError(null)
    try {
      const data = new FormData()
      data.set('file', file)
      data.set('model_id', model)
      data.set('language', language)
      data.set('device', device)
      data.set('compute_type', compute)
      const result = await api.create(data)
      navigate({ job: result.id })
      setJobs(prev => [result, ...prev])
      setFile(null)
    } catch (e) { setError(String(e)) }
    finally { setBusy(false) }
  }

  return <div className="shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-icon">S<span>✦</span></span><div><strong>SubDub</strong><small>TITLER / LOCAL AI</small></div></div>
      <div className="sidenav-label">WORKSPACE</div>
      {(Object.keys(ROUTES) as Route[]).map(r =>
        <a key={r} href={`/${r}${job ? `?trabajo=${job.id}` : ''}`} className={`nav ${route === r ? 'active' : ''}`}
          onClick={e => { e.preventDefault(); navigate({ route: r }) }}><span>{PAGES[r].icon}</span> {ROUTES[r]}</a>)}
      <div className="sidenav-label gap">HISTORIAL</div>
      <div className="history">{jobs.length === 0 ? <p className="muted small">Aún no hay trabajos.</p> : jobs.map(j =>
        <button className={`history-item ${job?.id === j.id ? 'chosen' : ''}`} key={j.id} onClick={() => navigate({ job: j.id })}>
          <span className="ellipsis">{j.original_name}</span><small>{STATUS[j.status]}</small>
        </button>)}</div>
      <div className="sidebar-foot"><div className="online-dot"/> BACKEND LOCAL <small>v0.1.0</small></div>
    </aside>

    <main className="main">
      <header className="topbar"><div className="breadcrumb">SUBDUB <b>/</b> {ROUTES[route]}</div><div className="top-right"><span className="chip">🔒 Offline-ready</span><span className="avatar">SD</span></div></header>
      <div className="page">
        {error && <div role="alert" className="error">{error}<button onClick={() => setError(null)}>✕</button></div>}
        <div className="heading"><div><div className="eyebrow">AUDIO INTELLIGENCE · LOCAL GPU</div><h1>{page.title}</h1><p className="subtitle">{page.subtitle}</p></div><span className="status-pill"><i/> Procesamiento local</span></div>

        {route === 'transcribir' && <>
          <div className="two-col">
            <section className="panel"><div className="panel-head"><h2>01 <span>· Nueva transcripción</span></h2><span className="step">INPUT</span></div>
              <label className="dropzone" onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); setFile(e.dataTransfer.files[0] || null) }}>
                <input type="file" accept="video/*,audio/*,.mkv,.flac,.m4a" onChange={e => setFile(e.target.files?.[0] || null)}/>
                <div className="upload-icon">⇧</div><strong>{file ? file.name : 'Arrastra tu archivo aquí'}</strong><span>{file ? `${(file.size / 1048576).toFixed(1)} MB · listo para procesar` : 'o haz clic para seleccionar un archivo'}</span><small>MP4, MKV, MOV, WAV, FLAC, MP3, M4A, WEBM · Máx. 2 GiB</small>
              </label>
              <div className="formgrid"><label>Idioma<select value={language} onChange={e => setLanguage(e.target.value)}>{options.map(o => <option key={o.code} value={o.code}>{o.title}</option>)}</select></label><label>Modelo<select value={model} onChange={e => setModel(e.target.value)}>{models.map(m => <option key={m.id} value={m.id}>{m.id}</option>)}</select></label><label>Dispositivo<select value={device} onChange={e => { const d=e.target.value;setDevice(d);setCompute(d === 'cpu' ? 'int8' : 'float16') }}><option value="cuda">NVIDIA CUDA</option><option value="cpu">CPU (pruebas)</option></select></label><label>Precisión<select value={compute} onChange={e => setCompute(e.target.value)}><option value="float16">float16</option><option value="int8_float16">int8_float16</option><option value="int8">int8</option><option value="float32">float32</option></select></label></div>
              <button disabled={!file || busy} className="primary-btn" onClick={submit}>{busy ? 'Enviando…' : '▶ Iniciar transcripción'} <span>→</span></button>
            </section>
            <section className="panel"><div className="panel-head"><h2>02 <span>· Vista previa</span></h2><span className="step">OUTPUT</span></div>
              {!job ? <div className="empty"><div className="empty-image">⌘</div><h3>Tu transcripción aparecerá aquí</h3><p>Carga un archivo para comenzar.</p></div> : <>
                <div className="job-summary"><div><span>ARCHIVO</span><strong title={job.original_name}>{job.original_name}</strong></div><span className={`badge ${job.status}`}>{STATUS[job.status]}</span></div>
                {job.status === 'failed' ? <p className="error-text">{job.error}</p> : job.status !== 'completed' ? <div className="processing"><div className="spinner"/> En cola o procesando en la GPU…</div> : <>
                  <div className="metrics"><div><small>IDIOMA</small><b>{transcript?.language.toUpperCase()}</b></div><div><small>TIEMPO ASR</small><b>{transcript?.elapsed_seconds.toFixed(2)} s</b></div><div><small>RTF</small><b>{transcript?.rtf?.toFixed(3) ?? 'N/D'}</b></div></div>
                  <MediaPlayer job={job} player={player} />
                  <CueList segments={transcript?.segments ?? []} time={player.time} onCueClick={i => player.seek(transcript!.segments[i].start, true)} />
                  <a className="download-btn" href={api.subtitles(job.id)}>↓ Descargar subtítulos .SRT</a>
                </>}
              </>}
            </section>
          </div>
        </>}

        {route === 'precision' && <QualityPage jobs={jobs} job={job} onSelect={id => navigate({ job: id })} />}
        {route === 'rendimiento' && <div className="panel"><div className="panel-head"><h2>Tiempo por etapa</h2><span className="step">{job?.original_name ?? 'SIN TRABAJO'}</span></div>
          {!job?.profiling || !job.transcript ? <div className="empty"><div className="empty-image">▥</div><h3>No hay mediciones todavía</h3><p>Completa una transcripción para ver cuánto tardó cada etapa.</p></div> : <>
            <div className="metrics perf-metrics">
              <div><small>DURACIÓN DEL AUDIO</small><b>{job.transcript.duration_seconds ? `${(job.transcript.duration_seconds / 60).toFixed(1)} min` : 'N/D'}</b></div>
              <div><small>TIEMPO DE PROCESO</small><b>{job.transcript.elapsed_seconds.toFixed(1)} s</b></div>
              <div><small>VELOCIDAD</small><b>{job.transcript.rtf ? `${(1 / job.transcript.rtf).toFixed(0)}× tiempo real` : 'N/D'}</b></div>
            </div>
            <table className="profiler-table"><thead><tr><th>Etapa</th><th>Duración</th><th>Peso relativo</th></tr></thead><tbody>{stagesInOrder(job.profiling).map((e, i) => <tr key={i} className={e.depth ? 'sub' : ''}><td title={e.name}>{e.depth ? '↳ ' : ''}{STAGES[e.name] ?? e.name}</td><td>{e.milliseconds >= 1000 ? `${(e.milliseconds / 1000).toFixed(2)} s` : `${e.milliseconds.toFixed(1)} ms`}</td><td><div className="bar-track"><div className="bar" style={{width: `${Math.min(100, 100 * e.milliseconds / Math.max(1, ...job.profiling!.map(x => x.milliseconds)))}%`}} /></div></td></tr>)}</tbody></table>
            <p className="hint">«Cargar modelo» solo ocurre en el primer trabajo de cada modelo; después queda en memoria. Velocidad: cuántos minutos de audio se procesan por cada minuto real.</p></>}
        </div>}
        <footer><span>SubDub-Titler · Clean Architecture + faster-whisper</span><span>EN · ES · JA · ZH</span></footer>
      </div>
    </main>
  </div>
}
