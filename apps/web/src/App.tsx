import { useEffect, useState } from 'react'
import { api, type Evaluation, type Job, type Model } from './api'

type Tab = 'workspace' | 'quality' | 'profiling'
const options = [{ code: 'auto', title: 'Automático' }, { code: 'es', title: 'Español' },
  { code: 'en', title: 'Inglés' }, { code: 'ja', title: 'Japonés' }, { code: 'zh', title: 'Chino' }]

const fmt = (seconds: number) => new Date(Math.round(seconds * 1000)).toISOString().slice(11, 23)
const prettyRate = (rate: number | null | undefined) => rate == null ? 'N/D' : `${(rate * 100).toFixed(2)}%`

export function App() {
  const [tab, setTab] = useState<Tab>('workspace')
  const [models, setModels] = useState<Model[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [model, setModel] = useState('small')
  const [language, setLanguage] = useState('auto')
  const [device, setDevice] = useState('cuda')
  const [compute, setCompute] = useState('float16')
  const [reference, setReference] = useState('')
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.models().then(setModels).catch(e => setError(String(e)))
    const poll = () => api.jobs().then(setJobs).catch(e => setError(String(e)))
    poll()
    const id = window.setInterval(poll, 1800)
    return () => window.clearInterval(id)
  }, [])

  const job = jobs.find(j => j.id === selected) || jobs[0]
  const transcript = job?.transcript

  async function submit() {
    if (!file) return
    setBusy(true); setError(null); setEvaluation(null)
    try {
      const data = new FormData()
      data.set('file', file)
      data.set('model_id', model)
      data.set('language', language)
      data.set('device', device)
      data.set('compute_type', compute)
      const result = await api.create(data)
      setSelected(result.id)
      setJobs(prev => [result, ...prev])
      setFile(null)
    } catch (e) { setError(String(e)) }
    finally { setBusy(false) }
  }

  async function validate() {
    if (!transcript || !reference.trim()) return
    setBusy(true); setError(null)
    try {
      setEvaluation(await api.evaluate(reference, transcript.text,
        ['en', 'es', 'ja', 'zh'].includes(transcript.language) ? transcript.language : 'es'))
    } catch (e) { setError(String(e)) }
    finally { setBusy(false) }
  }

  return <div className="shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-icon">S<span>✦</span></span><div><strong>SubDub</strong><small>TITLER / LOCAL AI</small></div></div>
      <div className="sidenav-label">WORKSPACE</div>
      <button className={`nav ${tab === 'workspace' ? 'active' : ''}`} onClick={() => setTab('workspace')}><span>▣</span> Transcripciones</button>
      <button className={`nav ${tab === 'quality' ? 'active' : ''}`} onClick={() => setTab('quality')}><span>◎</span> Quality Lab</button>
      <button className={`nav ${tab === 'profiling' ? 'active' : ''}`} onClick={() => setTab('profiling')}><span>▥</span> Performance Lab</button>
      <div className="sidenav-label gap">HISTORIAL</div>
      <div className="history">{jobs.length === 0 ? <p className="muted small">Aún no hay trabajos.</p> : jobs.map(j =>
        <button className={`history-item ${job?.id === j.id ? 'chosen' : ''}`} key={j.id} onClick={() => {setSelected(j.id); setEvaluation(null)}}>
          <span className="ellipsis">{j.original_name}</span><small>{j.status}</small>
        </button>)}</div>
      <div className="sidebar-foot"><div className="online-dot"/> BACKEND LOCAL <small>v0.1.0</small></div>
    </aside>

    <main className="main">
      <header className="topbar"><div className="breadcrumb">PROYECTOS <b>/</b> {tab === 'workspace' ? 'Transcripción' : tab === 'quality' ? 'Quality Lab' : 'Performance Lab'}</div><div className="top-right"><span className="chip">🔒 Offline-ready</span><span className="avatar">SD</span></div></header>
      <div className="page">
        {error && <div role="alert" className="error">{error}<button onClick={() => setError(null)}>✕</button></div>}
        <div className="heading"><div><div className="eyebrow">AUDIO INTELLIGENCE · LOCAL GPU</div><h1>{tab === 'workspace' ? 'Estudio de transcripción' : tab === 'quality' ? 'Laboratorio de calidad' : 'Análisis de rendimiento'}</h1><p className="subtitle">{tab === 'workspace' ? 'Convierte audio y video en subtítulos con modelos ejecutados en tu equipo.' : tab === 'quality' ? 'Evalúa la precisión frente a una referencia humana.' : 'Inspecciona tiempos por operación y el factor de tiempo real.'}</p></div><span className="status-pill"><i/> Procesamiento local</span></div>

        {tab === 'workspace' && <>
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
                <div className="job-summary"><div><span>ARCHIVO</span><strong title={job.original_name}>{job.original_name}</strong></div><span className={`badge ${job.status}`}>{job.status}</span></div>
                {job.status === 'failed' ? <p className="error-text">{job.error}</p> : job.status !== 'completed' ? <div className="processing"><div className="spinner"/> En cola o procesando en la GPU…</div> : <>
                  <div className="metrics"><div><small>IDIOMA</small><b>{transcript?.language.toUpperCase()}</b></div><div><small>TIEMPO ASR</small><b>{transcript?.elapsed_seconds.toFixed(2)} s</b></div><div><small>RTF</small><b>{transcript?.rtf?.toFixed(3) ?? 'N/D'}</b></div></div>
                  <div className="segments">{transcript?.segments.map((s, i) => <div className="segment" key={i}><span>{fmt(s.start)} → {fmt(s.end)}</span><p>{s.text}</p></div>)}</div>
                  <a className="download-btn" href={api.subtitles(job.id)}>↓ Descargar subtítulos .SRT</a>
                </>}
              </>}
            </section>
          </div>
        </>}

        {tab === 'quality' && <div className="two-col">
          <section className="panel"><div className="panel-head"><h2>01 <span>· Referencia humana</span></h2><span className="step">GROUND TRUTH</span></div><p className="panel-desc">Pega la transcripción revisada manualmente. Se compara con el trabajo seleccionado.</p>
            <textarea value={reference} onChange={e => setReference(e.target.value)} placeholder="Transcribe aquí exactamente lo que dice el hablante…" rows={11} />
            <p className="hint">Trabajo actual: {job?.original_name ?? 'Ninguno'} · Idioma detectado: {transcript?.language ?? 'N/D'}</p>
            <button className="primary-btn" disabled={!reference.trim() || !transcript || busy} onClick={validate}>◎ Evaluar precisión →</button>
          </section><section className="panel"><div className="panel-head"><h2>02 <span>· Reporte de errores</span></h2><span className="step">ACCURACY</span></div>
            {!evaluation ? <div className="empty"><div className="empty-image">◎</div><h3>Esperando evaluación</h3><p>Selecciona una transcripción y añade una referencia humana.</p></div> : <><div className="quality-number">{prettyRate(evaluation.normalized[evaluation.primary_metric]?.rate)}<span>{evaluation.primary_metric.toUpperCase()} normalizado</span></div><div className="quality-grid"><div>RAW CER<strong>{prettyRate(evaluation.raw.cer?.rate)}</strong></div><div>NORM CER<strong>{prettyRate(evaluation.normalized.cer?.rate)}</strong></div><div>RAW WER<strong>{prettyRate(evaluation.raw.wer?.rate)}</strong></div><div>NORM WER<strong>{prettyRate(evaluation.normalized.wer?.rate)}</strong></div></div><p className="hint">Política: {evaluation.normalization}. Cuanto menor la tasa, mejor.</p></>}
          </section>
        </div>}
        {tab === 'profiling' && <div className="panel"><div className="panel-head"><h2>Desglose de operaciones</h2><span className="step">PROFILING</span></div>
          {!job?.profiling ? <div className="empty"><div className="empty-image">▥</div><h3>No hay trazas disponibles</h3><p>Completa una transcripción para registrar las etapas.</p></div> : <><div className="metrics perf-metrics"><div><small>TRABAJO</small><b>{job.original_name}</b></div><div><small>TIEMPO ASR</small><b>{job.transcript?.elapsed_seconds.toFixed(3)} s</b></div><div><small>REAL-TIME FACTOR</small><b>{job.transcript?.rtf?.toFixed(3) ?? 'N/D'}</b></div></div><table className="profiler-table"><thead><tr><th>Operación</th><th>Duración</th><th>Contribución relativa</th></tr></thead><tbody>{job.profiling.map((e, i) => <tr key={i}><td>{e.name}</td><td>{e.milliseconds.toFixed(2)} ms</td><td><div className="bar-track"><div className="bar" style={{width: `${Math.min(100, 100 * e.milliseconds / Math.max(1, ...job.profiling!.map(x => x.milliseconds)))}%`}} /></div></td></tr>)}</tbody></table><p className="hint">Los spans miden paredes de CPU/proceso; kernels CUDA requieren NVIDIA Nsight externo.</p></>}
        </div>}
        <footer><span>SubDub-Titler · Clean Architecture + faster-whisper</span><span>EN · ES · JA · ZH</span></footer>
      </div>
    </main>
  </div>
}
