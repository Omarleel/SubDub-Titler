import { useEffect, useState } from 'react'
import { api, type Job, type Model, type ProfileEvent } from './api'
import { DubPage } from './DubPage'
import { useJobs } from './useJobs'
import { CueList, MediaPlayer, usePlayer } from './media'
import { QualityPage } from './QualityPage'
import { ROUTES, useLocation, type Route } from './router'
import * as ui from './ui'

const PAGES: Record<Route, { icon: string; title: string; subtitle: string }> = {
  transcribir: { icon: '▣', title: 'Estudio de transcripción', subtitle: 'Convierte audio y video en subtítulos con modelos ejecutados en tu equipo.' },
  precision: { icon: '◎', title: 'Medir precisión', subtitle: 'Compara la transcripción automática con el texto correcto y calcula el porcentaje de acierto.' },
  doblaje: { icon: '◐', title: 'Doblaje de voz', subtitle: 'Traduce y dobla con la voz clonada del hablante; mide sincronía, inteligibilidad y parecido de voz.' },
  rendimiento: { icon: '▥', title: 'Rendimiento', subtitle: 'Cuánto tardó cada etapa del proceso y qué tan rápido fue frente al tiempo real.' },
}
const STATUS: Record<Job['status'], string> = {
  queued: 'En cola', running: 'Procesando', completed: 'Completado', failed: 'Error', cancelled: 'Cancelado',
}
const STAGES: Record<string, string> = {
  'registry.engine_for': 'Preparar motor',
  'asr.isolate_dialogue': 'Aislar diálogo (Demucs)',
  'asr.transcribe_complete': 'Reconocimiento de voz (total)',
  'asr.model_load': 'Cargar modelo en la GPU',
  'audio.decode': 'Extraer audio del archivo',
  'asr.prepare_decode': 'Detectar voz (VAD) y preparar',
  'asr.consume_generator': 'Transcribir en la GPU',
  'subtitles.resegment': 'Dividir en subtítulos',
  'subtitles.write_srt': 'Guardar archivo SRT',
  'dub.components': 'Preparar componentes de doblaje',
  'dub.separate': 'Separar voz y fondo (total)',
  'dub.separate.load': 'Cargar Demucs',
  'dub.separate.apply': 'Separar voz y fondo en la GPU',
  'dub.voice_prompt': 'Preparar la voz clonada',
  'dub.tts.load': 'Cargar OmniVoice en la GPU',
  'dub.translate': 'Traducir (total)',
  'dub.translate.load': 'Cargar traductor NLLB',
  'dub.fit': 'Ajustar traducciones a su tiempo',
  'dub.tts.generate': 'Sintetizar la voz en la GPU',
  'dub.save_cues': 'Guardar clips por subtítulo',
  'dub.mix': 'Mezclar y normalizar sonoridad',
  'dub.mux': 'Exportar video multipista',
  'dub.eval': 'Medir calidad (total)',
  'dub.eval.cues': 'Re-transcribir y puntuar el doblaje',
  'dub.eval.original': 'Puntuar la voz original',
  'dub.eval.background': 'Buscar voz residual en el fondo',
  'dub.eval.load_sim': 'Cargar evaluador de parecido de voz',
  'dub.eval.load_mos': 'Cargar evaluador UTMOS',
  'dub.diarize': 'Detectar hablantes',
  'dub.load_cues': 'Cargar clips existentes',
  'dub.reference_probe': 'Elegir la mejor voz de referencia',
  'dub.verify': 'Verificar inteligibilidad y reintentar',
  'dub.mix.place': 'Colocar la voz en su tiempo',
  'dub.mix.level': 'Igualar nivel con la voz original',
  'dub.mix.write_voice': 'Guardar pista de voz',
  'dub.mix.background': 'Cargar el fondo',
  'dub.mix.normalize': 'Normalizar sonoridad y limitar picos',
  'dub.mix.measure': 'Medir sonoridad y pico real',
}
const SIDENAV_LABEL = 'text-[#66788f] text-[10px] tracking-[1.8px] font-bold mt-0 mx-[12px] mb-[14px] upto680:hidden'
const FIELD_LABEL = 'text-[11px] text-[#96a8be] font-semibold'
const FIELD = 'mt-[9px] block w-full bg-field border border-field-line rounded-[7px] py-[11px] px-[12px] text-field-text outline-0 cursor-pointer'
const TH = 'text-left p-[16px] border-b border-[#283548] text-[10px] text-[#8d9eb4] tracking-[1px]'
const TD = 'text-left p-[16px] border-b border-[#283548]'
const BADGE: Record<Job['status'], string> = {
  queued: 'bg-[#283247] text-[#c8d4ef]', running: 'bg-[#283247] text-[#c8d4ef]',
  completed: 'bg-ok-bg text-ok', failed: 'bg-bad-bg text-bad', cancelled: 'bg-[#283247] text-[#c8d4ef]',
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
  const { jobs, setJobs, hasMore, loadMore, error: jobsError } = useJobs(jobId ?? undefined)
  const [file, setFile] = useState<File | null>(null)
  const [model, setModel] = useState('small')
  const [language, setLanguage] = useState('auto')
  const [device, setDevice] = useState('cuda')
  const [compute, setCompute] = useState('float16')
  const [profileType, setProfileType] = useState<'standard' | 'sensitive' | 'isolate'>('standard')
  const [normalizeLoudness, setNormalizeLoudness] = useState(true)
  const [isolateDialogue, setIsolateDialogue] = useState(false)
  const [forcedAlignment, setForcedAlignment] = useState(false)
  const [diarize, setDiarize] = useState(false)
  const [maxSpeakers, setMaxSpeakers] = useState(4)
  const [conformNetflix, setConformNetflix] = useState(false)
  const [suppressHallucinations, setSuppressHallucinations] = useState(true)
  const [adaptiveTwoPass, setAdaptiveTwoPass] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.models().then(setModels).catch(e => setError(String(e)))
  }, [])

  // Las regeneraciones de un subtítulo son trabajos internos: no aparecen en el historial.
  const visible = jobs.filter(j => j.kind === 'transcribe' || j.kind === 'dub')
  const transcriptions = visible.filter(j => (j.kind ?? 'transcribe') === 'transcribe')
  const scope = route === 'doblaje' || route === 'rendimiento' ? visible : transcriptions
  const selectedJob = jobs.find(j => j.id === jobId)
  const job = (selectedJob && scope.includes(selectedJob) ? selectedJob : undefined)
    ?? (selectedJob?.kind === 'dub' ? transcriptions.find(j => j.id === selectedJob.parent_id) : undefined)
    ?? scope[0]
  const page = PAGES[route]
  const transcript = job?.transcript
  const dubbing = job?.kind === 'dub' ? job.result : null
  const dubAudio = dubbing ? dubbing.track.reduce((total, c) => total + c.generated_seconds, 0) : 0
  const processMs = (job?.profiling ?? []).filter(e => e.depth === 0 && !e.name.startsWith('dub.eval')).reduce((t, e) => t + e.milliseconds, 0)
  const ttsMs = job?.profiling?.find(e => e.name === 'dub.tts.generate')?.milliseconds

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
      if (profileType === 'sensitive') {
        data.set('vad_threshold', '0.28')
        data.set('vad_speech_pad_ms', '1000')
        data.set('vad_min_silence_ms', '2500')
        data.set('no_speech_threshold', '0.30')
      } else {
        data.set('vad_threshold', '0.35')
        data.set('vad_speech_pad_ms', '800')
        data.set('vad_min_silence_ms', '2000')
        data.set('no_speech_threshold', '0.40')
      }
      data.set('normalize_loudness', String(normalizeLoudness))
      data.set('isolate_dialogue', String(isolateDialogue))
      data.set('forced_alignment', String(forcedAlignment))
      data.set('diarize', String(diarize))
      data.set('max_speakers', String(maxSpeakers))
      data.set('conform_netflix', String(conformNetflix))
      data.set('suppress_hallucinations', String(suppressHallucinations))
      data.set('adaptive_two_pass', String(adaptiveTwoPass))
      const result = await api.create(data)
      navigate({ job: result.id })
      setJobs(prev => [result, ...prev])
      setFile(null)
    } catch (e) { setError(String(e)) }
    finally { setBusy(false) }
  }

  return <div className="flex min-h-screen upto680:block">
    <aside className="w-[258px] shrink-0 border-r border-[#202a3b] bg-chrome flex flex-col py-[28px] px-[16px] upto1150:w-[214px] upto680:w-full upto680:p-[14px] upto680:block">
      <div className="flex items-center gap-[12px] mb-[52px] px-[11px] upto680:mb-[16px]">
        <span className="w-[42px] h-[42px] rounded-[11px] bg-[linear-gradient(145deg,#655cfe,#20c8bc)] grid place-items-center font-display text-[26px] font-bold text-white relative">S<span className="text-[11px] absolute right-[4px] top-[1px]">✦</span></span>
        <div><strong className="block tracking-[-.5px] font-display text-[22px]">SubDub</strong><small className="text-[9px] tracking-[2px] text-[#7a92aa]">TITLER / LOCAL AI</small></div>
      </div>
      <div className={SIDENAV_LABEL}>WORKSPACE</div>
      {(Object.keys(ROUTES) as Route[]).map(r =>
        <a key={r} href={`/${r}${job ? `?trabajo=${job.id}` : ''}`}
          className={`w-full text-left border-0 rounded-[9px] py-[13px] px-[12px] my-[1px] text-[13px] flex gap-[12px] items-center no-underline upto680:w-auto upto680:p-[9px] upto680:text-[11px] ${route === r ? 'bg-[#1c2639] text-white shadow-[inset_3px_0_#7b81ff]' : 'bg-transparent text-inherit hover:bg-[#1c2639] hover:text-white'}`}
          onClick={e => { e.preventDefault(); navigate({ route: r }) }}><span className={`text-[18px] ${route === r ? 'text-[#a1a6ff]' : 'text-[#75899f]'}`}>{PAGES[r].icon}</span> {ROUTES[r]}</a>)}
      <div className={`${SIDENAV_LABEL} mt-[39px]`}>HISTORIAL</div>
      <div className="overflow-auto max-h-[40vh] upto680:hidden">{visible.length === 0 ? <p className="text-[#7b8ca5] text-[12px]">Aún no hay trabajos.</p> : visible.map(j =>
        <button className={`block w-full text-left border-0 rounded-[7px] py-[11px] px-[13px] ${job?.id === j.id ? 'bg-[#192338] text-[#f3f4ff]' : 'bg-transparent text-[#9aaec3] hover:bg-[#192338] hover:text-[#f3f4ff]'}`} key={j.id} onClick={() => navigate({ job: j.id })}>
          <span className="block overflow-hidden text-ellipsis whitespace-nowrap text-[12px]">{j.original_name}</span><small className="block text-[#71859e] text-[10px] mt-[4px]">{j.kind === 'dub' ? `Doblaje ${String(j.params?.target_language ?? '').toUpperCase()} · ` : ''}{STATUS[j.status]}</small>
        </button>)}{hasMore && <button type="button" className={ui.ghostBtn} onClick={loadMore}>Cargar trabajos anteriores</button>}</div>
      <div className="mt-auto pt-[15px] px-[10px] pb-[4px] text-[10px] font-bold tracking-[1px] text-[#99b0bb] flex items-center gap-[8px] upto680:hidden"><div className={ui.liveDot}/> BACKEND LOCAL <small className="ml-auto text-[#67758e]">v0.1.0</small></div>
    </aside>

    <main className="flex-1 min-w-0">
      <header className="h-[72px] border-b border-[#1d2938] flex items-center justify-between px-[47px] bg-chrome upto680:h-[55px] upto680:px-[18px]">
        <div className="text-[11px] tracking-[1.8px] text-[#8698b0] font-bold">SUBDUB <b className="text-[#3f5063] px-[8px]">/</b> {ROUTES[route]}</div>
        <div className="flex gap-[19px] items-center"><span className="text-[11px] text-[#8fa5b9] border border-[#263447] py-[7px] px-[12px] rounded-[7px] upto680:hidden">🔒 Offline-ready</span><span className="grid place-items-center bg-[#293653] border border-[#3b4f69] w-[34px] h-[34px] rounded-[50%] font-bold text-[11px]">SD</span></div>
      </header>
      <div className="max-w-[1320px] m-auto pt-[48px] px-[48px] pb-[25px] upto1150:p-[28px] upto680:py-[24px] upto680:px-[16px]">
        {jobsError && <div role="alert" className={ui.errorBanner}>{jobsError}</div>}
        {error && <div role="alert" className={ui.errorBanner}>{error}<button className={ui.errorClose} onClick={() => setError(null)}>✕</button></div>}
        <div className="flex justify-between items-center gap-[12px] mb-[31px] upto680:block">
          <div><div className="text-[10px] text-[#7f86ff] tracking-[2.4px] font-bold mb-[12px]">AUDIO INTELLIGENCE · LOCAL GPU</div>
            <h1 className="font-display text-[36px] tracking-[-1.3px] mt-0 mx-0 mb-[8px] font-semibold upto680:text-[26px]">{page.title}</h1>
            <p className="text-[13px] text-[#8091a7] max-w-[600px] m-0 leading-[1.6]">{page.subtitle}</p></div>
          <span className="border border-[#224f48] bg-[#102923] text-[#77d9be] text-[11px] flex gap-[9px] items-center py-[9px] px-[13px] rounded-[24px] whitespace-nowrap upto680:inline-flex upto680:mt-[16px]"><i className={ui.liveDot}/> Procesamiento local</span>
        </div>

        {job && <div className="flex gap-[12px] items-center mb-[20px]">
          {(job.status === 'queued' || job.status === 'running') && <>
            <button type="button" className={ui.ghostBtn} disabled={!!job.cancel_requested} onClick={async () => {
              try { const updated = await api.cancel(job.id); setJobs(prev => prev.map(j => j.id === updated.id ? updated : j)) }
              catch (e) { setError(String(e)) }
            }}>{job.cancel_requested ? 'Cancelación solicitada…' : 'Cancelar trabajo'}</button>
            <span className="text-[12px] text-[#96a8be]">La etapa actual termina antes de cancelar.</span>
          </>}
          {(job.status === 'failed' || job.status === 'cancelled') && <button type="button" className={ui.ghostBtn} onClick={async () => {
            try { const created = await api.retry(job.id); setJobs(prev => [created, ...prev]); navigate({ job: created.id }) }
            catch (e) { setError(String(e)) }
          }}>Reintentar con la misma configuración</button>}
        </div>}
        {route === 'transcribir' && <>
          <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-[22px] upto1150:grid-cols-[1fr]">
            <section className={ui.panel}><div className={ui.panelHead}><h2 className={ui.panelTitle}>01 <span className={ui.panelTitleSub}>· Nueva transcripción</span></h2><span className={ui.step}>INPUT</span></div>
              <label className="min-h-[210px] border-[1.5px] border-dashed border-[#354660] rounded-[11px] flex items-center justify-center flex-col gap-[10px] text-center bg-[linear-gradient(150deg,#151e30,#101824)] relative cursor-pointer p-[24px] hover:border-[#8485ff]" onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); setFile(e.dataTransfer.files[0] || null) }}>
                <input className="absolute inset-0 opacity-0 cursor-pointer w-full h-full" type="file" accept="video/*,audio/*,.mkv,.flac,.m4a" onChange={e => setFile(e.target.files?.[0] || null)}/>
                <div className="text-[27px] rounded-[11px] grid place-items-center bg-[#263354] w-[49px] h-[49px] text-[#98a4ff] mb-[4px]">⇧</div>
                <strong className="text-[14px] max-w-[94%] overflow-hidden text-ellipsis whitespace-nowrap">{file ? file.name : 'Arrastra tu archivo aquí'}</strong>
                <span className="text-[12px] text-[#94a2b5]">{file ? `${(file.size / 1048576).toFixed(1)} MB · listo para procesar` : 'o haz clic para seleccionar un archivo'}</span>
                <small className="text-[10px] text-[#657a94] mt-[7px]">MP4, MKV, MOV, WAV, FLAC, MP3, M4A, WEBM · Límite configurable en el servidor</small>
              </label>
              <div className="grid grid-cols-[1fr_1fr] gap-[17px] my-[22px]">
                <label className={FIELD_LABEL}>Idioma<select className={FIELD} value={language} onChange={e => setLanguage(e.target.value)}>{options.map(o => <option key={o.code} value={o.code}>{o.title}</option>)}</select></label>
                <label className={FIELD_LABEL}>Modelo<select className={FIELD} value={model} onChange={e => setModel(e.target.value)}>{models.map(m => <option key={m.id} value={m.id}>{m.id}</option>)}</select></label>
                <label className={FIELD_LABEL}>Dispositivo<select className={FIELD} value={device} onChange={e => { const d=e.target.value;setDevice(d);setCompute(d === 'cpu' ? 'int8' : 'float16') }}><option value="cuda">NVIDIA CUDA</option><option value="cpu">CPU (pruebas)</option></select></label>
                <label className={FIELD_LABEL}>Precisión<select className={FIELD} value={compute} onChange={e => setCompute(e.target.value)}><option value="float16">float16</option><option value="int8_float16">int8_float16</option><option value="int8">int8</option><option value="float32">float32</option></select></label>
                <label className={`${FIELD_LABEL} col-span-2`}>
                  Perfil acústico y VAD
                  <select
                    className={FIELD}
                    value={profileType}
                    onChange={e => {
                      const val = e.target.value as 'standard' | 'sensitive' | 'isolate'
                      setProfileType(val)
                      if (val === 'isolate') {
                        setIsolateDialogue(true)
                        setNormalizeLoudness(true)
                      } else if (val === 'sensitive') {
                        setIsolateDialogue(false)
                        setNormalizeLoudness(true)
                      } else {
                        setIsolateDialogue(false)
                        setNormalizeLoudness(true)
                      }
                    }}
                  >
                    <option value="standard">Estándar · VAD calibrado (pad=800ms, th=0.35)</option>
                    <option value="sensitive">Diálogo difícil / Susurros · VAD sensible (pad=1000ms, th=0.28)</option>
                    <option value="isolate">Con música / SFX · Aislar voz con Demucs pre-ASR</option>
                  </select>
                </label>
              </div>
              <div className="flex flex-col gap-[9px] mb-[22px] px-[2px]">
                <label className="flex items-center gap-[10px] text-[12px] text-[#97a8bc] cursor-pointer">
                  <input
                    type="checkbox"
                    className="cursor-pointer accent-[#7b81ff]"
                    checked={normalizeLoudness}
                    onChange={e => setNormalizeLoudness(e.target.checked)}
                  />
                  <span>Normalización de sonoridad EBU R128 (-16 LUFS con ffmpeg)</span>
                </label>
                <label className="flex items-center gap-[10px] text-[12px] text-[#97a8bc] cursor-pointer">
                  <input
                    type="checkbox"
                    className="cursor-pointer accent-[#7b81ff]"
                    checked={isolateDialogue}
                    onChange={e => setIsolateDialogue(e.target.checked)}
                  />
                  <span>Aislar diálogo con Demucs (elimina música y ruido ambiental antes de ASR)</span>
                </label>
                <label className="flex items-center gap-[10px] text-[12px] text-[#97a8bc] cursor-pointer">
                  <input
                    type="checkbox"
                    className="cursor-pointer accent-[#7b81ff]"
                    checked={forcedAlignment}
                    onChange={e => setForcedAlignment(e.target.checked)}
                  />
                  <span>Alineación forzada CTC (MMS-FA: sincronización milimétrica de palabras)</span>
                </label>
                <label className="flex items-center gap-[10px] text-[12px] text-[#97a8bc] cursor-pointer">
                  <input
                    type="checkbox"
                    className="cursor-pointer accent-[#7b81ff]"
                    checked={diarize}
                    onChange={e => setDiarize(e.target.checked)}
                  />
                  <span>Diarización ASR (identificar y etiquetar hablantes: [Hablante 1], [Hablante 2])</span>
                </label>
                <label className="flex items-center gap-[10px] text-[12px] text-[#97a8bc] cursor-pointer">
                  <input
                    type="checkbox"
                    className="cursor-pointer accent-[#7b81ff]"
                    checked={conformNetflix}
                    onChange={e => setConformNetflix(e.target.checked)}
                  />
                  <span>Norma profesional Netflix / BBC (máx. 42 CPL, saltos sintácticos y límite de 17 CPS)</span>
                </label>
                <label className="flex items-center gap-[10px] text-[12px] text-[#97a8bc] cursor-pointer">
                  <input
                    type="checkbox"
                    className="cursor-pointer accent-[#7b81ff]"
                    checked={suppressHallucinations}
                    onChange={e => setSuppressHallucinations(e.target.checked)}
                  />
                  <span>Supresión activa de alucinaciones en silencio y BGM (bucles zlib / frases fantasma de YouTube)</span>
                </label>
                <label className="flex items-center gap-[10px] text-[12px] text-[#97a8bc] cursor-pointer">
                  <input
                    type="checkbox"
                    className="cursor-pointer accent-[#7b81ff]"
                    checked={adaptiveTwoPass}
                    onChange={e => setAdaptiveTwoPass(e.target.checked)}
                  />
                  <span>Inferencia adaptativa en dos pasadas (Dynamic Prompting para nombres propios y rescate de baja confianza)</span>
                </label>
              </div>
              <button disabled={!file || busy} className={ui.primaryBtn} onClick={submit}>{busy ? 'Enviando…' : '▶ Iniciar transcripción'} <span className="ml-auto">→</span></button>
            </section>
            <section className={ui.panel}><div className={ui.panelHead}><h2 className={ui.panelTitle}>02 <span className={ui.panelTitleSub}>· Vista previa</span></h2><span className={ui.step}>OUTPUT</span></div>
              {!job ? <div className={ui.empty}><div className={ui.emptyIcon}>⌘</div><h3 className={ui.emptyTitle}>Tu transcripción aparecerá aquí</h3><p className={ui.emptyText}>Carga un archivo para comenzar.</p></div> : <>
                <div className="border-b border-[#273247] pb-[18px] flex justify-between gap-[12px]">
                  <div className="min-w-0"><span className="block text-[#7c8fa5] tracking-[1px] text-[10px] mb-[7px]">ARCHIVO</span><strong className="text-[13px] max-w-[220px] overflow-hidden block text-ellipsis whitespace-nowrap" title={job.original_name}>{job.original_name}</strong></div>
                  <span className={`text-[10px] rounded-[6px] p-[7px] h-fit ${BADGE[job.status]}`}>{STATUS[job.status]}</span>
                </div>
                {job.status === 'failed' ? <p className={ui.errorText}>{job.error}</p> : job.status !== 'completed' ? <div className="py-[95px] px-[8px] text-center text-[#9eafce] text-[13px]"><div className="border-[3px] border-[#2b3950] border-t-[#8e87fa] rounded-[50%] w-[33px] h-[33px] animate-spin mt-0 mx-auto mb-[20px]"/> En cola o procesando en la GPU…</div> : <>
                  <div className={ui.metrics}>
                    <div className={ui.metric}><small className={ui.metricLabel}>IDIOMA</small><b className={ui.metricValue}>{transcript?.language.toUpperCase()}</b></div>
                    <div className={ui.metric}><small className={ui.metricLabel}>TIEMPO ASR</small><b className={ui.metricValue}>{transcript?.elapsed_seconds.toFixed(2)} s</b></div>
                    <div className={ui.metric}><small className={ui.metricLabel}>RTF</small><b className={ui.metricValue}>{transcript?.rtf?.toFixed(3) ?? 'N/D'}</b></div>
                  </div>
                  <MediaPlayer job={job} player={player} />
                  <CueList segments={transcript?.segments ?? []} time={player.time} onCueClick={i => player.seek(transcript!.segments[i].start, true)} />
                  <a className="block text-center text-[#b5b6ff] border border-[#555cc1] rounded-[8px] p-[13px] mt-[15px] no-underline text-[12px] font-bold hover:bg-[#25264e]" href={api.subtitles(job.id)}>↓ Descargar subtítulos .SRT</a>
                </>}
              </>}
            </section>
          </div>
        </>}

        {route === 'precision' && <QualityPage jobs={transcriptions} job={job} onSelect={id => navigate({ job: id })} />}
        {route === 'doblaje' && (
          <DubPage
            jobs={jobs}
            job={job}
            onSelect={(id, created) => {
              if (created) setJobs(prev => [created, ...prev.filter(j => j.id !== created.id)])
              navigate({ job: id })
            }}
          />
        )}
        {route === 'rendimiento' && <div className={ui.panel}><div className={ui.panelHead}><h2 className={ui.panelTitle}>Tiempo por etapa</h2><span className={ui.step}>{job?.original_name ?? 'SIN TRABAJO'}</span></div>
          {job?.kind === 'dub' && dubbing && job.profiling ? <>
            <div className={`${ui.metrics} max-w-[760px]`}>
              <div className={ui.metric}><small className={ui.metricLabel}>VOZ DOBLADA</small><b className={ui.metricValue}>{(dubAudio / 60).toFixed(1)} min · {dubbing.track.length} subtítulos</b></div>
              <div className={ui.metric}><small className={ui.metricLabel}>TIEMPO DE PROCESO</small><b className={ui.metricValue}>{(processMs / 1000).toFixed(1)} s</b></div>
              <div className={ui.metric}><small className={ui.metricLabel}>RTF SÍNTESIS</small><b className={ui.metricValue}>{ttsMs && dubAudio ? (ttsMs / 1000 / dubAudio).toFixed(3) : 'N/D'}</b></div>
            </div>
            <table className="w-full border-collapse text-[#cad5e8] text-[12px]"><thead><tr><th className={TH}>Etapa</th><th className={TH}>Duración</th><th className={TH}>VRAM pico (PyTorch)</th><th className={TH}>GPU ocupada</th><th className={TH}>Peso relativo</th></tr></thead><tbody>{stagesInOrder(job.profiling).map((e, i) => <tr key={i}><td className={`${TD} ${e.depth ? 'pl-[26px] text-[#9fb0c6]' : ''}`} title={e.name}>{e.depth ? '↳ ' : ''}{STAGES[e.name] ?? e.name}</td><td className={TD}>{e.milliseconds >= 1000 ? `${(e.milliseconds / 1000).toFixed(2)} s` : `${e.milliseconds.toFixed(1)} ms`}</td><td className={TD}>{e.vram_peak_mb != null ? `${(e.vram_peak_mb / 1024).toFixed(2)} GB` : '—'}</td><td className={TD}>{e.vram_device_mb ? `${(e.vram_device_mb / 1024).toFixed(1)} GB` : '—'}</td><td className={TD}><div className="h-[8px] bg-[#222c40] rounded-[10px] w-full overflow-hidden"><div className="h-full rounded-[8px] bg-[linear-gradient(90deg,#6d73f9,#5cbbd7)]" style={{width: `${Math.min(100, 100 * e.milliseconds / Math.max(1, ...job.profiling!.map(x => x.milliseconds)))}%`}} /></div></td></tr>)}</tbody></table>
            <p className={ui.hint}>RTF de síntesis: segundos de cálculo por segundo de voz generada (menor es mejor). «VRAM pico» mide solo la memoria de PyTorch; «GPU ocupada» es la memoria total de la tarjeta al terminar la etapa (incluye CTranslate2 y otros programas). La evaluación de calidad no cuenta en el tiempo de proceso.</p></>
          : !job?.profiling || !job.transcript ? <div className={ui.empty}><div className={ui.emptyIcon}>▥</div><h3 className={ui.emptyTitle}>No hay mediciones todavía</h3><p className={ui.emptyText}>Completa una transcripción para ver cuánto tardó cada etapa.</p></div> : <>
            <div className={`${ui.metrics} max-w-[760px]`}>
              <div className={ui.metric}><small className={ui.metricLabel}>DURACIÓN DEL AUDIO</small><b className={ui.metricValue}>{job.transcript.duration_seconds ? `${(job.transcript.duration_seconds / 60).toFixed(1)} min` : 'N/D'}</b></div>
              <div className={ui.metric}><small className={ui.metricLabel}>TIEMPO DE PROCESO</small><b className={ui.metricValue}>{job.transcript.elapsed_seconds.toFixed(1)} s</b></div>
              <div className={ui.metric}><small className={ui.metricLabel}>VELOCIDAD</small><b className={ui.metricValue}>{job.transcript.rtf ? `${(1 / job.transcript.rtf).toFixed(0)}× tiempo real` : 'N/D'}</b></div>
            </div>
            <table className="w-full border-collapse text-[#cad5e8] text-[12px]"><thead><tr><th className={TH}>Etapa</th><th className={TH}>Duración</th><th className={TH}>Peso relativo</th></tr></thead><tbody>{stagesInOrder(job.profiling).map((e, i) => <tr key={i}><td className={`${TD} ${e.depth ? 'pl-[26px] text-[#9fb0c6]' : ''}`} title={e.name}>{e.depth ? '↳ ' : ''}{STAGES[e.name] ?? e.name}</td><td className={TD}>{e.milliseconds >= 1000 ? `${(e.milliseconds / 1000).toFixed(2)} s` : `${e.milliseconds.toFixed(1)} ms`}</td><td className={TD}><div className="h-[8px] bg-[#222c40] rounded-[10px] w-full overflow-hidden"><div className="h-full rounded-[8px] bg-[linear-gradient(90deg,#6d73f9,#5cbbd7)]" style={{width: `${Math.min(100, 100 * e.milliseconds / Math.max(1, ...job.profiling!.map(x => x.milliseconds)))}%`}} /></div></td></tr>)}</tbody></table>
            <p className={ui.hint}>«Cargar modelo» solo ocurre en el primer trabajo de cada modelo; después queda en memoria. Velocidad: cuántos minutos de audio se procesan por cada minuto real.</p></>}
        </div>}
        <footer className="flex justify-between text-[#52677f] text-[10px] tracking-[.5px] mt-[37px]"><span>SubDub-Titler</span><span>EN · ES · JA · ZH</span></footer>
      </div>
    </main>
  </div>
}
