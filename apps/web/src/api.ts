export type Segment = { start: number; end: number; text: string; speaker?: string | null }
export type Transcript = {
  language: string; model_id: string; duration_seconds: number | null;
  text: string; segments: Segment[]; elapsed_seconds: number; rtf: number | null;
}
export type ProfileEvent = {
  name: string; milliseconds: number; parent: string | null; depth: number;
  vram_peak_mb?: number | null; vram_device_mb?: number | null;
}
export type JobKind = 'transcribe' | 'dub' | 'dub-cue' | 'dub-undo'
export type Job = {
  id: string; original_name: string; status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  cancel_requested?: number; created_at: string; updated_at: string; model_id: string; language: string; error: string | null;
  transcript: Transcript | null; profiling: ProfileEvent[] | null;
  kind: JobKind; parent_id: string | null; params: Record<string, unknown>;
  result: DubResult | null; stage: string | null; progress: number | null;
}

/** Espejo de domain/dubbing.py (report_to_dict). */
export type DubCue = {
  index: number; start: number; source_text: string; target_text: string;
  slot_seconds: number; natural_seconds: number; target_seconds: number;
  generated_seconds: number; speed: number; action: 'ok' | 'shorten'; speaker: string;
  candidates: string[]; attempts?: number;
}
export type CueQuality = {
  index: number; cer: number | null; cer_errors: number; cer_units: number;
  asr_text: string; similarity: number | null; similarity_slot?: number | null; mos: number | null;
  language_prob?: number | null;
}
export type Gate = {
  name: string; value: number | null; threshold: number; higher_is_better: boolean;
  passed: boolean | null;
}
export type SyncReport = {
  cues: number; within_tolerance: number; max_abs_error_ms: number; mean_abs_error_ms: number;
  overlaps: number; speed_in_range: number; speed_min: number; speed_max: number;
  shortened: number;
}
export type QualitySummary = {
  cer: number | null; similarity: number | null; similarity_original: number | null;
  similarity_slot?: number | null;
  mos: number | null; mos_original: number | null; background_wpm: number | null;
  language_prob?: number | null;
  lufs: number | null; true_peak: number | null;
}
export type DubResult = {
  track: DubCue[]; sync: SyncReport;
  reference: { start: number; end: number; text: string };
  outputs: Record<string, string>; quality: QualitySummary | null;
  cue_quality: CueQuality[]; gates: Gate[]; cue?: DubCue;
  references?: Record<string, { start: number; end: number; text: string }>;
  speech?: [number, number][];  // habla original; el resto del stem de voz se conserva como ambiente
}
export type DubOptions = {
  target_language: string; num_step: number; tts_batch: number; n_best: number;
  separator: string | null; evaluate: boolean; tts_engine: string; translator: string;
  diarize: boolean; max_speakers: number;
  merge_fragments: boolean; verify_attempts: number; reference_candidates: number;
}
export type Component = { id: string; kind: string; description: string; license: string }
export type Ratings = { ratings: Record<string, { score: number; note: string | null }>; count: number; mean: number | null }
export type Model = { id: string; backend: string; languages: string[]; description: string }
export type Counts = {
  substitutions: number; deletions: number; insertions: number;
  reference_units: number; rate: number | null;
}
export type Evaluation = {
  primary_metric: 'wer' | 'cer'; language: string; normalization: string;
  raw: { wer: Counts | null; cer: Counts | null };
  normalized: { wer: Counts | null; cer: Counts | null };
}

async function unwrap<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const message = await response.text()
    throw new Error(`${response.status}: ${message.slice(0, 400)}`)
  }
  return response.json() as Promise<T>
}
export const api = {
  models: () => fetch('/api/v1/models').then(r => unwrap<Model[]>(r)),
  jobs: (offset = 0, limit = 30) => fetch(`/api/v1/jobs?visible=true&summary=true&offset=${offset}&limit=${limit}`).then(r => unwrap<Job[]>(r)),
  job: (id: string) => fetch(`/api/v1/jobs/${encodeURIComponent(id)}`).then(r => unwrap<Job>(r)),
  revisions: (id: string) => fetch(`/api/v1/jobs/${id}/dub/revisions`).then(r => unwrap<{ count: number }>(r)),
  undo: (id: string) => fetch(`/api/v1/jobs/${id}/dub/undo`, { method: 'POST' }).then(r => unwrap<Job>(r)),
  cancel: (id: string) => fetch(`/api/v1/jobs/${id}/cancel`, { method: 'POST' }).then(r => unwrap<Job>(r)),
  retry: (id: string) => fetch(`/api/v1/jobs/${id}/retry`, { method: 'POST' }).then(r => unwrap<Job>(r)),
  create: (form: FormData) => fetch('/api/v1/jobs', { method: 'POST', body: form }).then(r => unwrap<Job>(r)),
  evaluate: (reference: string, hypothesis: string, language: string) =>
    fetch('/api/v1/evaluate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reference, hypothesis, language }),
    }).then(r => unwrap<Evaluation>(r)),
  createDub: (id: string, options: DubOptions) =>
    fetch(`/api/v1/jobs/${id}/dub`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(options),
    }).then(r => unwrap<Job>(r)),
  regenerate: (id: string, index: number, text: string) =>
    fetch(`/api/v1/jobs/${id}/dub/cues/${index}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }),
    }).then(r => unwrap<Job>(r)),
  rate: (id: string, index: number, score: number) =>
    fetch(`/api/v1/jobs/${id}/dub/cues/${index}/rating`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ score }),
    }).then(r => unwrap<Ratings>(r)),
  ratings: (id: string) => fetch(`/api/v1/jobs/${id}/dub/ratings`).then(r => unwrap<Ratings>(r)),
  dubComponents: () => fetch('/api/v1/dub/components')
    .then(r => unwrap<{ languages: string[]; components: Component[] }>(r)),
  /** `version` evita la caché del navegador tras regenerar un cue. */
  dubArtifact: (id: string, artifact: 'video' | 'mix' | 'voice' | 'subtitles', version = '') =>
    `/api/v1/jobs/${id}/dub/${artifact}${version ? `?v=${encodeURIComponent(version)}` : ''}`,
  dubCueAudio: (id: string, index: number, version = '') =>
    `/api/v1/jobs/${id}/dub/cues/${index}/audio${version ? `?v=${encodeURIComponent(version)}` : ''}`,
  subtitles: (id: string) => `/api/v1/jobs/${id}/subtitles`,
  media: (id: string) => `/api/v1/jobs/${id}/media`,
  vtt: (id: string) => `/api/v1/jobs/${id}/subtitles.vtt`,
}
