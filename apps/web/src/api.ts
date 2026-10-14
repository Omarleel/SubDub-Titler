export type Segment = { start: number; end: number; text: string }
export type Transcript = {
  language: string; model_id: string; duration_seconds: number | null;
  text: string; segments: Segment[]; elapsed_seconds: number; rtf: number | null;
}
export type ProfileEvent = { name: string; milliseconds: number; parent: string | null; depth: number }
export type Job = {
  id: string; original_name: string; status: 'queued' | 'running' | 'completed' | 'failed';
  created_at: string; model_id: string; language: string; error: string | null;
  transcript: Transcript | null; profiling: ProfileEvent[] | null
}
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
  jobs: () => fetch('/api/v1/jobs').then(r => unwrap<Job[]>(r)),
  create: (form: FormData) => fetch('/api/v1/jobs', { method: 'POST', body: form }).then(r => unwrap<Job>(r)),
  evaluate: (reference: string, hypothesis: string, language: string) =>
    fetch('/api/v1/evaluate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reference, hypothesis, language }),
    }).then(r => unwrap<Evaluation>(r)),
  subtitles: (id: string) => `/api/v1/jobs/${id}/subtitles`,
  media: (id: string) => `/api/v1/jobs/${id}/media`,
  vtt: (id: string) => `/api/v1/jobs/${id}/subtitles.vtt`,
}
