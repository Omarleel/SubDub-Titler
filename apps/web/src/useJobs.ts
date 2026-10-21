import { useEffect, useRef, useState } from 'react'
import { api, type Job } from './api'

/** Lista ligera; conserva detalles sólo mientras su versión siga vigente. */
export function useJobs(selectedId: string | undefined) {
  const [jobs, setJobs] = useState<Job[]>([])
  const [pages, setPages] = useState(1)
  const [hasMore, setHasMore] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const cache = useRef(new Map<string, Job>())
  useEffect(() => {
    let stopped = false
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => {
      try {
        const list: Job[] = []
        let lastSize = 0
        for (let page = 0; page < pages; page++) {
          const batch = await api.jobs(page * 30)
          list.push(...batch)
          lastSize = batch.length
          if (batch.length < 30) break
        }
        const ids = new Set(list.map(j => j.id))
        const detail = async (id: string) => {
          const summary = list.find(j => j.id === id)
          const cached = cache.current.get(id)
          const full = summary && cached?.updated_at === summary.updated_at ? cached : await api.job(id)
          cache.current.set(id, full)
          if (!ids.has(id)) { list.push(full); ids.add(id) }
          return full
        }
        const currentId = selectedId ?? list[0]?.id
        if (currentId) {
          const current = await detail(currentId)
          if (current.parent_id) await detail(current.parent_id)
          if (current.kind === 'transcribe') {
            const dub = list.find(j => j.kind === 'dub' && j.parent_id === current.id)
            if (dub) await detail(dub.id)
          }
        }
        if (stopped) return
        setHasMore(lastSize === 30)
        setJobs(list.map(j => cache.current.get(j.id)?.updated_at === j.updated_at ? cache.current.get(j.id)! : j))
        setError(null)
      } catch (e) {
        if (!stopped) setError(String(e))
      } finally {
        if (!stopped) timer = setTimeout(poll, 1800)
      }
    }
    void poll()
    return () => { stopped = true; clearTimeout(timer) }
  }, [selectedId, pages])
  return { jobs, setJobs, hasMore, loadMore: () => setPages(n => n + 1), error }
}
