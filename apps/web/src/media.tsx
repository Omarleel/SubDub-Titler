import { useEffect, useRef, useState, type RefObject } from 'react'
import { api, type Job, type Segment } from './api'

const AUDIO = /\.(mp3|wav|flac|aac|m4a|ogg)$/i

export const clock = (seconds: number) => {
  const s = Math.max(0, Math.floor(seconds))
  const h = Math.floor(s / 3600), m = Math.floor(s / 60) % 60, r = s % 60
  return `${h ? `${h}:${String(m).padStart(2, '0')}` : m}:${String(r).padStart(2, '0')}`
}

export type Range = { start: number; end: number }

export type Player = {
  ref: RefObject<HTMLVideoElement | null>
  time: number
  playing: boolean
  rate: number
  setRate: (rate: number) => void
  seek: (seconds: number, play?: boolean) => void
  /** Reproduce [start, end) y pausa al llegar al final. */
  playRange: (range: Range) => void
  /** Pausa, o reanuda dentro del rango (vuelve al inicio si ya terminó). */
  toggle: (range?: Range | null) => void
  skip: (seconds: number) => void
}

export function usePlayer(): Player {
  const ref = useRef<HTMLVideoElement | null>(null)
  const stopAt = useRef<number | null>(null)
  const [time, setTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [rate, setRate] = useState(1)

  useEffect(() => {
    const media = ref.current
    if (!media) return
    media.playbackRate = rate
    const onTime = () => {
      setTime(media.currentTime)
      if (stopAt.current != null && media.currentTime >= stopAt.current) {
        media.pause(); stopAt.current = null
      }
    }
    const onSeekByUser = () => { if (media.paused) stopAt.current = null }
    const onPlay = () => setPlaying(true), onPause = () => setPlaying(false)
    const onLoaded = () => { media.playbackRate = rate }
    media.addEventListener('timeupdate', onTime)
    media.addEventListener('seeking', onSeekByUser)
    media.addEventListener('play', onPlay)
    media.addEventListener('pause', onPause)
    media.addEventListener('loadedmetadata', onLoaded)
    return () => {
      media.removeEventListener('timeupdate', onTime); media.removeEventListener('seeking', onSeekByUser)
      media.removeEventListener('play', onPlay); media.removeEventListener('pause', onPause)
      media.removeEventListener('loadedmetadata', onLoaded)
    }
  })

  const seek = (seconds: number, play = false) => {
    const media = ref.current
    if (!media) return
    stopAt.current = null
    media.currentTime = seconds
    if (play) void media.play()
  }
  const playRange = ({ start, end }: Range) => {
    seek(start, true)
    stopAt.current = end
  }
  const toggle = (range?: Range | null) => {
    const media = ref.current
    if (!media) return
    if (!media.paused) { media.pause(); return }
    if (!range) { void media.play(); return }
    if (media.currentTime < range.start || media.currentTime >= range.end - 0.1) media.currentTime = range.start
    void media.play()
    stopAt.current = range.end
  }
  const skip = (seconds: number) => {
    const media = ref.current
    if (media) media.currentTime = Math.max(0, media.currentTime + seconds)
  }
  return { ref, time, playing, rate, setRate, seek, playRange, toggle, skip }
}

/** Video (o audio) original con los subtítulos generados superpuestos. */
export function MediaPlayer({ job, player }: { job: Job; player: Player }) {
  const [failed, setFailed] = useState(false)
  useEffect(() => setFailed(false), [job.id])
  const isAudio = AUDIO.test(job.original_name)
  return <div className={`player ${isAudio ? 'audio' : ''}`}>
    {failed
      ? <p className="error-text">El navegador no puede reproducir este formato. Los subtítulos y la selección siguen funcionando.</p>
      : <video key={job.id} ref={player.ref} controls preload="metadata" src={api.media(job.id)}
          onError={() => setFailed(true)}>
          {job.status === 'completed' && <track kind="subtitles" src={api.vtt(job.id)} default
            srcLang={job.transcript?.language} label="SubDub" />}
        </video>}
  </div>
}

type CueListProps = {
  segments: Segment[]
  time: number
  onCueClick: (index: number, event: React.MouseEvent) => void
  selection?: [number, number] | null
}

/** Lista de subtítulos: resalta el que suena y, opcionalmente, un rango seleccionado. */
export function CueList({ segments, time, onCueClick, selection }: CueListProps) {
  const playing = segments.findIndex(s => time >= s.start && time < s.end)
  const listRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Seguir la reproducción sin robar el scroll de la página.
    const list = listRef.current, row = list?.children[playing] as HTMLElement | undefined
    if (list && row && (row.offsetTop < list.scrollTop || row.offsetTop > list.scrollTop + list.clientHeight - row.offsetHeight)) {
      list.scrollTo({ top: row.offsetTop - list.clientHeight / 3, behavior: 'smooth' })
    }
  }, [playing])

  return <div className="segments cues" ref={listRef}>
    {segments.map((s, i) => {
      const selected = selection && i >= selection[0] && i <= selection[1]
      return <button type="button" key={i} onClick={e => onCueClick(i, e)}
        className={`segment cue ${i === playing ? 'playing' : ''} ${selected ? 'selected' : ''}`}>
        <span>{clock(s.start)} → {clock(s.end)}</span><p>{s.text}</p>
      </button>
    })}
  </div>
}
