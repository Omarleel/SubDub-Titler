import { useEffect, useRef, useState, type RefObject } from 'react'
import { api, type Job, type Segment } from './api'
import { errorText } from './ui'

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

type Variant = 'preview' | 'bench'

/** Video (o audio) original con los subtítulos generados superpuestos. */
export function MediaPlayer({ job, player, variant = 'preview' }: { job: Job; player: Player; variant?: Variant }) {
  const [failed, setFailed] = useState(false)
  useEffect(() => setFailed(false), [job.id])
  const isAudio = AUDIO.test(job.original_name)
  const height = variant === 'bench' ? 'max-h-[38vh] short:max-h-[30vh]' : isAudio ? 'max-h-[54px]' : 'max-h-[420px]'
  return <div className="bg-[#05080e] border border-panel-line rounded-[10px] overflow-hidden grid place-items-center min-h-[120px]">
    {failed
      ? <p className={`${errorText} p-[18px] text-center`}>El navegador no puede reproducir este formato. Los subtítulos y la selección siguen funcionando.</p>
      : <video key={job.id} ref={player.ref} controls preload="metadata" src={api.media(job.id)}
          className={`w-full block bg-black [&::cue]:bg-[#000000c0] [&::cue]:text-[15px] ${height}`}
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
  variant?: Variant
}

const LIST = {
  preview: 'mt-[14px] max-h-[330px]',
  bench: 'flex-1 min-h-[120px] max-h-none mt-0 upto1100:max-h-[340px]',
}

function cueState(playing: boolean, selected: boolean) {
  if (selected) return `border-l-[#8b8cff] ${playing ? 'bg-[#263b4d]' : 'bg-[#25264d]'}`
  if (playing) return 'border-l-ok bg-[#13231f]'
  return 'border-l-transparent bg-transparent hover:bg-field'
}

/** Lista de subtítulos: resalta el que suena y, opcionalmente, un rango seleccionado. */
export function CueList({ segments, time, onCueClick, selection, variant = 'preview' }: CueListProps) {
  const playing = segments.findIndex(s => time >= s.start && time < s.end)
  const listRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Seguir la reproducción sin robar el scroll de la página.
    const list = listRef.current, row = list?.children[playing] as HTMLElement | undefined
    if (list && row && (row.offsetTop < list.scrollTop || row.offsetTop > list.scrollTop + list.clientHeight - row.offsetHeight)) {
      list.scrollTo({ top: row.offsetTop - list.clientHeight / 3, behavior: 'smooth' })
    }
  }, [playing])

  return <div className={`overflow-auto border border-[#283347] rounded-[8px] relative ${LIST[variant]}`} ref={listRef}>
    {segments.map((s, i) => {
      const selected = !!selection && i >= selection[0] && i <= selection[1]
      return <button type="button" key={i} onClick={e => onCueClick(i, e)}
        className={`block w-full text-left text-inherit cursor-pointer py-[12px] px-[15px] border-0 border-b border-b-[#263144] border-l-[3px] last:border-b-0 ${cueState(i === playing, selected)}`}>
        <span className="font-[monospace] text-[10px] text-[#9899ff]">{clock(s.start)} → {clock(s.end)}</span>
        <p className="text-[13px] leading-[1.6] mt-[7px] mx-0 mb-0 text-[#d7dfed]">{s.text}</p>
      </button>
    })}
  </div>
}
