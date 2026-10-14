import { useCallback, useEffect, useState } from 'react'

export const ROUTES = {
  transcribir: 'Transcribir',
  precision: 'Medir precisión',
  rendimiento: 'Rendimiento',
} as const
export type Route = keyof typeof ROUTES

type Location = { route: Route; job: string | null }

function parse(): Location {
  const segment = window.location.pathname.replace(/^\/+|\/+$/g, '')
  const route = (segment in ROUTES ? segment : 'transcribir') as Route
  return { route, job: new URLSearchParams(window.location.search).get('trabajo') }
}

function href({ route, job }: Location) {
  return `/${route}${job ? `?trabajo=${encodeURIComponent(job)}` : ''}`
}

/** Router mínimo sobre History API: la sección y el trabajo viven en la URL. */
export function useLocation() {
  const [location, setLocation] = useState(parse)

  useEffect(() => {
    // Normaliza "/" o rutas desconocidas sin crear entradas extra en el historial.
    if (window.location.pathname + window.location.search !== href(location)) {
      window.history.replaceState(null, '', href(location))
    }
    const onPop = () => setLocation(parse())
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])

  const navigate = useCallback((next: Partial<Location>) => {
    setLocation(current => {
      const target = { ...current, ...next }
      if (href(target) !== href(current)) window.history.pushState(null, '', href(target))
      return target
    })
  }, [])

  return { ...location, navigate }
}
