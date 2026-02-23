import { useEffect, useRef, useCallback } from 'react'

interface UseSSEOptions {
  url: string
  enabled?: boolean
  onMessage: (event: MessageEvent) => void
  onError?: (event: Event) => void
}

export function useSSE({ url, enabled = true, onMessage, onError }: UseSSEOptions) {
  const sourceRef = useRef<EventSource | null>(null)
  const onMessageRef = useRef(onMessage)
  const onErrorRef = useRef(onError)
  onMessageRef.current = onMessage
  onErrorRef.current = onError

  const close = useCallback(() => {
    if (sourceRef.current) {
      sourceRef.current.close()
      sourceRef.current = null
    }
  }, [])

  useEffect(() => {
    if (!enabled) {
      close()
      return
    }

    const fullUrl = `/api${url}`
    const es = new EventSource(fullUrl)
    sourceRef.current = es

    es.onmessage = (e) => onMessageRef.current(e)
    es.onerror = (e) => {
      onErrorRef.current?.(e)
      // Auto-reconnect is handled by EventSource spec
    }

    return () => {
      es.close()
      sourceRef.current = null
    }
  }, [url, enabled, close])

  return { close }
}
