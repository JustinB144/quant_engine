import { useEffect, useRef, useCallback } from 'react'

interface UseSSEOptions {
  url: string
  enabled?: boolean
  onMessage: (event: MessageEvent) => void
  onError?: (event: Event) => void
}

const EVENT_TYPES = ['status', 'started', 'progress', 'completed', 'failed', 'cancelled', 'done', 'message'] as const

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

    const handler = (e: MessageEvent) => {
      onMessageRef.current(e)
    }

    // Listen for all named events AND the default message event
    EVENT_TYPES.forEach(type => {
      es.addEventListener(type, handler)
    })
    es.onmessage = handler  // Catch any unnamed events as fallback

    es.onerror = (e) => {
      onErrorRef.current?.(e)
      // Auto-reconnect is handled by EventSource spec
    }

    return () => {
      EVENT_TYPES.forEach(type => {
        es.removeEventListener(type, handler)
      })
      es.close()
      sourceRef.current = null
    }
  }, [url, enabled, close])

  return { close }
}
