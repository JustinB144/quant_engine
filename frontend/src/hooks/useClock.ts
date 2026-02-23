import { useState, useEffect } from 'react'

export function useClock(): string {
  const [time, setTime] = useState(() => new Date().toLocaleTimeString('en-US', { hour12: false }))

  useEffect(() => {
    const id = setInterval(() => {
      setTime(new Date().toLocaleTimeString('en-US', { hour12: false }))
    }, 1000)
    return () => clearInterval(id)
  }, [])

  return time
}
