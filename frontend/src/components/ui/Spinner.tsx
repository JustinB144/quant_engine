import React from 'react'
import { Loader2 } from 'lucide-react'

export default function Spinner({ size = 24, className = '' }: { size?: number; className?: string }) {
  return (
    <Loader2
      size={size}
      className={`animate-spin ${className}`}
      style={{ color: 'var(--accent-blue)' }}
    />
  )
}
