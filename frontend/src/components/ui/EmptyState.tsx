import React from 'react'
import { Inbox } from 'lucide-react'

export default function EmptyState({ message = 'No data available' }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-12">
      <Inbox size={32} style={{ color: 'var(--text-tertiary)', marginBottom: 8, opacity: 0.5 }} />
      <div style={{ color: 'var(--text-tertiary)', fontSize: 13 }}>{message}</div>
    </div>
  )
}
