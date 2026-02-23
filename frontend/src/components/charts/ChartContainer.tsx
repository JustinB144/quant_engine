import React from 'react'
import DataProvenanceBadge from '@/components/ui/DataProvenanceBadge'
import Spinner from '@/components/ui/Spinner'
import EmptyState from '@/components/ui/EmptyState'
import ErrorPanel from '@/components/ui/ErrorPanel'
import type { ResponseMeta } from '@/types/api'

interface ChartContainerProps {
  title?: string
  meta?: ResponseMeta
  isLoading?: boolean
  error?: string
  isEmpty?: boolean
  emptyMessage?: string
  onRetry?: () => void
  children: React.ReactNode
  className?: string
}

export default function ChartContainer({
  title,
  meta,
  isLoading,
  error,
  isEmpty,
  emptyMessage,
  onRetry,
  children,
  className = '',
}: ChartContainerProps) {
  return (
    <div
      className={`card-panel relative ${className}`}
    >
      {(title || meta) && (
        <div className="flex items-center justify-between mb-3">
          {title && <div className="card-panel-header mb-0">{title}</div>}
          <DataProvenanceBadge meta={meta} />
        </div>
      )}

      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <Spinner />
        </div>
      ) : error ? (
        <ErrorPanel message={error} onRetry={onRetry} />
      ) : isEmpty ? (
        <EmptyState message={emptyMessage} />
      ) : (
        children
      )}
    </div>
  )
}
