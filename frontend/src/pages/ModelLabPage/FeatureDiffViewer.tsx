import React from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import { useFeatureImportance } from '@/api/queries/useModels'

/**
 * Feature importance diff viewer — shows changes in feature importance
 * between the current model and a previous version.
 */
export default function FeatureDiffViewer() {
  const { data, isLoading } = useFeatureImportance()
  const fi = data?.data

  const globalImp = fi?.global_importance ?? {}
  const previousImp = fi?.previous_importance ?? {}

  // If no previous data, show current only
  const hasPrevious = Object.keys(previousImp).length > 0

  // Compute diffs
  const allFeatures = new Set([...Object.keys(globalImp), ...Object.keys(previousImp)])
  const diffs = Array.from(allFeatures)
    .map((name) => ({
      name,
      current: globalImp[name] ?? 0,
      previous: previousImp[name] ?? 0,
      diff: (globalImp[name] ?? 0) - (previousImp[name] ?? 0),
    }))
    .sort((a, b) => Math.abs(b.diff) - Math.abs(a.diff))
    .slice(0, 20)

  const newFeatures = diffs.filter((d) => d.previous === 0 && d.current > 0)
  const droppedFeatures = diffs.filter((d) => d.current === 0 && d.previous > 0)
  const changedFeatures = diffs.filter((d) => d.current > 0 && d.previous > 0)

  return (
    <ChartContainer
      title="Feature Importance Changes"
      meta={data?.meta}
      isLoading={isLoading}
      isEmpty={!hasPrevious}
      emptyMessage="No previous model version to compare"
    >
      <div className="space-y-1">
        {newFeatures.length > 0 && (
          <div className="mb-3">
            <div
              className="font-mono uppercase mb-1"
              style={{ fontSize: 10, color: 'var(--accent-green)', letterSpacing: '0.5px' }}
            >
              New Features ({newFeatures.length})
            </div>
            {newFeatures.map((f) => (
              <FeatureDiffRow key={f.name} {...f} status="new" />
            ))}
          </div>
        )}

        {droppedFeatures.length > 0 && (
          <div className="mb-3">
            <div
              className="font-mono uppercase mb-1"
              style={{ fontSize: 10, color: 'var(--accent-red)', letterSpacing: '0.5px' }}
            >
              Dropped Features ({droppedFeatures.length})
            </div>
            {droppedFeatures.map((f) => (
              <FeatureDiffRow key={f.name} {...f} status="dropped" />
            ))}
          </div>
        )}

        {changedFeatures.length > 0 && (
          <div>
            <div
              className="font-mono uppercase mb-1"
              style={{ fontSize: 10, color: 'var(--text-tertiary)', letterSpacing: '0.5px' }}
            >
              Changed Features (Top {changedFeatures.length})
            </div>
            {changedFeatures.map((f) => (
              <FeatureDiffRow key={f.name} {...f} status="changed" />
            ))}
          </div>
        )}
      </div>
    </ChartContainer>
  )
}

function FeatureDiffRow({
  name,
  current,
  previous,
  diff,
  status,
}: {
  name: string
  current: number
  previous: number
  diff: number
  status: 'new' | 'dropped' | 'changed'
}) {
  const diffColor =
    status === 'new'
      ? 'var(--accent-green)'
      : status === 'dropped'
        ? 'var(--accent-red)'
        : diff > 0
          ? 'var(--accent-green)'
          : diff < 0
            ? 'var(--accent-red)'
            : 'var(--text-tertiary)'

  const barWidth = Math.min(Math.abs(diff) * 1000, 100) // Scale for visibility

  return (
    <div
      className="flex items-center gap-3 px-3 py-1.5 rounded"
      style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-light)' }}
    >
      <span className="font-mono" style={{ fontSize: 11, color: 'var(--text-primary)', minWidth: 150 }}>
        {name}
      </span>
      <div className="flex-1 flex items-center gap-2">
        <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-tertiary)', minWidth: 50 }}>
          {previous.toFixed(4)}
        </span>
        <span style={{ color: 'var(--text-tertiary)', fontSize: 10 }}>→</span>
        <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-secondary)', minWidth: 50 }}>
          {current.toFixed(4)}
        </span>
        <div className="flex-1 relative h-3 rounded overflow-hidden" style={{ backgroundColor: 'var(--bg-tertiary)' }}>
          <div
            className="absolute top-0 h-full rounded"
            style={{
              backgroundColor: diffColor,
              width: `${barWidth}%`,
              left: diff < 0 ? `${50 - barWidth / 2}%` : '50%',
              opacity: 0.6,
            }}
          />
        </div>
        <span
          className="font-mono"
          style={{ fontSize: 11, color: diffColor, fontWeight: 600, minWidth: 60, textAlign: 'right' }}
        >
          {diff > 0 ? '+' : ''}
          {diff.toFixed(4)}
        </span>
      </div>
    </div>
  )
}
