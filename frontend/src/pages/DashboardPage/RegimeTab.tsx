import React, { useMemo } from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import BarChart from '@/components/charts/BarChart'
import HeatmapChart from '@/components/charts/HeatmapChart'
import AreaChart from '@/components/charts/AreaChart'
import { REGIME_COLORS, REGIME_NAMES } from '@/components/charts/theme'
import { useRegimeMetadata } from '@/api/queries/useDashboard'
import type { RegimeInfo, RegimeMetadata } from '@/types/dashboard'
import type { ResponseMeta } from '@/types/api'

interface Props {
  regime?: RegimeInfo
  meta?: ResponseMeta
}

/** Badge showing the current regime with colored indicator dot. */
function RegimeBadge({ label, color }: { label: string; color: string }) {
  return (
    <div
      className="flex items-center gap-2 px-3 py-2 rounded-md"
      style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-light)' }}
    >
      <span
        className="inline-block rounded-full flex-shrink-0"
        style={{ width: 10, height: 10, backgroundColor: color }}
      />
      <span className="font-mono" style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>
        {label}
      </span>
    </div>
  )
}

/** Card explaining the current regime's definition and portfolio impact. */
function RegimeDetailCard({
  metadata,
  currentLabel,
  durationDays,
}: {
  metadata?: RegimeMetadata
  currentLabel: string
  durationDays?: number
}) {
  // Find the matching regime entry from metadata
  const entry = useMemo(() => {
    if (!metadata) return null
    for (const [, info] of Object.entries(metadata.regimes)) {
      if (info.name === currentLabel || info.name.toLowerCase() === currentLabel.toLowerCase()) {
        return info
      }
    }
    return null
  }, [metadata, currentLabel])

  if (!entry) return null

  return (
    <div className="card-panel mb-3">
      <div className="flex items-start gap-4">
        {/* Regime definition */}
        <div className="flex-1">
          <div className="font-mono mb-1" style={{ fontSize: 10, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Current Regime
          </div>
          <RegimeBadge label={entry.name} color={entry.color} />
          <p className="font-mono mt-2" style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            {entry.definition}
          </p>
          {durationDays != null && (
            <div className="font-mono mt-2" style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>
              In this regime for <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{durationDays} days</span>
            </div>
          )}
        </div>

        {/* Portfolio impact */}
        <div
          className="flex-shrink-0 rounded-md"
          style={{
            width: 260,
            padding: '12px 16px',
            backgroundColor: 'var(--bg-tertiary)',
            border: '1px solid var(--border-light)',
          }}
        >
          <div className="font-mono mb-2" style={{ fontSize: 10, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Portfolio Impact
          </div>
          <div className="flex items-center justify-between mb-1.5">
            <span className="font-mono" style={{ fontSize: 11, color: 'var(--text-secondary)' }}>Position Size</span>
            <span className="font-mono" style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>
              {(entry.portfolio_impact.position_size_multiplier * 100).toFixed(0)}%
            </span>
          </div>
          <div className="flex items-center justify-between mb-2">
            <span className="font-mono" style={{ fontSize: 11, color: 'var(--text-secondary)' }}>Stop Multiplier</span>
            <span className="font-mono" style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>
              {entry.portfolio_impact.stop_loss_multiplier.toFixed(1)}x
            </span>
          </div>
          <p className="font-mono" style={{ fontSize: 10, color: 'var(--text-tertiary)', lineHeight: 1.5 }}>
            {entry.portfolio_impact.description}
          </p>
        </div>
      </div>

      {/* Detection method */}
      <div className="mt-3 pt-3" style={{ borderTop: '1px solid var(--border-light)' }}>
        <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>
          Detection: </span>
        <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-secondary)' }}>
          {entry.detection}
        </span>
      </div>
    </div>
  )
}

/** Horizontal timeline showing colored regime blocks. */
function RegimeTimeline({ changes }: { changes: RegimeInfo['regime_changes'] }) {
  if (!changes || changes.length === 0) return null

  // Build blocks: each change marks the END of a regime period
  // We show blocks for the "from_regime" durations plus the final "to_regime"
  const blocks = changes.map((c) => ({
    regime: c.from_regime,
    date: c.date,
    duration: c.duration_days,
  }))

  const totalDays = blocks.reduce((sum, b) => sum + Math.max(b.duration, 1), 0)

  const regimeColorMap: Record<string, string> = {}
  for (const [key, name] of Object.entries(REGIME_NAMES)) {
    regimeColorMap[name] = REGIME_COLORS[Number(key)]
  }

  return (
    <div className="card-panel mb-3">
      <div className="card-panel-header">Regime History Timeline</div>
      <div className="flex w-full rounded overflow-hidden" style={{ height: 28 }}>
        {blocks.map((block, i) => {
          const widthPct = Math.max((block.duration / totalDays) * 100, 1)
          const color = regimeColorMap[block.regime] || '#888888'
          return (
            <div
              key={i}
              title={`${block.regime}: ${block.duration} days (until ${block.date})`}
              style={{
                width: `${widthPct}%`,
                backgroundColor: color,
                opacity: 0.85,
                cursor: 'pointer',
                transition: 'opacity 150ms',
              }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLDivElement).style.opacity = '1' }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLDivElement).style.opacity = '0.85' }}
            />
          )
        })}
      </div>
      {/* Legend */}
      <div className="flex gap-4 mt-2">
        {Object.entries(REGIME_NAMES).map(([key, name]) => (
          <div key={key} className="flex items-center gap-1.5">
            <span
              className="inline-block rounded-sm"
              style={{ width: 8, height: 8, backgroundColor: REGIME_COLORS[Number(key)] }}
            />
            <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>
              {name}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function RegimeTab({ regime, meta }: Props) {
  const { data: metadataResp } = useRegimeMetadata()
  const metadata = metadataResp?.data

  if (!regime) return <div style={{ color: 'var(--text-tertiary)', padding: 40, textAlign: 'center' }}>No regime data available</div>

  const { prob_history, current_probs, transition_matrix } = regime

  // Stacked area probability history from prob_history array
  const dates = prob_history.map((p) => p.date)
  const areaSeries = [0, 1, 2, 3].map((i) => ({
    name: REGIME_NAMES[i],
    data: prob_history.map((p) => {
      const key = `regime_prob_${i}` as keyof typeof p
      return (p[key] as number) ?? 0
    }),
    color: REGIME_COLORS[i],
    stack: 'one',
  }))

  // Current probabilities bar
  const probLabels = Object.keys(current_probs)
  const probValues = probLabels.map((k) => current_probs[k])
  const probColors = probLabels.map((label) => {
    const idx = Object.entries(REGIME_NAMES).find(([, v]) => v === label)?.[0]
    return idx != null ? REGIME_COLORS[Number(idx)] : '#58a6ff'
  })

  // Transition matrix
  const regimeLabels = [0, 1, 2, 3].map((i) => REGIME_NAMES[i])

  // Build enriched heatmap tooltip using metadata
  const matrixExplanation = metadata?.transition_matrix_explanation

  return (
    <div>
      {/* Regime explanation card */}
      <RegimeDetailCard
        metadata={metadata}
        currentLabel={regime.current_label}
        durationDays={regime.current_regime_duration_days}
      />

      {/* Regime history timeline */}
      <RegimeTimeline changes={regime.regime_changes} />

      <ChartContainer title="Probability History" meta={meta}>
        {dates.length > 1 ? (
          <AreaChart
            categories={dates}
            series={areaSeries}
            height={380}
            yAxisName="Probability"
            yAxisRange={[0, 1]}
            showRangeSelector
          />
        ) : (
          <div style={{ color: 'var(--text-tertiary)', textAlign: 'center', padding: 40 }}>No regime history available</div>
        )}
      </ChartContainer>

      <div className="grid grid-cols-12 gap-3">
        <div className="col-span-5">
          <ChartContainer title="Current Probabilities">
            {probLabels.length > 0 ? (
              <BarChart
                categories={probLabels}
                series={[{ name: 'Probability', data: probValues, color: probColors }]}
                horizontal
                height={280}
                showValues
              />
            ) : (
              <div style={{ color: 'var(--text-tertiary)', textAlign: 'center', padding: 40 }}>No current probabilities</div>
            )}
          </ChartContainer>
        </div>
        <div className="col-span-7">
          <ChartContainer title="Transition Matrix">
            {transition_matrix.length >= 2 ? (
              <>
                <HeatmapChart
                  xLabels={regimeLabels}
                  yLabels={regimeLabels}
                  data={transition_matrix}
                  height={380}
                  colorRange={['#161b22', '#58a6ff', '#3fb950']}
                />
                {matrixExplanation && (
                  <p className="font-mono mt-2" style={{ fontSize: 10, color: 'var(--text-tertiary)', lineHeight: 1.5 }}>
                    {matrixExplanation}
                  </p>
                )}
              </>
            ) : (
              <div style={{ color: 'var(--text-tertiary)', textAlign: 'center', padding: 40 }}>No transition matrix available</div>
            )}
          </ChartContainer>
        </div>
      </div>
    </div>
  )
}
