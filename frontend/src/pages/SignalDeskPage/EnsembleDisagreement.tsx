import React from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import RadarChart from '@/components/charts/RadarChart'
import MetricCard from '@/components/ui/MetricCard'

interface EnsembleDisagreementProps {
  signals: Array<{
    ticker: string
    predicted_return?: number
    confidence?: number
    ensemble_members?: Record<string, number>
    ensemble_std?: number
  }>
}

/**
 * Displays ensemble model disagreement — how much individual models
 * within the ensemble differ in their predictions. High disagreement
 * suggests uncertainty; low disagreement suggests consensus.
 */
export default function EnsembleDisagreement({ signals }: EnsembleDisagreementProps) {
  // Filter signals that have ensemble member data
  const withEnsemble = signals.filter((s) => s.ensemble_members && Object.keys(s.ensemble_members).length > 0)

  if (withEnsemble.length === 0) {
    return (
      <ChartContainer title="Ensemble Disagreement" isEmpty emptyMessage="No ensemble member data available">
        <div />
      </ChartContainer>
    )
  }

  // Compute overall disagreement stats
  const disagreements = withEnsemble.map((s) => {
    const members = Object.values(s.ensemble_members!)
    const mean = members.reduce((a, b) => a + b, 0) / members.length
    const std = Math.sqrt(members.reduce((sum, v) => sum + (v - mean) ** 2, 0) / members.length)
    return { ticker: s.ticker, std, members: s.ensemble_members! }
  })

  const avgDisagreement = disagreements.reduce((sum, d) => sum + d.std, 0) / disagreements.length
  const maxDisagreement = Math.max(...disagreements.map((d) => d.std))
  const highDisagreementCount = disagreements.filter((d) => d.std > avgDisagreement * 1.5).length

  // Top 5 most-disagreed signals
  const topDisagreed = [...disagreements].sort((a, b) => b.std - a.std).slice(0, 5)

  // Model-level average predictions for radar chart
  const modelNames = Object.keys(withEnsemble[0].ensemble_members!)
  const modelAvgs = modelNames.map((name) => {
    const vals = withEnsemble.map((s) => s.ensemble_members![name] ?? 0)
    return vals.reduce((a, b) => a + b, 0) / vals.length
  })

  return (
    <ChartContainer title="Ensemble Disagreement">
      <div className="grid grid-cols-3 gap-3 mb-4">
        <MetricCard
          label="Avg Disagreement"
          value={`${(avgDisagreement * 100).toFixed(2)}%`}
          color={avgDisagreement < 0.005 ? 'var(--accent-green)' : avgDisagreement < 0.01 ? 'var(--accent-amber)' : 'var(--accent-red)'}
          subtitle="Std dev of member predictions"
        />
        <MetricCard
          label="Max Disagreement"
          value={`${(maxDisagreement * 100).toFixed(2)}%`}
          color="var(--accent-amber)"
        />
        <MetricCard
          label="High Disagreement"
          value={`${highDisagreementCount} / ${disagreements.length}`}
          color={highDisagreementCount === 0 ? 'var(--accent-green)' : 'var(--accent-amber)'}
          subtitle={`Signals >1.5x avg disagreement`}
        />
      </div>

      <div className="grid grid-cols-2 gap-3">
        {/* Model consensus radar */}
        {modelNames.length >= 3 && (
          <div>
            <div
              className="font-mono uppercase mb-2"
              style={{ fontSize: 10, color: 'var(--text-tertiary)', letterSpacing: '0.5px' }}
            >
              Model Avg Prediction
            </div>
            <RadarChart
              categories={modelNames}
              series={[{ name: 'Avg Prediction', data: modelAvgs.map((v) => Math.abs(v) * 10000) }]}
              height={260}
            />
          </div>
        )}

        {/* Top disagreed signals */}
        <div>
          <div
            className="font-mono uppercase mb-2"
            style={{ fontSize: 10, color: 'var(--text-tertiary)', letterSpacing: '0.5px' }}
          >
            Top Disagreed Signals
          </div>
          <div className="space-y-1">
            {topDisagreed.map((d) => (
              <div
                key={d.ticker}
                className="flex items-center justify-between px-3 py-2 rounded"
                style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-light)' }}
              >
                <span className="font-mono" style={{ fontSize: 12, color: 'var(--text-primary)' }}>
                  {d.ticker}
                </span>
                <div className="flex items-center gap-3">
                  {Object.entries(d.members).map(([model, pred]) => (
                    <span
                      key={model}
                      className="font-mono"
                      style={{
                        fontSize: 10,
                        color: pred >= 0 ? 'var(--accent-green)' : 'var(--accent-red)',
                      }}
                    >
                      {model}: {(pred * 100).toFixed(2)}%
                    </span>
                  ))}
                  <span
                    className="font-mono px-1.5 py-0.5 rounded"
                    style={{
                      fontSize: 10,
                      fontWeight: 600,
                      backgroundColor: 'rgba(248,81,73,0.12)',
                      color: 'var(--accent-red)',
                    }}
                  >
                    ±{(d.std * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </ChartContainer>
  )
}
