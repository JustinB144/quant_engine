import React from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import MetricCard from '@/components/ui/MetricCard'
import LineChart from '@/components/charts/LineChart'
import type { ModelHealth } from '@/types/models'

interface Props {
  modelHealth?: ModelHealth
}

export default function ModelHealthTab({ modelHealth }: Props) {
  const cvGap = modelHealth?.cv_gap ?? 0
  const r2 = modelHealth?.holdout_r2 ?? 0
  const ic = modelHealth?.holdout_ic ?? 0
  const drift = modelHealth?.ic_drift ?? 0
  const retrain = modelHealth?.retrain_triggered ?? false
  const reasons = modelHealth?.retrain_reasons ?? []
  const history = modelHealth?.registry_history ?? []

  const r2Color = r2 > 0.02 ? 'var(--accent-green)' : r2 > 0 ? 'var(--accent-amber)' : 'var(--accent-red)'
  const icColor = ic > 0.05 ? 'var(--accent-green)' : ic > 0 ? 'var(--accent-amber)' : 'var(--accent-red)'
  const driftColor = Math.abs(drift) < 0.05 ? 'var(--accent-green)' : Math.abs(drift) < 0.1 ? 'var(--accent-amber)' : 'var(--accent-red)'
  const gapColor = cvGap < 0.05 ? 'var(--accent-green)' : cvGap < 0.15 ? 'var(--accent-amber)' : 'var(--accent-red)'

  // Registry timeline from array of version objects
  const versions = history.map((v) => v.version_id.slice(0, 8))
  const timelineSeries = [
    { name: 'CV Gap', data: history.map((v) => v.cv_gap), color: '#d29922' },
    { name: 'Holdout R\u00b2', data: history.map((v) => v.holdout_r2), color: '#58a6ff' },
    { name: 'Holdout IC', data: history.map((v) => v.holdout_spearman), color: '#3fb950' },
  ]

  const statusLine = retrain ? 'STATUS: RETRAIN TRIGGERED' : 'STATUS: Stable -- no retrain signals'
  const statusColor = retrain ? 'var(--accent-red)' : 'var(--accent-green)'
  const retrainLines = [statusLine, '', ...(reasons.length ? ['Trigger reasons:', ...reasons.map((r) => `  - ${r}`)] : ['No active retrain triggers.'])]

  return (
    <div>
      <div className="grid grid-cols-4 gap-3 mb-3">
        <MetricCard label="Holdout R\u00b2" value={r2.toFixed(4)} color={r2Color} />
        <MetricCard label="Holdout IC" value={ic.toFixed(4)} color={icColor} />
        <MetricCard label="IC Drift" value={drift.toFixed(4)} color={driftColor} subtitle="Recent vs baseline" />
        <MetricCard label="CV Gap" value={cvGap.toFixed(4)} color={gapColor} subtitle="IS-OOS degradation" />
      </div>

      <div className="grid grid-cols-12 gap-3">
        <div className="col-span-7">
          <ChartContainer title="Model Health Timeline">
            {versions.length > 0 ? (
              <LineChart
                categories={versions}
                series={timelineSeries}
                height={360}
                yAxisName="Metric Value"
              />
            ) : (
              <div style={{ color: 'var(--text-tertiary)', textAlign: 'center', padding: 40 }}>No model registry history</div>
            )}
          </ChartContainer>
        </div>
        <div className="col-span-5">
          <ChartContainer title="Retrain Monitor">
            <pre
              className="font-mono"
              style={{ color: statusColor, fontSize: 12, whiteSpace: 'pre-wrap', margin: 0 }}
            >
              {retrainLines.join('\n')}
            </pre>
          </ChartContainer>
        </div>
      </div>
    </div>
  )
}
