import React from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import RegimeBadge from '@/components/ui/RegimeBadge'
import BarChart from '@/components/charts/BarChart'
import HeatmapChart from '@/components/charts/HeatmapChart'
import { useDashboardRegime } from '@/api/queries/useDashboard'
import { REGIME_COLORS, REGIME_NAMES } from '@/components/charts/theme'

export default function RegimeTab() {
  const { data, isLoading } = useDashboardRegime()
  const regime = data?.data

  const currentProbs = regime?.current_probs ?? {}
  const probLabels = Object.keys(currentProbs)
  const probValues = probLabels.map((k) => currentProbs[k])
  const probColors = probLabels.map((label) => {
    const idx = Object.entries(REGIME_NAMES).find(([, v]) => v === label)?.[0]
    return idx != null ? REGIME_COLORS[Number(idx)] : '#58a6ff'
  })

  const transMatrix = regime?.transition_matrix ?? []
  const regimeLabels = [0, 1, 2, 3].map((i) => REGIME_NAMES[i])

  // Determine current regime from max probability
  const maxIdx = probValues.length > 0 ? probValues.indexOf(Math.max(...probValues)) : -1
  const currentRegimeIdx = maxIdx >= 0
    ? Object.entries(REGIME_NAMES).find(([, v]) => v === probLabels[maxIdx])?.[0] ?? '0'
    : '0'

  return (
    <div>
      <div className="mb-4 flex items-center gap-3">
        <span className="font-mono" style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>Current Regime:</span>
        <RegimeBadge regime={Number(currentRegimeIdx)} />
      </div>

      <div className="grid grid-cols-2 gap-3">
        <ChartContainer title="Regime Probabilities" meta={data?.meta} isLoading={isLoading}>
          {probLabels.length > 0 ? (
            <BarChart
              categories={probLabels}
              series={[{ name: 'Probability', data: probValues, color: probColors }]}
              horizontal
              height={280}
              showValues
            />
          ) : (
            <div style={{ color: 'var(--text-tertiary)', textAlign: 'center', padding: 40 }}>No regime data</div>
          )}
        </ChartContainer>
        <ChartContainer title="Transition Matrix">
          {transMatrix.length >= 2 ? (
            <HeatmapChart
              xLabels={regimeLabels}
              yLabels={regimeLabels}
              data={transMatrix}
              height={350}
              colorRange={['#161b22', '#58a6ff', '#3fb950']}
            />
          ) : (
            <div style={{ color: 'var(--text-tertiary)', textAlign: 'center', padding: 40 }}>No transition matrix</div>
          )}
        </ChartContainer>
      </div>
    </div>
  )
}
