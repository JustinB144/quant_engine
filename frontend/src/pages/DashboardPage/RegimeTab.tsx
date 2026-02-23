import React from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import BarChart from '@/components/charts/BarChart'
import HeatmapChart from '@/components/charts/HeatmapChart'
import AreaChart from '@/components/charts/AreaChart'
import { REGIME_COLORS, REGIME_NAMES } from '@/components/charts/theme'
import type { RegimeInfo } from '@/types/dashboard'
import type { ResponseMeta } from '@/types/api'

interface Props {
  regime?: RegimeInfo
  meta?: ResponseMeta
}

export default function RegimeTab({ regime, meta }: Props) {
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

  return (
    <div>
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
              <HeatmapChart
                xLabels={regimeLabels}
                yLabels={regimeLabels}
                data={transition_matrix}
                height={380}
                colorRange={['#161b22', '#58a6ff', '#3fb950']}
              />
            ) : (
              <div style={{ color: 'var(--text-tertiary)', textAlign: 'center', padding: 40 }}>No transition matrix available</div>
            )}
          </ChartContainer>
        </div>
      </div>
    </div>
  )
}
