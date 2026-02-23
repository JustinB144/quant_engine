import React from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import BarChart from '@/components/charts/BarChart'
import HeatmapChart from '@/components/charts/HeatmapChart'
import type { FeatureImportance } from '@/types/models'

interface Props {
  featureImportance?: FeatureImportance
}

export default function FeatureImportanceTab({ featureImportance }: Props) {
  // Global importance: Record<string, number> -> sorted pairs
  const globalImp = featureImportance?.global_importance ?? {}
  const sortedPairs = Object.entries(globalImp)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => a.value - b.value)
    .slice(-20) // top 20

  // Regime heatmap: Record<string, Record<string, number>>
  const regimeHeatmap = featureImportance?.regime_heatmap ?? {}
  const regimeNames = Object.keys(regimeHeatmap)
  // Collect all feature names across all regimes
  const featureSet = new Set<string>()
  for (const regime of regimeNames) {
    for (const feature of Object.keys(regimeHeatmap[regime])) {
      featureSet.add(feature)
    }
  }
  const featureNames = Array.from(featureSet).slice(0, 20) // top 20
  const heatmapData = featureNames.map((feature) =>
    regimeNames.map((regime) => regimeHeatmap[regime][feature] ?? 0)
  )

  return (
    <div className="grid grid-cols-2 gap-3">
      <ChartContainer title="Global Feature Importance (Top 20)">
        {sortedPairs.length > 0 ? (
          <BarChart
            categories={sortedPairs.map((p) => p.name)}
            series={[{ name: 'Importance', data: sortedPairs.map((p) => p.value), color: '#58a6ff' }]}
            horizontal
            height={Math.max(400, sortedPairs.length * 24)}
            showValues
          />
        ) : (
          <div style={{ color: 'var(--text-tertiary)', textAlign: 'center', padding: 40 }}>No feature importance data</div>
        )}
      </ChartContainer>
      <ChartContainer title="Regime Feature Heatmap">
        {featureNames.length > 0 && regimeNames.length > 0 ? (
          <HeatmapChart
            xLabels={regimeNames}
            yLabels={featureNames}
            data={heatmapData}
            height={Math.max(380, featureNames.length * 28)}
            colorRange={['#161b22', '#d29922', '#3fb950']}
          />
        ) : (
          <div style={{ color: 'var(--text-tertiary)', textAlign: 'center', padding: 40 }}>No regime-specific importance data</div>
        )}
      </ChartContainer>
    </div>
  )
}
