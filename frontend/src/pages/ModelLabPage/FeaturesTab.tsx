import React from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import BarChart from '@/components/charts/BarChart'
import HeatmapChart from '@/components/charts/HeatmapChart'
import { useFeatureImportance } from '@/api/queries/useModels'

export default function FeaturesTab() {
  const { data, isLoading } = useFeatureImportance()
  const fi = data?.data

  const globalImp = fi?.global_importance ?? {}
  const entries = Object.entries(globalImp)
    .sort(([, a], [, b]) => a - b)
    .slice(-20)

  // Regime heatmap: Record<string, Record<string, number>>
  const regimeHeatmap = fi?.regime_heatmap ?? {}
  const regimeNames = Object.keys(regimeHeatmap)
  const featureSet = new Set<string>()
  for (const regime of regimeNames) {
    for (const feature of Object.keys(regimeHeatmap[regime])) {
      featureSet.add(feature)
    }
  }
  const featureNames = Array.from(featureSet).slice(0, 20)
  const heatmapData = featureNames.map((feature) =>
    regimeNames.map((regime) => regimeHeatmap[regime][feature] ?? 0)
  )

  return (
    <div className="grid grid-cols-2 gap-3">
      <ChartContainer title="Global Feature Importance (Top 20)" meta={data?.meta} isLoading={isLoading}>
        {entries.length > 0 ? (
          <BarChart
            categories={entries.map(([name]) => name)}
            series={[{ name: 'Importance', data: entries.map(([, val]) => val), color: '#58a6ff' }]}
            horizontal
            height={Math.max(400, entries.length * 24)}
            showValues
          />
        ) : (
          <div style={{ color: 'var(--text-tertiary)', textAlign: 'center', padding: 40 }}>No feature importance data</div>
        )}
      </ChartContainer>
      <ChartContainer title="Regime Feature Heatmap" isLoading={isLoading}>
        {featureNames.length > 0 && regimeNames.length > 0 ? (
          <HeatmapChart
            xLabels={regimeNames}
            yLabels={featureNames}
            data={heatmapData}
            height={Math.max(380, featureNames.length * 28)}
            colorRange={['#161b22', '#d29922', '#3fb950']}
          />
        ) : (
          <div style={{ color: 'var(--text-tertiary)', textAlign: 'center', padding: 40 }}>No regime heatmap data</div>
        )}
      </ChartContainer>
    </div>
  )
}
