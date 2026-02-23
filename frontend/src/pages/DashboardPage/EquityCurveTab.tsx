import React, { useMemo } from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import LineChart from '@/components/charts/LineChart'
import BarChart from '@/components/charts/BarChart'
import { useEquityWithBenchmark, useAttribution } from '@/api/queries/useDashboard'
import type { EquityCurveResponse } from '@/types/backtests'
import type { ResponseMeta } from '@/types/api'

interface Props {
  equityCurve?: EquityCurveResponse
  meta?: ResponseMeta
}

export default function EquityCurveTab({ equityCurve, meta }: Props) {
  const { data: eqBenchData, isLoading: eqBenchLoading } = useEquityWithBenchmark()
  const { data: attrData, isLoading: attrLoading } = useAttribution()
  const eqBench = eqBenchData?.data
  const attr = attrData?.data

  // Prefer the equity-with-benchmark endpoint (has both series).
  // Fall back to the plain equity curve if benchmark endpoint hasn't loaded.
  const hasOverlay = !!(eqBench && eqBench.strategy.length > 0)
  const strategyPoints = hasOverlay ? eqBench.strategy : (equityCurve?.points ?? [])
  const benchmarkPoints = hasOverlay ? eqBench.benchmark : []

  const categories = useMemo(
    () => strategyPoints.map(p => p.date),
    [strategyPoints],
  )

  const benchMap = useMemo(
    () => new Map(benchmarkPoints.map(p => [p.date, p.value])),
    [benchmarkPoints],
  )

  const series = useMemo(() => {
    const s: Array<{ name: string; data: (number | null)[]; color: string; dash?: boolean }> = [
      { name: 'Strategy', data: strategyPoints.map(p => p.value), color: '#58a6ff' },
    ]
    if (benchmarkPoints.length > 0) {
      s.push({
        name: 'SPY Benchmark',
        data: categories.map(d => benchMap.get(d) ?? null),
        color: '#8b949e',
        dash: true,
      })
    }
    return s
  }, [strategyPoints, benchmarkPoints, categories, benchMap])

  // Attribution chart data
  const attrCategories = useMemo(() => {
    if (!attr) return []
    return [...attr.factors.map(f => f.name), 'Alpha']
  }, [attr])

  const attrColors = useMemo(() => {
    if (!attr) return []
    return [
      ...attr.factors.map(f => f.annualized_contribution >= 0 ? '#3fb950' : '#f85149'),
      attr.residual_alpha >= 0 ? '#3fb950' : '#f85149',
    ]
  }, [attr])

  const attrValues = useMemo(() => {
    if (!attr) return []
    return [
      ...attr.factors.map(f => Number((f.annualized_contribution * 100).toFixed(2))),
      Number((attr.residual_alpha * 100).toFixed(2)),
    ]
  }, [attr])

  const attrSummary = useMemo(() => {
    if (!attr) return ''
    return [
      'Factor Attribution Analysis',
      '='.repeat(40),
      ...attr.factors.map(f =>
        `${f.name.padEnd(16)} beta=${f.coefficient.toFixed(4)}  contribution=${(f.annualized_contribution * 100).toFixed(2)}%`
      ),
      '',
      `Residual Alpha:  ${(attr.residual_alpha * 100).toFixed(2)}% (annualized)`,
      `R-squared:       ${(attr.r_squared * 100).toFixed(1)}%`,
      `Observations:    ${attr.points}`,
    ].join('\n')
  }, [attr])

  const isEquityLoading = eqBenchLoading && strategyPoints.length === 0

  return (
    <div>
      {/* ── Equity Curve with Benchmark Overlay ── */}
      <ChartContainer title="Equity Curve" meta={meta} isLoading={isEquityLoading}>
        {strategyPoints.length > 0 ? (
          <LineChart
            categories={categories}
            series={series}
            height={420}
            yAxisName="Cumulative Return"
            showRangeSelector
          />
        ) : (
          <div style={{ color: 'var(--text-tertiary)', textAlign: 'center', padding: 40 }}>
            No equity curve data available
          </div>
        )}
      </ChartContainer>

      {/* ── Factor Attribution ── */}
      {attr && attr.factors.length > 0 && (
        <div className="mt-3">
          <ChartContainer title="Factor Attribution" isLoading={attrLoading}>
            <div className="grid grid-cols-2 gap-4">
              <BarChart
                categories={attrCategories}
                series={[{
                  name: 'Annualized Contribution (%)',
                  data: attrValues,
                  color: attrColors,
                }]}
                height={250}
                horizontal
                xAxisName="Contribution (%)"
                showValues
              />
              <pre
                className="font-mono"
                style={{
                  color: 'var(--text-secondary)',
                  fontSize: 12,
                  margin: 0,
                  whiteSpace: 'pre-wrap',
                }}
              >
                {attrSummary}
              </pre>
            </div>
          </ChartContainer>
        </div>
      )}

      {/* Show loading placeholder for attribution while it fetches */}
      {attrLoading && !attr && (
        <div className="mt-3">
          <ChartContainer title="Factor Attribution" isLoading>
            <div />
          </ChartContainer>
        </div>
      )}
    </div>
  )
}
