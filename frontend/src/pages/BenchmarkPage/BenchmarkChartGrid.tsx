import React, { useMemo, useEffect, useState } from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import LineChart from '@/components/charts/LineChart'
import AreaChart from '@/components/charts/AreaChart'
import type { BenchmarkData, BenchmarkEquityCurves, BenchmarkRollingMetrics } from '@/api/queries/useBenchmark'
import type { ResponseMeta } from '@/types/api'
import type { TimeSeriesPoint } from '@/types/dashboard'

interface Props {
  comparison: BenchmarkData
  equityCurves?: BenchmarkEquityCurves
  rollingMetrics?: BenchmarkRollingMetrics
  meta?: ResponseMeta
}

/** Staggered fade-in animation for progressive chart reveal */
function useStaggeredReveal(count: number, delayMs = 120) {
  const [visible, setVisible] = useState<boolean[]>(Array(count).fill(false))
  useEffect(() => {
    const timers: ReturnType<typeof setTimeout>[] = []
    for (let i = 0; i < count; i++) {
      timers.push(setTimeout(() => {
        setVisible((prev) => {
          const next = [...prev]
          next[i] = true
          return next
        })
      }, i * delayMs))
    }
    return () => timers.forEach(clearTimeout)
  }, [count, delayMs])
  return visible
}

const CHART_TRANSITION: React.CSSProperties = {
  transition: 'opacity 0.4s ease-out, transform 0.4s ease-out',
}

function chartStyle(isVisible: boolean): React.CSSProperties {
  return {
    ...CHART_TRANSITION,
    opacity: isVisible ? 1 : 0,
    transform: isVisible ? 'translateY(0)' : 'translateY(12px)',
  }
}

/** Extract categories (date strings) and values from a TimeSeriesPoint array */
function unzip(points: TimeSeriesPoint[]): { categories: string[]; data: number[] } {
  const categories: string[] = []
  const data: number[] = []
  for (const p of points) {
    categories.push(p.date)
    data.push(p.value)
  }
  return { categories, data }
}

export default function BenchmarkChartGrid({ comparison, equityCurves, rollingMetrics, meta }: Props) {
  // ---------- Equity Curves ----------
  const equity = useMemo(() => {
    if (!equityCurves) return null
    const strat = unzip(equityCurves.strategy)
    const bench = unzip(equityCurves.benchmark)
    return {
      categories: strat.categories,
      strategy: strat.data,
      benchmark: bench.data,
    }
  }, [equityCurves])

  // ---------- Rolling Correlation ----------
  const correlation = useMemo(() => {
    if (!rollingMetrics) return null
    const { categories, data } = unzip(rollingMetrics.rolling_correlation)
    return { categories, data }
  }, [rollingMetrics])

  // ---------- Rolling Alpha & Beta ----------
  const alphaBeta = useMemo(() => {
    if (!rollingMetrics) return null
    const alpha = unzip(rollingMetrics.rolling_alpha)
    const beta = unzip(rollingMetrics.rolling_beta)
    return {
      categories: alpha.categories,
      alpha: alpha.data,
      beta: beta.data,
    }
  }, [rollingMetrics])

  // ---------- Relative Strength ----------
  const relStrength = useMemo(() => {
    if (!rollingMetrics) return null
    const { categories, data } = unzip(rollingMetrics.relative_strength)
    return { categories, data }
  }, [rollingMetrics])

  // ---------- Drawdown Comparison ----------
  const drawdowns = useMemo(() => {
    if (!rollingMetrics) return null
    const strat = unzip(rollingMetrics.drawdown_strategy)
    const bench = unzip(rollingMetrics.drawdown_benchmark)
    return {
      categories: strat.categories,
      strategy: strat.data,
      benchmark: bench.data,
    }
  }, [rollingMetrics])

  const windowLabel = rollingMetrics?.window ? `${rollingMetrics.window}D` : '60D'

  // 6 sections: equity, corr, alpha/beta, rel strength, drawdown, summary
  const reveal = useStaggeredReveal(6, 120)

  return (
    <div className="space-y-3 mt-4">
      {/* 1. Equity Comparison - full width */}
      <div style={chartStyle(reveal[0])}>
        <ChartContainer title="Equity Comparison" meta={meta}>
          {equity ? (
            <LineChart
              categories={equity.categories}
              series={[
                { name: 'Strategy', data: equity.strategy, color: '#58a6ff' },
                { name: 'SPY', data: equity.benchmark, color: '#8b949e' },
              ]}
              height={420}
              yAxisName="Portfolio Value"
              showRangeSelector
            />
          ) : (
            <div style={{ color: 'var(--text-tertiary)', textAlign: 'center', padding: 40 }}>
              Loading equity curves...
            </div>
          )}
        </ChartContainer>
      </div>

      {/* 2x2 grid for the remaining 4 charts */}
      <div className="grid grid-cols-2 gap-3">
        {/* 2. Rolling Correlation */}
        <div style={chartStyle(reveal[1])}>
          <ChartContainer title={`Rolling ${windowLabel} Correlation`} meta={meta}>
            {correlation ? (
              <LineChart
                categories={correlation.categories}
                series={[
                  { name: 'Correlation', data: correlation.data, color: '#bc8cff' },
                ]}
                height={300}
                yAxisName="Correlation"
                yAxisRange={[-1, 1]}
              />
            ) : (
              <div style={{ color: 'var(--text-tertiary)', textAlign: 'center', padding: 40 }}>
                Loading rolling metrics...
              </div>
            )}
          </ChartContainer>
        </div>

        {/* 3. Rolling Alpha & Beta */}
        <div style={chartStyle(reveal[2])}>
          <ChartContainer title={`Rolling ${windowLabel} Alpha & Beta`} meta={meta}>
            {alphaBeta ? (
              <LineChart
                categories={alphaBeta.categories}
                series={[
                  { name: 'Alpha', data: alphaBeta.alpha, color: '#3fb950' },
                  { name: 'Beta', data: alphaBeta.beta, color: '#39d2c0' },
                ]}
                height={300}
                yAxisName="Value"
              />
            ) : (
              <div style={{ color: 'var(--text-tertiary)', textAlign: 'center', padding: 40 }}>
                Loading rolling metrics...
              </div>
            )}
          </ChartContainer>
        </div>

        {/* 4. Relative Strength */}
        <div style={chartStyle(reveal[3])}>
          <ChartContainer title="Relative Strength (Strategy / Benchmark)" meta={meta}>
            {relStrength ? (
              <LineChart
                categories={relStrength.categories}
                series={[
                  { name: 'Relative Strength', data: relStrength.data, color: '#d29922' },
                ]}
                height={300}
                yAxisName="Ratio"
              />
            ) : (
              <div style={{ color: 'var(--text-tertiary)', textAlign: 'center', padding: 40 }}>
                Loading rolling metrics...
              </div>
            )}
          </ChartContainer>
        </div>

        {/* 5. Drawdown Comparison */}
        <div style={chartStyle(reveal[4])}>
          <ChartContainer title="Drawdown Comparison" meta={meta}>
            {drawdowns ? (
              <AreaChart
                categories={drawdowns.categories}
                series={[
                  { name: 'Strategy DD', data: drawdowns.strategy, color: '#f85149', stack: 'strat' },
                  { name: 'Benchmark DD', data: drawdowns.benchmark, color: '#8b949e', stack: 'bench' },
                ]}
                height={300}
                yAxisName="Drawdown"
              />
            ) : (
              <div style={{ color: 'var(--text-tertiary)', textAlign: 'center', padding: 40 }}>
                Loading rolling metrics...
              </div>
            )}
          </ChartContainer>
        </div>
      </div>

      {/* Data point summary row */}
      <div style={chartStyle(reveal[5])}>
        <div className="card-panel">
          <div className="card-panel-header">Data Points</div>
          <pre className="font-mono" style={{ fontSize: 11, color: 'var(--text-secondary)', margin: 0 }}>
            Strategy: {comparison.strategy_points} observations | Benchmark: {comparison.benchmark_points} observations
            {equityCurves ? ` | Equity Curve: ${equityCurves.points} points` : ''}
            {rollingMetrics ? ` | Rolling Window: ${rollingMetrics.window}D over ${rollingMetrics.points} points` : ''}
          </pre>
        </div>
      </div>
    </div>
  )
}
