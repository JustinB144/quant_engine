import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import ChartContainer from '@/components/charts/ChartContainer'
import DualAxisChart from '@/components/charts/DualAxisChart'
import AreaChart from '@/components/charts/AreaChart'
import MetricCard from '@/components/ui/MetricCard'
import { bloombergDarkTheme } from '@/components/charts/theme'
import { useReturnsDistribution, useRollingRisk } from '@/api/queries/useDashboard'
import type { DashboardSummary } from '@/types/dashboard'
import type { ResponseMeta } from '@/types/api'

interface Props {
  summary: DashboardSummary
  meta?: ResponseMeta
}

export default function RiskTab({ summary, meta }: Props) {
  const { data: distData, isLoading: distLoading } = useReturnsDistribution()
  const { data: riskData, isLoading: riskLoading } = useRollingRisk()
  const dist = distData?.data
  const rolling = riskData?.data

  // Colors
  const arColor = summary.annualized_return >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'
  const sColor = summary.sharpe >= 1.0 ? 'var(--accent-green)' : summary.sharpe >= 0.5 ? 'var(--accent-amber)' : 'var(--accent-red)'
  const ddColor = summary.max_drawdown > -0.05 ? 'var(--accent-green)' : summary.max_drawdown > -0.15 ? 'var(--accent-amber)' : 'var(--accent-red)'

  // ── Return Distribution histogram (pre-binned) with VaR markLines ──
  const histogramOption = useMemo(() => {
    if (!dist || dist.bins.length === 0) return null

    const labels = dist.bins.map(b => `${(b.x * 100).toFixed(2)}%`)
    const counts = dist.bins.map(b => b.count)

    // Find closest bin index for a given value
    const findBinIdx = (val: number): number => {
      let closest = 0
      let minDist = Math.abs(dist.bins[0].x - val)
      for (let i = 1; i < dist.bins.length; i++) {
        const d = Math.abs(dist.bins[i].x - val)
        if (d < minDist) {
          minDist = d
          closest = i
        }
      }
      return closest
    }

    return {
      ...bloombergDarkTheme,
      tooltip: { ...bloombergDarkTheme.tooltip, trigger: 'axis' as const },
      grid: { left: 50, right: 20, top: 20, bottom: 30 },
      xAxis: {
        type: 'category' as const,
        data: labels,
        ...bloombergDarkTheme.xAxis,
        axisLabel: {
          ...bloombergDarkTheme.xAxis.axisLabel,
          rotate: 45,
          interval: Math.max(1, Math.floor(dist.bins.length / 10)),
        },
        name: 'Daily Return',
      },
      yAxis: {
        type: 'value' as const,
        name: 'Frequency',
        ...bloombergDarkTheme.yAxis,
      },
      series: [
        {
          type: 'bar' as const,
          data: counts,
          itemStyle: { color: '#58a6ff', opacity: 0.75 },
          barCategoryGap: '5%',
          markLine: {
            symbol: 'none',
            data: [
              {
                xAxis: findBinIdx(dist.var95),
                label: { formatter: 'VaR 95%', color: '#d29922', fontSize: 10 },
                lineStyle: { color: '#d29922', type: 'dashed' as const, width: 2 },
              },
              {
                xAxis: findBinIdx(dist.var99),
                label: { formatter: 'VaR 99%', color: '#f85149', fontSize: 10 },
                lineStyle: { color: '#f85149', type: 'dashed' as const, width: 2 },
              },
              {
                xAxis: findBinIdx(dist.cvar95),
                label: { formatter: 'CVaR 95%', color: '#d29922', fontSize: 10, position: 'insideEndBottom' },
                lineStyle: { color: '#d29922', type: 'dotted' as const, width: 1.5 },
              },
            ],
          },
        },
      ],
    }
  }, [dist])

  // ── Rolling Risk: align vol + sharpe to unified date axis ──
  const rollingCategories = useMemo(() => {
    if (!rolling) return []
    // Use the longer series (vol has more points because smaller window)
    const allDates = new Set<string>()
    rolling.rolling_vol.forEach(p => allDates.add(p.date))
    rolling.rolling_sharpe.forEach(p => allDates.add(p.date))
    return Array.from(allDates).sort()
  }, [rolling])

  const rollingVolMap = useMemo(() => {
    if (!rolling) return new Map<string, number>()
    return new Map(rolling.rolling_vol.map(p => [p.date, p.value]))
  }, [rolling])

  const rollingSharpeMap = useMemo(() => {
    if (!rolling) return new Map<string, number>()
    return new Map(rolling.rolling_sharpe.map(p => [p.date, p.value]))
  }, [rolling])

  // ── Drawdown: separate AreaChart ──
  const drawdownCategories = useMemo(() => {
    if (!rolling) return []
    return rolling.drawdown.map(p => p.date)
  }, [rolling])

  const drawdownData = useMemo(() => {
    if (!rolling) return []
    return rolling.drawdown.map(p => p.value)
  }, [rolling])

  // ── Risk summary text ──
  const riskSummary = [
    'Risk Summary',
    '='.repeat(40),
    `Annualized Return:   ${summary.annualized_return >= 0 ? '+' : ''}${(summary.annualized_return * 100).toFixed(2)}%`,
    `Sharpe Ratio:        ${summary.sharpe.toFixed(4)}`,
    `Sortino Ratio:       ${summary.sortino.toFixed(4)}`,
    `Max Drawdown:        ${(summary.max_drawdown * 100).toFixed(2)}%`,
    `Profit Factor:       ${summary.profit_factor.toFixed(4)}`,
    ...(dist ? [
      '',
      'Value at Risk',
      '-'.repeat(40),
      `VaR 95%:             ${(dist.var95 * 100).toFixed(3)}%`,
      `VaR 99%:             ${(dist.var99 * 100).toFixed(3)}%`,
      `CVaR 95%:            ${(dist.cvar95 * 100).toFixed(3)}%`,
      `CVaR 99%:            ${(dist.cvar99 * 100).toFixed(3)}%`,
      '',
      'Distribution Stats',
      '-'.repeat(40),
      `Mean Daily Return:   ${(dist.mean * 100).toFixed(4)}%`,
      `Std Dev (Daily):     ${(dist.std * 100).toFixed(4)}%`,
      `Skewness:            ${dist.skew.toFixed(4)}`,
      `Kurtosis:            ${dist.kurtosis.toFixed(4)}`,
      `Observations:        ${dist.count}`,
    ] : []),
    '',
    'Trade Statistics',
    '-'.repeat(40),
    `Total Trades:        ${summary.total_trades}`,
    `Win Rate:            ${(summary.win_rate * 100).toFixed(1)}%`,
    `Avg Return/Trade:    ${(summary.avg_return * 100).toFixed(4)}%`,
    `Trades/Year:         ${summary.trades_per_year.toFixed(0)}`,
    '',
    'Regime Breakdown',
    '-'.repeat(40),
    ...Object.entries(summary.regime_breakdown).map(([regime, stats]) =>
      `Regime ${regime}: ${stats.n_trades} trades, Win ${(stats.win_rate * 100).toFixed(1)}%, Sharpe ${stats.sharpe.toFixed(2)}`
    ),
  ].join('\n')

  return (
    <div>
      {/* ── Metric cards row ── */}
      <div className="grid grid-cols-4 gap-3 mb-3">
        <MetricCard label="Annual Return" value={`${(summary.annualized_return * 100).toFixed(2)}%`} color={arColor} />
        <MetricCard label="Sharpe" value={summary.sharpe.toFixed(2)} color={sColor} />
        <MetricCard label="Sortino" value={summary.sortino.toFixed(2)} color={summary.sortino >= 1 ? 'var(--accent-green)' : 'var(--accent-amber)'} />
        <MetricCard label="Max Drawdown" value={`${(summary.max_drawdown * 100).toFixed(2)}%`} color={ddColor} />
      </div>

      {/* ── Distribution + Rolling Risk side by side ── */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        {/* Return Distribution Histogram with VaR lines */}
        <ChartContainer title="Return Distribution" isLoading={distLoading}>
          {histogramOption ? (
            <ReactECharts option={histogramOption} style={{ height: 300 }} notMerge />
          ) : (
            <div style={{ color: 'var(--text-tertiary)', textAlign: 'center', padding: 40 }}>
              {distLoading ? 'Loading distribution data...' : 'No distribution data available'}
            </div>
          )}
        </ChartContainer>

        {/* Rolling Volatility + Rolling Sharpe (dual axis) */}
        <ChartContainer title="Rolling Risk Metrics" isLoading={riskLoading}>
          {rolling && rollingCategories.length > 0 ? (
            <DualAxisChart
              categories={rollingCategories}
              series={[
                {
                  name: `Rolling Vol ${rolling.vol_window}D`,
                  data: rollingCategories.map(d => rollingVolMap.get(d) ?? null),
                  yAxisIndex: 0,
                  color: '#f85149',
                },
                {
                  name: `Rolling Sharpe ${rolling.sharpe_window}D`,
                  data: rollingCategories.map(d => rollingSharpeMap.get(d) ?? null),
                  yAxisIndex: 1,
                  color: '#58a6ff',
                },
              ]}
              height={300}
              yAxisNames={['Volatility', 'Sharpe']}
            />
          ) : (
            <div style={{ color: 'var(--text-tertiary)', textAlign: 'center', padding: 40 }}>
              {riskLoading ? 'Loading rolling risk data...' : 'No rolling risk data available'}
            </div>
          )}
        </ChartContainer>
      </div>

      {/* ── Drawdown ── */}
      {rolling && drawdownCategories.length > 0 && (
        <div className="mb-3">
          <ChartContainer title="Drawdown">
            <AreaChart
              categories={drawdownCategories}
              series={[
                {
                  name: 'Drawdown',
                  data: drawdownData,
                  color: '#f85149',
                  stack: 'dd',
                },
              ]}
              height={200}
              yAxisName="Drawdown"
              showRangeSelector
            />
          </ChartContainer>
        </div>
      )}

      {/* ── Text risk summary ── */}
      <ChartContainer title="Risk Summary" meta={meta}>
        <pre className="font-mono" style={{ color: 'var(--text-secondary)', fontSize: 12, whiteSpace: 'pre-wrap', margin: 0 }}>
          {riskSummary}
        </pre>
      </ChartContainer>
    </div>
  )
}
