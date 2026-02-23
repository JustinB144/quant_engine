import React from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import MetricCard from '@/components/ui/MetricCard'
import type { DashboardSummary } from '@/types/dashboard'
import type { ResponseMeta } from '@/types/api'

interface Props {
  summary: DashboardSummary
  meta?: ResponseMeta
}

export default function RiskTab({ summary, meta }: Props) {
  const arColor = summary.annualized_return >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'
  const sColor = summary.sharpe >= 1.0 ? 'var(--accent-green)' : summary.sharpe >= 0.5 ? 'var(--accent-amber)' : 'var(--accent-red)'
  const ddColor = summary.max_drawdown > -0.05 ? 'var(--accent-green)' : summary.max_drawdown > -0.15 ? 'var(--accent-amber)' : 'var(--accent-red)'

  const riskSummary = [
    'Risk Summary',
    '='.repeat(40),
    `Annualized Return:   ${summary.annualized_return >= 0 ? '+' : ''}${(summary.annualized_return * 100).toFixed(2)}%`,
    `Sharpe Ratio:        ${summary.sharpe.toFixed(4)}`,
    `Sortino Ratio:       ${summary.sortino.toFixed(4)}`,
    `Max Drawdown:        ${(summary.max_drawdown * 100).toFixed(2)}%`,
    `Profit Factor:       ${summary.profit_factor.toFixed(4)}`,
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
      <div className="grid grid-cols-4 gap-3 mb-3">
        <MetricCard label="Annual Return" value={`${(summary.annualized_return * 100).toFixed(2)}%`} color={arColor} />
        <MetricCard label="Sharpe" value={summary.sharpe.toFixed(2)} color={sColor} />
        <MetricCard label="Sortino" value={summary.sortino.toFixed(2)} color={summary.sortino >= 1 ? 'var(--accent-green)' : 'var(--accent-amber)'} />
        <MetricCard label="Max Drawdown" value={`${(summary.max_drawdown * 100).toFixed(2)}%`} color={ddColor} />
      </div>

      <ChartContainer title="Risk Summary" meta={meta}>
        <pre className="font-mono" style={{ color: 'var(--text-secondary)', fontSize: 12, whiteSpace: 'pre-wrap', margin: 0 }}>
          {riskSummary}
        </pre>
      </ChartContainer>
    </div>
  )
}
