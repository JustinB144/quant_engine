import React from 'react'
import MetricCard from '@/components/ui/MetricCard'
import type { DashboardSummary, RegimeInfo } from '@/types/dashboard'
import type { ModelHealth } from '@/types/models'

const REGIME_COLORS: Record<string, string> = {
  'Trending Bull': 'var(--regime-0)',
  'Trending Bear': 'var(--regime-1)',
  'Mean Reverting': 'var(--regime-2)',
  'High Volatility': 'var(--regime-3)',
}

interface KPIGridProps {
  summary?: DashboardSummary
  regime?: RegimeInfo
  modelHealth?: ModelHealth
}

export default function KPIGrid({ summary, regime, modelHealth }: KPIGridProps) {
  if (!summary) return null

  const sharpe = summary.sharpe
  const sColor = sharpe >= 1.0 ? 'var(--accent-green)' : sharpe >= 0.5 ? 'var(--accent-amber)' : 'var(--accent-red)'

  const winRate = summary.win_rate
  const wrColor = winRate >= 0.5 ? 'var(--accent-green)' : winRate >= 0.45 ? 'var(--accent-amber)' : 'var(--accent-red)'

  const annReturn = summary.annualized_return
  const arColor = annReturn >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'

  const regimeLabel = regime?.current_label ?? 'Unavailable'
  const regimeColor = REGIME_COLORS[regimeLabel] ?? 'var(--text-tertiary)'
  const currentProbs = regime?.current_probs ?? {}
  const maxConf = Object.values(currentProbs).length > 0 ? Math.max(...Object.values(currentProbs)) : 0
  const regimeSub = `${(maxConf * 100).toFixed(0)}% confidence  |  ${regime?.as_of ?? '---'}`

  const retrain = modelHealth?.retrain_triggered ?? false
  const rtVal = retrain ? 'TRIGGERED' : 'Stable'
  const rtColor = retrain ? 'var(--accent-red)' : 'var(--accent-green)'
  const reasons = modelHealth?.retrain_reasons ?? []
  const rtSub = reasons.length ? reasons.slice(0, 2).join('; ') : 'No retrain signals'

  const cvg = modelHealth?.cv_gap ?? 0
  const cvgColor = cvg < 0.05 ? 'var(--accent-green)' : cvg < 0.15 ? 'var(--accent-amber)' : 'var(--accent-red)'

  const maxDD = summary.max_drawdown
  const ddColor = maxDD > -0.05 ? 'var(--accent-green)' : maxDD > -0.15 ? 'var(--accent-amber)' : 'var(--accent-red)'

  return (
    <div className="grid grid-cols-4 gap-3 mb-4">
      <MetricCard label="Win Rate" value={`${(winRate * 100).toFixed(1)}%`} color={wrColor} subtitle={`${summary.total_trades} trades`} />
      <MetricCard label="Annual Return" value={`${annReturn >= 0 ? '+' : ''}${(annReturn * 100).toFixed(2)}%`} color={arColor} subtitle={`Profit factor: ${summary.profit_factor.toFixed(2)}`} />
      <MetricCard label="Sharpe Ratio" value={sharpe.toFixed(2)} color={sColor} subtitle={`Sortino: ${summary.sortino.toFixed(2)}`} />
      <MetricCard label="Current Regime" value={regimeLabel} color={regimeColor} subtitle={regimeSub} />
      <MetricCard label="Retrain Trigger" value={rtVal} color={rtColor} subtitle={rtSub} />
      <MetricCard label="CV Gap" value={cvg.toFixed(4)} color={cvgColor} subtitle="IS vs OOS degradation" />
      <MetricCard label="Max Drawdown" value={`${(maxDD * 100).toFixed(2)}%`} color={ddColor} subtitle="Peak-to-trough" />
      <MetricCard label="Avg Return" value={`${(summary.avg_return * 100).toFixed(4)}%`} color={summary.avg_return >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'} subtitle={`${summary.trades_per_year.toFixed(0)} trades/year`} />
    </div>
  )
}
