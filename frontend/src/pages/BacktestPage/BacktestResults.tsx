import React, { useState } from 'react'
import TabGroup from '@/components/ui/TabGroup'
import ChartContainer from '@/components/charts/ChartContainer'
import MetricCard from '@/components/ui/MetricCard'
import LineChart from '@/components/charts/LineChart'
import TradeTable from '@/components/tables/TradeTable'
import { useLatestBacktest, useTrades, useEquityCurve } from '@/api/queries/useBacktests'
import { useFilterStore } from '@/store/filterStore'

const TABS = [
  { key: 'equity', label: 'Equity Curve' },
  { key: 'risk', label: 'Risk Metrics' },
  { key: 'trades', label: 'Trade Analysis' },
]

/** Display label for walk-forward mode */
function wfLabel(mode: string): string {
  if (mode === 'full') return 'Full Walk-Forward'
  if (mode === 'single_split') return 'Single Split'
  return mode
}

/** Display label for sizing method */
function sizingLabel(method: string): string {
  if (method === 'kelly') return 'Kelly Criterion'
  if (method === 'dynamic') return 'Dynamic'
  if (method === 'static') return 'Static'
  return method
}

export default function BacktestResults() {
  const [tab, setTab] = useState('equity')
  const horizon = useFilterStore((s) => s.horizon)
  const backtest = useLatestBacktest(horizon)
  const trades = useTrades(horizon, 500)
  const equityCurve = useEquityCurve(horizon)

  const bt = backtest.data?.data
  const btMeta = backtest.data?.meta
  const tradeList = trades.data?.data?.trades ?? []
  const ec = equityCurve.data?.data?.points ?? []

  if (!bt) return null

  return (
    <div>
      {/* Transparency: walk-forward mode + sizing method + model version */}
      <div className="flex items-center gap-4 mb-3" style={{ fontSize: 11 }}>
        <div className="flex items-center gap-1.5">
          <span style={{ color: 'var(--text-tertiary)' }}>Walk-Forward:</span>
          <span
            className="font-mono px-2 py-0.5 rounded"
            style={{
              backgroundColor: bt.walk_forward_mode === 'full' ? 'rgba(63,185,80,0.15)' : 'rgba(210,153,34,0.15)',
              color: bt.walk_forward_mode === 'full' ? 'var(--accent-green)' : 'var(--accent-amber)',
            }}
          >
            {wfLabel(bt.walk_forward_mode)}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <span style={{ color: 'var(--text-tertiary)' }}>Sizing:</span>
          <span
            className="font-mono px-2 py-0.5 rounded"
            style={{
              backgroundColor: 'rgba(88,166,255,0.12)',
              color: 'var(--accent-blue)',
            }}
          >
            {sizingLabel(bt.sizing_method)}
          </span>
        </div>
        {bt.model_version && (
          <div className="flex items-center gap-1.5">
            <span style={{ color: 'var(--text-tertiary)' }}>Model:</span>
            <span className="font-mono" style={{ color: 'var(--text-secondary)' }}>
              {bt.model_version.slice(0, 8)}
            </span>
          </div>
        )}
        {bt.retrain_overdue && (
          <span
            className="font-mono uppercase px-2 py-0.5 rounded"
            style={{
              backgroundColor: 'rgba(248,81,73,0.15)',
              color: 'var(--accent-red)',
              fontWeight: 600,
              fontSize: 10,
            }}
          >
            Retrain Overdue
          </span>
        )}
      </div>

      <TabGroup tabs={TABS} activeKey={tab} onChange={setTab} />

      {tab === 'equity' && (
        <ChartContainer title="Equity Curve" meta={equityCurve.data?.meta ?? btMeta} isLoading={equityCurve.isLoading}>
          <LineChart
            categories={ec.map((p) => p.date)}
            series={[{ name: 'Portfolio', data: ec.map((p) => p.value), color: '#58a6ff' }]}
            height={400}
            yAxisName="Portfolio Value"
            showRangeSelector
          />
        </ChartContainer>
      )}

      {tab === 'risk' && (
        <div>
          <div className="grid grid-cols-4 gap-3 mb-3">
            <MetricCard label="Sharpe" value={bt.sharpe.toFixed(2)} color={bt.sharpe >= 1 ? 'var(--accent-green)' : 'var(--accent-amber)'} />
            <MetricCard label="Sortino" value={bt.sortino.toFixed(2)} />
            <MetricCard label="Annual Return" value={`${(bt.annualized_return * 100).toFixed(1)}%`} color={bt.annualized_return >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'} />
            <MetricCard label="Max Drawdown" value={`${(bt.max_drawdown * 100).toFixed(1)}%`} color="var(--accent-red)" />
          </div>
          <div className="grid grid-cols-4 gap-3 mb-3">
            <MetricCard label="Win Rate" value={`${(bt.win_rate * 100).toFixed(1)}%`} color={bt.win_rate >= 0.5 ? 'var(--accent-green)' : 'var(--accent-amber)'} />
            <MetricCard label="Profit Factor" value={bt.profit_factor.toFixed(2)} color={bt.profit_factor >= 1 ? 'var(--accent-green)' : 'var(--accent-red)'} />
            <MetricCard label="Avg Return" value={`${(bt.avg_return * 100).toFixed(4)}%`} />
            <MetricCard label="Trades" value={String(bt.total_trades)} />
          </div>
        </div>
      )}

      {tab === 'trades' && (
        <ChartContainer title={`Trade Analysis (${tradeList.length} trades)`} meta={trades.data?.meta} isLoading={trades.isLoading}>
          <TradeTable data={tradeList} pageSize={25} />
        </ChartContainer>
      )}
    </div>
  )
}
