import React, { useMemo } from 'react'
import { useArbFreeSVI } from '@/api/queries/useIVSurface'
import IVSurfacePlot from '@/components/charts/IVSurfacePlot'
import LineChart from '@/components/charts/LineChart'
import ChartContainer from '@/components/charts/ChartContainer'
import MetricCard from '@/components/ui/MetricCard'

/**
 * Build a 2D meshgrid from 1D row/col arrays (mimics numpy.meshgrid).
 * Returns [colGrid, rowGrid] where each is nRows x nCols.
 */
function meshgrid(cols: number[], rows: number[]): [number[][], number[][]] {
  const colGrid: number[][] = []
  const rowGrid: number[][] = []
  for (let i = 0; i < rows.length; i++) {
    const colRow: number[] = []
    const rowRow: number[] = []
    for (let j = 0; j < cols.length; j++) {
      colRow.push(cols[j])
      rowRow.push(rows[i])
    }
    colGrid.push(colRow)
    rowGrid.push(rowRow)
  }
  return [colGrid, rowGrid]
}

export default function ArbAwareSVITab() {
  const { data: resp, isLoading, error, refetch } = useArbFreeSVI()
  const d = resp?.data

  const grids = useMemo(() => {
    if (!d) return null
    const [mGrid, tGrid] = meshgrid(d.moneyness, d.expiries)
    return { mGrid, tGrid }
  }, [d])

  // Pick the ATM expiry slice (middle expiry index) for the smile comparison chart
  const smileData = useMemo(() => {
    if (!d) return null
    const midIdx = Math.floor(d.expiries.length / 2)
    const expLabel = d.expiries[midIdx].toFixed(3)
    const categories = d.moneyness.map((m) => m.toFixed(2))
    const rawSlice = d.raw_iv[midIdx]
    const adjSlice = d.adj_iv[midIdx]
    const mktSlice = d.market_iv[midIdx]
    return { categories, rawSlice, adjSlice, mktSlice, expLabel }
  }, [d])

  const errorMsg = error instanceof Error ? error.message : error ? String(error) : undefined

  return (
    <div style={{ paddingTop: 16 }}>
      {/* Metric cards row */}
      <ChartContainer
        meta={resp?.meta}
        isLoading={isLoading}
        error={errorMsg}
        onRetry={() => refetch()}
      >
        {d && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
            <MetricCard
              label="Raw Cal. Violation"
              value={d.raw_calendar_violation.toFixed(6)}
              color={d.raw_calendar_violation > 1e-4 ? 'var(--accent-red)' : 'var(--accent-green)'}
              subtitle="max(w_i - w_{i+1})"
            />
            <MetricCard
              label="Adj Cal. Violation"
              value={d.adj_calendar_violation.toFixed(6)}
              color={d.adj_calendar_violation > 1e-4 ? 'var(--accent-red)' : 'var(--accent-green)'}
              subtitle="after monotonicity fix"
            />
            <MetricCard
              label="Expiries"
              value={String(d.n_expiries)}
              color="var(--accent-blue)"
              subtitle="maturity slices"
            />
            <MetricCard
              label="Strikes"
              value={String(d.n_strikes)}
              color="var(--accent-blue)"
              subtitle="moneyness points"
            />
          </div>
        )}
      </ChartContainer>

      {/* Side-by-side surface plots */}
      {d && grids && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
          <ChartContainer title="Raw SVI Surface">
            <IVSurfacePlot
              moneyness={grids.mGrid}
              expiries={grids.tGrid}
              ivGrid={d.raw_iv}
              title="Raw SVI Surface"
              height={480}
            />
          </ChartContainer>
          <ChartContainer title="Arb-Free SVI Surface">
            <IVSurfacePlot
              moneyness={grids.mGrid}
              expiries={grids.tGrid}
              ivGrid={d.adj_iv}
              title="Arb-Free SVI Surface"
              height={480}
            />
          </ChartContainer>
        </div>
      )}

      {/* Smile comparison line chart */}
      {d && smileData && (
        <ChartContainer title={`Smile Comparison (T = ${smileData.expLabel}Y)`}>
          <LineChart
            categories={smileData.categories}
            series={[
              { name: 'Market IV', data: smileData.mktSlice, color: '#8b949e' },
              { name: 'Raw SVI', data: smileData.rawSlice, color: '#58a6ff' },
              { name: 'Arb-Free SVI', data: smileData.adjSlice, color: '#3fb950' },
            ]}
            height={340}
            yAxisName="IV (%)"
          />
        </ChartContainer>
      )}
    </div>
  )
}
