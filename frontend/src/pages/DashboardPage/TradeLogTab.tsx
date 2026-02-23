import React from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import TradeTable from '@/components/tables/TradeTable'
import type { TradesResponse } from '@/types/backtests'
import type { ResponseMeta } from '@/types/api'

interface Props {
  trades?: TradesResponse
  meta?: ResponseMeta
}

export default function TradeLogTab({ trades, meta }: Props) {
  const tradeList = trades?.trades ?? []
  const total = trades?.total ?? 0
  const subtitle = total > 0
    ? `Showing ${Math.min(tradeList.length, total)} of ${total} trades`
    : 'No trades'

  return (
    <ChartContainer title={`Trade Log  |  ${subtitle}`} meta={meta}>
      {tradeList.length > 0 ? (
        <TradeTable data={tradeList} pageSize={25} />
      ) : (
        <div style={{ color: 'var(--text-tertiary)', textAlign: 'center', padding: 40 }}>
          No trade records available.
        </div>
      )}
    </ChartContainer>
  )
}
