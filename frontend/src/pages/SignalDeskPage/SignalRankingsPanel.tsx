import React from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import SignalTable from '@/components/tables/SignalTable'
import type { SignalRow } from '@/types/signals'
import type { ResponseMeta } from '@/types/api'

interface Props {
  signals: SignalRow[]
  meta?: ResponseMeta
}

export default function SignalRankingsPanel({ signals, meta }: Props) {
  return (
    <ChartContainer
      title={`Signal Rankings (${signals.length} signals)`}
      meta={meta}
      isEmpty={signals.length === 0}
      emptyMessage="No signals available. Generate predictions first."
    >
      <SignalTable data={signals} />
    </ChartContainer>
  )
}
