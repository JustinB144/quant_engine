import React from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import LogTable from '@/components/tables/LogTable'
import { useLogStream } from '@/hooks/useLogStream'
import type { LogEntry } from '@/api/queries/useLogs'

interface Props {
  searchText: string
  selectedLevels: string[]
}

export default function LogStream({ searchText, selectedLevels }: Props) {
  const { logs, isLoading, meta } = useLogStream(500)

  const filtered = logs.filter((log: LogEntry) => {
    if (!selectedLevels.includes(log.level)) return false
    if (searchText && !log.message.toLowerCase().includes(searchText.toLowerCase()) && !log.logger.toLowerCase().includes(searchText.toLowerCase())) return false
    return true
  })

  return (
    <ChartContainer
      title={`Log Stream (${filtered.length} entries, auto-refresh 3s)`}
      meta={meta}
      isLoading={isLoading && logs.length === 0}
      isEmpty={filtered.length === 0}
      emptyMessage="No log entries match the current filters"
    >
      <LogTable data={filtered} />
    </ChartContainer>
  )
}
