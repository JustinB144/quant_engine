import React, { useState } from 'react'
import PageContainer from '@/components/layout/PageContainer'
import PageHeader from '@/components/ui/PageHeader'
import LogFilters from './LogsPage/LogFilters'
import LogStream from './LogsPage/LogStream'

const ALL_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

export default function LogsPage() {
  const [searchText, setSearchText] = useState('')
  const [selectedLevels, setSelectedLevels] = useState(ALL_LEVELS)

  return (
    <PageContainer>
      <PageHeader title="System Logs" subtitle="Real-time log monitoring with auto-refresh" />
      <LogFilters
        searchText={searchText}
        onSearchChange={setSearchText}
        selectedLevels={selectedLevels}
        onLevelsChange={setSelectedLevels}
      />
      <LogStream searchText={searchText} selectedLevels={selectedLevels} />
    </PageContainer>
  )
}
