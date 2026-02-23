import React from 'react'
import { X } from 'lucide-react'

interface LogFiltersProps {
  searchText: string
  onSearchChange: (text: string) => void
  selectedLevels: string[]
  onLevelsChange: (levels: string[]) => void
}

const LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
const LEVEL_COLORS: Record<string, string> = {
  DEBUG: 'var(--text-tertiary)',
  INFO: 'var(--accent-blue)',
  WARNING: 'var(--accent-amber)',
  ERROR: 'var(--accent-red)',
  CRITICAL: 'var(--accent-red)',
}

export default function LogFilters({
  searchText,
  onSearchChange,
  selectedLevels,
  onLevelsChange,
}: LogFiltersProps) {
  const toggleLevel = (level: string) => {
    if (selectedLevels.includes(level)) {
      onLevelsChange(selectedLevels.filter((l) => l !== level))
    } else {
      onLevelsChange([...selectedLevels, level])
    }
  }

  return (
    <div className="flex items-center gap-3 mb-3">
      <input
        type="text"
        value={searchText}
        onChange={(e) => onSearchChange(e.target.value)}
        placeholder="Search logs..."
        className="rounded px-3 py-1.5 font-mono flex-1"
        style={{
          fontSize: 11,
          backgroundColor: 'var(--bg-input)',
          border: '1px solid var(--border)',
          color: 'var(--text-secondary)',
          maxWidth: 300,
        }}
      />
      <div className="flex gap-1">
        {LEVELS.map((level) => (
          <button
            key={level}
            onClick={() => toggleLevel(level)}
            className="px-2 py-1 rounded font-mono"
            style={{
              fontSize: 9,
              backgroundColor: selectedLevels.includes(level) ? LEVEL_COLORS[level] : 'var(--bg-tertiary)',
              color: selectedLevels.includes(level) ? 'var(--text-primary)' : 'var(--text-tertiary)',
              border: '1px solid var(--border-light)',
              cursor: 'pointer',
              opacity: selectedLevels.includes(level) ? 1 : 0.6,
            }}
          >
            {level}
          </button>
        ))}
      </div>
      {(searchText || selectedLevels.length < LEVELS.length) && (
        <button
          onClick={() => { onSearchChange(''); onLevelsChange(LEVELS) }}
          className="flex items-center gap-1 px-2 py-1 rounded"
          style={{
            fontSize: 10,
            backgroundColor: 'var(--bg-tertiary)',
            border: '1px solid var(--border-light)',
            color: 'var(--text-tertiary)',
            cursor: 'pointer',
          }}
        >
          <X size={10} /> Clear
        </button>
      )}
    </div>
  )
}
