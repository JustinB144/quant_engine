import React from 'react'

interface Tab {
  key: string
  label: string
}

interface TabGroupProps {
  tabs: Tab[]
  activeKey: string
  onChange: (key: string) => void
}

export default function TabGroup({ tabs, activeKey, onChange }: TabGroupProps) {
  return (
    <div
      className="flex gap-0 mb-4"
      style={{ borderBottom: '1px solid var(--border-light)' }}
    >
      {tabs.map((tab) => {
        const isActive = tab.key === activeKey
        return (
          <button
            key={tab.key}
            onClick={() => onChange(tab.key)}
            className="px-4 py-2 transition-colors duration-150"
            style={{
              fontSize: 12,
              fontWeight: isActive ? 600 : 400,
              color: isActive ? 'var(--accent-blue)' : 'var(--text-tertiary)',
              background: 'transparent',
              border: 'none',
              borderBottom: isActive ? '2px solid var(--accent-blue)' : '2px solid transparent',
              cursor: 'pointer',
              marginBottom: -1,
            }}
          >
            {tab.label}
          </button>
        )
      })}
    </div>
  )
}
