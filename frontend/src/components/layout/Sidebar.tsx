import React from 'react'
import { NavLink } from 'react-router-dom'
import {
  ChartLine,
  HeartPulse,
  Terminal,
  Database,
  FlaskConical,
  Signal,
  AreaChart,
  Layers,
  Scale,
  Bot,
  Zap,
} from 'lucide-react'
import { useUIStore } from '@/store/uiStore'

const NAV_ITEMS = [
  { label: 'Dashboard', path: '/', icon: ChartLine },
  { label: 'System Health', path: '/system-health', icon: HeartPulse },
  { label: 'System Logs', path: '/system-logs', icon: Terminal },
  { label: 'Data Explorer', path: '/data-explorer', icon: Database },
  { label: 'Model Lab', path: '/model-lab', icon: FlaskConical },
  { label: 'Signal Desk', path: '/signal-desk', icon: Signal },
  { label: 'Backtest & Risk', path: '/backtest-risk', icon: AreaChart },
  { label: 'IV Surface', path: '/iv-surface', icon: Layers },
  { label: 'S&P Comparison', path: '/sp-comparison', icon: Scale },
  { label: 'Autopilot & Events', path: '/autopilot', icon: Bot },
] as const

export default function Sidebar() {
  const collapsed = useUIStore((s) => s.sidebarCollapsed)

  if (collapsed) return null

  return (
    <aside
      className="fixed top-0 left-0 h-screen flex flex-col z-50"
      style={{
        width: 240,
        backgroundColor: 'var(--bg-sidebar)',
        borderRight: '1px solid var(--border-light)',
      }}
    >
      {/* Logo */}
      <div className="px-4 pt-5 pb-2">
        <div className="flex items-center">
          <Zap size={16} style={{ color: 'var(--accent-blue)', marginRight: 8 }} />
          <span
            className="font-bold"
            style={{
              fontSize: 15,
              color: 'var(--accent-blue)',
              letterSpacing: '1.5px',
            }}
          >
            QUANT ENGINE
          </span>
        </div>
        <div
          style={{
            fontSize: 10,
            color: 'var(--text-tertiary)',
            marginTop: 4,
            letterSpacing: '0.3px',
          }}
        >
          Professional Trading System
        </div>
      </div>

      {/* Separator */}
      <hr
        style={{
          borderColor: 'var(--border)',
          margin: '12px 16px 16px',
          borderWidth: '0.5px',
        }}
      />

      {/* Section label */}
      <div
        className="font-mono"
        style={{
          fontSize: 9,
          color: 'var(--text-tertiary)',
          letterSpacing: '1.5px',
          fontWeight: 600,
          padding: '0 16px 8px',
        }}
      >
        NAVIGATION
      </div>

      {/* Nav items */}
      <nav className="flex-1 overflow-y-auto">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === '/'}
            className="no-underline"
            style={{ textDecoration: 'none' }}
          >
            {({ isActive }) => (
              <div
                className="flex items-center mx-2 px-3 py-2 rounded-md transition-colors duration-150"
                style={{
                  backgroundColor: isActive ? 'var(--bg-active)' : 'transparent',
                  borderLeft: isActive ? '2px solid var(--accent-blue)' : '2px solid transparent',
                  cursor: 'pointer',
                }}
                onMouseEnter={(e) => {
                  if (!isActive)
                    (e.currentTarget as HTMLDivElement).style.backgroundColor = 'var(--bg-hover)'
                }}
                onMouseLeave={(e) => {
                  if (!isActive)
                    (e.currentTarget as HTMLDivElement).style.backgroundColor = 'transparent'
                }}
              >
                <item.icon
                  size={13}
                  style={{
                    marginRight: 10,
                    width: 18,
                    opacity: isActive ? 1 : 0.85,
                    color: isActive ? 'var(--accent-blue)' : 'var(--text-secondary)',
                  }}
                />
                <span
                  style={{
                    fontSize: 13,
                    color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                    fontWeight: isActive ? 500 : 400,
                  }}
                >
                  {item.label}
                </span>
              </div>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Bottom section */}
      <div className="px-4 pb-4">
        <hr
          style={{
            borderColor: 'var(--border)',
            marginBottom: 12,
            borderWidth: '0.5px',
          }}
        />
        <div className="flex items-center">
          <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>
            v2.0.0
          </span>
          <span className="font-mono" style={{ fontSize: 9, color: 'var(--accent-green)', fontWeight: 600 }}>
            {' '}| LIVE
          </span>
          <span
            className="animate-pulse-dot ml-1.5 inline-block rounded-full"
            style={{
              width: 6,
              height: 6,
              backgroundColor: 'var(--accent-green)',
            }}
          />
        </div>
      </div>
    </aside>
  )
}
