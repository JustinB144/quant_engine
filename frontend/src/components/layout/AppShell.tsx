import React from 'react'
import Sidebar from './Sidebar'
import StatusBar from './StatusBar'
import { useUIStore } from '@/store/uiStore'

export default function AppShell({ children }: { children: React.ReactNode }) {
  const collapsed = useUIStore((s) => s.sidebarCollapsed)

  return (
    <div className="flex h-screen overflow-hidden bg-bg-primary">
      <Sidebar />
      <div
        className="flex flex-col flex-1 min-w-0 transition-all duration-200"
        style={{ marginLeft: collapsed ? 0 : 240 }}
      >
        <main className="flex-1 overflow-auto p-6">{children}</main>
        <StatusBar />
      </div>
    </div>
  )
}
