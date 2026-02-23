import React, { Suspense, lazy } from 'react'
import { Routes, Route, useLocation } from 'react-router-dom'
import AppShell from '@/components/layout/AppShell'
import Spinner from '@/components/ui/Spinner'
import ErrorBoundary from '@/components/ui/ErrorBoundary'

const DashboardPage = lazy(() => import('@/pages/DashboardPage'))
const SystemHealthPage = lazy(() => import('@/pages/SystemHealthPage'))
const LogsPage = lazy(() => import('@/pages/LogsPage'))
const DataExplorerPage = lazy(() => import('@/pages/DataExplorerPage'))
const ModelLabPage = lazy(() => import('@/pages/ModelLabPage'))
const SignalDeskPage = lazy(() => import('@/pages/SignalDeskPage'))
const BacktestPage = lazy(() => import('@/pages/BacktestPage'))
const IVSurfacePage = lazy(() => import('@/pages/IVSurfacePage'))
const BenchmarkPage = lazy(() => import('@/pages/BenchmarkPage'))
const AutopilotPage = lazy(() => import('@/pages/AutopilotPage'))
const NotFoundPage = lazy(() => import('@/pages/NotFoundPage'))

function PageWrapper({ children }: { children: React.ReactNode }) {
  const location = useLocation()
  return <ErrorBoundary key={location.pathname}>{children}</ErrorBoundary>
}

export default function App() {
  return (
    <AppShell>
      <Suspense fallback={<div className="flex items-center justify-center h-full"><Spinner /></div>}>
        <Routes>
          <Route path="/" element={<PageWrapper><DashboardPage /></PageWrapper>} />
          <Route path="/system-health" element={<PageWrapper><SystemHealthPage /></PageWrapper>} />
          <Route path="/system-logs" element={<PageWrapper><LogsPage /></PageWrapper>} />
          <Route path="/data-explorer" element={<PageWrapper><DataExplorerPage /></PageWrapper>} />
          <Route path="/model-lab" element={<PageWrapper><ModelLabPage /></PageWrapper>} />
          <Route path="/signal-desk" element={<PageWrapper><SignalDeskPage /></PageWrapper>} />
          <Route path="/backtest-risk" element={<PageWrapper><BacktestPage /></PageWrapper>} />
          <Route path="/iv-surface" element={<PageWrapper><IVSurfacePage /></PageWrapper>} />
          <Route path="/sp-comparison" element={<PageWrapper><BenchmarkPage /></PageWrapper>} />
          <Route path="/autopilot" element={<PageWrapper><AutopilotPage /></PageWrapper>} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </Suspense>
    </AppShell>
  )
}
