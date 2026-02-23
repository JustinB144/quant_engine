import React, { useState } from 'react'
import IVControls from './IVControls'
import ChartContainer from '@/components/charts/ChartContainer'
import EmptyState from '@/components/ui/EmptyState'
import AlertBanner from '@/components/ui/AlertBanner'

// Exact Heston presets from iv_surface.py:44-48
const HESTON_PRESETS: Record<string, Record<string, number>> = {
  'Normal Market': { v0: 0.04, theta: 0.06, kappa: 2.0, sigma: 0.40, rho: -0.70 },
  'Vol Storm': { v0: 0.15, theta: 0.08, kappa: 1.0, sigma: 0.80, rho: -0.80 },
  'Low Vol': { v0: 0.02, theta: 0.03, kappa: 3.0, sigma: 0.25, rho: -0.50 },
}

export default function HestonSurfaceTab() {
  const [preset, setPreset] = useState('Normal Market')
  const [v0, setV0] = useState(0.04)
  const [theta, setTheta] = useState(0.06)
  const [kappa, setKappa] = useState(2.0)
  const [sigma, setSigma] = useState(0.40)
  const [rho, setRho] = useState(-0.70)

  const handlePreset = (name: string) => {
    const p = HESTON_PRESETS[name]
    if (p) {
      setPreset(name)
      setV0(p.v0); setTheta(p.theta); setKappa(p.kappa); setSigma(p.sigma); setRho(p.rho)
    }
  }

  // Check Feller condition: 2 * kappa * theta > sigma^2
  const fellerOK = 2 * kappa * theta > sigma * sigma

  return (
    <div className="grid grid-cols-12 gap-3" style={{ paddingTop: 16 }}>
      <div className="col-span-3 lg:col-span-2">
        <IVControls
          title="Heston Parameters"
          presets={HESTON_PRESETS}
          activePreset={preset}
          onPresetChange={handlePreset}
          sliders={[
            { id: 'v0', label: 'v0 (init var)', min: 0.005, max: 0.30, step: 0.005, value: v0, onChange: setV0 },
            { id: 'theta', label: 'theta (long var)', min: 0.005, max: 0.20, step: 0.005, value: theta, onChange: setTheta },
            { id: 'kappa', label: 'kappa (reversion)', min: 0.10, max: 5.00, step: 0.10, value: kappa, onChange: setKappa },
            { id: 'sigma', label: 'sigma (vol-of-vol)', min: 0.05, max: 1.50, step: 0.05, value: sigma, onChange: setSigma },
            { id: 'rho', label: 'rho (correlation)', min: -0.95, max: 0.50, step: 0.05, value: rho, onChange: setRho },
          ]}
          actionButton={
            <button
              className="w-full py-2 rounded-md font-mono mt-2"
              style={{
                backgroundColor: 'var(--accent-blue)',
                color: 'var(--text-primary)',
                fontSize: 12,
                border: 'none',
                cursor: 'not-allowed',
                opacity: 0.5,
              }}
              disabled
            >
              Compute Surface
            </button>
          }
        />
      </div>
      <div className="col-span-9 lg:col-span-10">
        {!fellerOK && (
          <AlertBanner
            severity="warning"
            message="Feller condition violated"
            detail={`2 * kappa * theta (${(2 * kappa * theta).toFixed(3)}) < sigma^2 (${(sigma * sigma).toFixed(3)}). Variance process may reach zero.`}
          />
        )}
        <ChartContainer title="Heston Implied Volatility Surface">
          <EmptyState message="Heston computation requires server-side HestonModel (Python). A backend endpoint for Heston IV is not yet available." />
        </ChartContainer>
        <ChartContainer title="Heston Smile Curves">
          <EmptyState message="Heston smiles will appear after server-side computation" />
        </ChartContainer>
      </div>
    </div>
  )
}
