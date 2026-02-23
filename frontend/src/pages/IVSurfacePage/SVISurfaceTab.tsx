import React, { useState, useMemo } from 'react'
import IVControls from './IVControls'
import IVSurfacePlot from '@/components/charts/IVSurfacePlot'
import LineChart from '@/components/charts/LineChart'
import ChartContainer from '@/components/charts/ChartContainer'

// Exact SVI presets from iv_surface.py:36-42
const SVI_PRESETS: Record<string, Record<string, number>> = {
  'Equity Normal': { a: 0.04, b: 0.35, rho: -0.40, m: 0.00, sigma: 0.25 },
  'Equity Stressed': { a: 0.08, b: 0.55, rho: -0.65, m: -0.05, sigma: 0.15 },
  'Commodity Smile': { a: 0.03, b: 0.30, rho: 0.10, m: 0.00, sigma: 0.35 },
  'Flat Vol': { a: 0.06, b: 0.08, rho: -0.10, m: 0.00, sigma: 0.40 },
  'Steep Skew': { a: 0.03, b: 0.50, rho: -0.80, m: -0.08, sigma: 0.12 },
}

// Grid parameters matching iv_surface.py
const S = 100.0
const R = 0.05
const N_K = 35
const N_T = 25

function linspace(start: number, end: number, n: number): number[] {
  const step = (end - start) / (n - 1)
  return Array.from({ length: n }, (_, i) => start + i * step)
}

const MONEYNESS = linspace(0.65, 1.35, N_K)
const STRIKES = MONEYNESS.map((m) => S * m)
const EXPIRIES = linspace(0.08, 2.0, N_T)

const SMILE_EXPIRIES: Record<string, number> = { '1M': 1 / 12, '3M': 0.25, '6M': 0.5, '1Y': 1.0, '2Y': 2.0 }
const SMILE_COLORS = ['#58a6ff', '#3fb950', '#d29922', '#bc8cff', '#f85149']

function computeSVISurface(a: number, b: number, rho: number, m: number, sigma: number) {
  const ivGrid: number[][] = []
  const kGrid: number[][] = []
  const tGrid: number[][] = []

  for (let i = 0; i < EXPIRIES.length; i++) {
    const T = EXPIRIES[i]
    const F = S * Math.exp(R * T)
    const ivRow: number[] = []
    const kRow: number[] = []
    const tRow: number[] = []

    for (let j = 0; j < STRIKES.length; j++) {
      const k = Math.log(STRIKES[j] / F)
      const w = a + b * (rho * (k - m) + Math.sqrt((k - m) ** 2 + sigma ** 2))
      const wClamped = Math.max(w, 1e-10)
      const iv = Math.sqrt(wClamped / T) * 100 // as percentage
      ivRow.push(iv)
      kRow.push(MONEYNESS[j])
      tRow.push(T)
    }
    ivGrid.push(ivRow)
    kGrid.push(kRow)
    tGrid.push(tRow)
  }
  return { kGrid, tGrid, ivGrid }
}

function computeSVISmiles(a: number, b: number, rho: number, m: number, sigma: number) {
  const smiles: Record<string, number[]> = {}
  for (const [label, T] of Object.entries(SMILE_EXPIRIES)) {
    const F = S * Math.exp(R * T)
    smiles[label] = STRIKES.map((K) => {
      const k = Math.log(K / F)
      const w = a + b * (rho * (k - m) + Math.sqrt((k - m) ** 2 + sigma ** 2))
      const wClamped = Math.max(w, 1e-10)
      return Math.sqrt(wClamped / T) * 100
    })
  }
  return smiles
}

export default function SVISurfaceTab() {
  const [preset, setPreset] = useState('Equity Normal')
  const [a, setA] = useState(0.04)
  const [b, setB] = useState(0.35)
  const [rho, setRho] = useState(-0.40)
  const [m, setM] = useState(0.00)
  const [sigma, setSigma] = useState(0.25)

  const handlePreset = (name: string) => {
    const p = SVI_PRESETS[name]
    if (p) {
      setPreset(name)
      setA(p.a); setB(p.b); setRho(p.rho); setM(p.m); setSigma(p.sigma)
    }
  }

  const surface = useMemo(() => computeSVISurface(a, b, rho, m, sigma), [a, b, rho, m, sigma])
  const smiles = useMemo(() => computeSVISmiles(a, b, rho, m, sigma), [a, b, rho, m, sigma])

  const smileSeries = Object.entries(smiles).map(([label, data], i) => ({
    name: label,
    data,
    color: SMILE_COLORS[i],
  }))

  return (
    <div className="grid grid-cols-12 gap-3" style={{ paddingTop: 16 }}>
      <div className="col-span-3 lg:col-span-2">
        <IVControls
          title="SVI Parameters"
          presets={SVI_PRESETS}
          activePreset={preset}
          onPresetChange={handlePreset}
          sliders={[
            { id: 'a', label: 'a (level)', min: -0.10, max: 0.20, step: 0.005, value: a, onChange: setA },
            { id: 'b', label: 'b (angle)', min: 0.01, max: 1.00, step: 0.01, value: b, onChange: setB },
            { id: 'rho', label: 'rho (skew)', min: -0.95, max: 0.95, step: 0.05, value: rho, onChange: setRho },
            { id: 'm', label: 'm (shift)', min: -0.30, max: 0.30, step: 0.01, value: m, onChange: setM },
            { id: 'sigma', label: 'sigma (curvature)', min: 0.01, max: 0.80, step: 0.01, value: sigma, onChange: setSigma },
          ]}
        />
      </div>
      <div className="col-span-9 lg:col-span-10">
        <ChartContainer title="SVI Implied Volatility Surface">
          <IVSurfacePlot
            moneyness={surface.kGrid}
            expiries={surface.tGrid}
            ivGrid={surface.ivGrid}
            title="SVI Implied Volatility Surface"
          />
        </ChartContainer>
        <ChartContainer title="SVI Smile Curves by Expiry">
          <LineChart
            categories={MONEYNESS.map((m) => m.toFixed(2))}
            series={smileSeries}
            height={320}
            yAxisName="IV (%)"
          />
        </ChartContainer>
      </div>
    </div>
  )
}
