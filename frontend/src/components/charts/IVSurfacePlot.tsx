import React from 'react'
import Plot from 'react-plotly.js'

interface IVSurfacePlotProps {
  moneyness: number[][]
  expiries: number[][]
  ivGrid: number[][]
  title?: string
  height?: number
}

export default function IVSurfacePlot({
  moneyness,
  expiries,
  ivGrid,
  title = 'IV Surface',
  height = 520,
}: IVSurfacePlotProps) {
  return (
    <Plot
      data={[
        {
          type: 'surface',
          x: moneyness,
          y: expiries,
          z: ivGrid,
          colorscale: 'Inferno',
          colorbar: {
            title: { text: 'IV (%)', font: { color: '#c9d1d9', size: 11 } },
            tickfont: { color: '#c9d1d9', size: 10 },
            len: 0.6,
          },
          hovertemplate:
            'Moneyness: %{x:.2f}<br>Expiry: %{y:.2f}Y<br>IV: %{z:.1f}%<extra></extra>',
        },
      ]}
      layout={{
        title: { text: title, font: { color: '#ffffff', size: 14 } },
        height,
        paper_bgcolor: '#0d1117',
        plot_bgcolor: '#161b22',
        font: { family: 'Menlo, monospace', color: '#ffffff', size: 12 },
        margin: { l: 0, r: 0, t: 40, b: 0 },
        scene: {
          xaxis: {
            title: { text: 'Moneyness (K/S)' },
            backgroundcolor: '#161b22',
            gridcolor: '#21262d',
            color: '#c9d1d9',
          },
          yaxis: {
            title: { text: 'Expiry (Years)' },
            backgroundcolor: '#161b22',
            gridcolor: '#21262d',
            color: '#c9d1d9',
          },
          zaxis: {
            title: { text: 'IV (%)' },
            backgroundcolor: '#161b22',
            gridcolor: '#21262d',
            color: '#c9d1d9',
          },
          bgcolor: '#0d1117',
          camera: {
            eye: { x: 1.6, y: -1.6, z: 0.8 },
            up: { x: 0, y: 0, z: 1 },
            center: { x: 0, y: 0, z: -0.1 },
          },
        },
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%' }}
    />
  )
}
