import React, { useState } from 'react'
import type { ResponseMeta } from '@/types/api'

const MODE_CONFIG: Record<string, { color: string; label: string }> = {
  live: { color: 'var(--provenance-live)', label: 'LIVE' },
  fallback: { color: 'var(--provenance-fallback)', label: 'FALLBACK' },
  demo: { color: 'var(--provenance-demo)', label: 'DEMO' },
}

export default function DataProvenanceBadge({ meta }: { meta?: ResponseMeta }) {
  const [showTooltip, setShowTooltip] = useState(false)

  if (!meta) return null

  const mode = meta.data_mode?.toLowerCase() ?? 'live'
  const config = MODE_CONFIG[mode] ?? MODE_CONFIG.live

  return (
    <div
      className="relative inline-flex items-center gap-1.5 cursor-default"
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      <span
        className="inline-block rounded-full animate-pulse-dot"
        style={{ width: 6, height: 6, backgroundColor: config.color }}
      />
      <span className="font-mono" style={{ fontSize: 9, color: config.color, fontWeight: 600 }}>
        {config.label}
      </span>

      {showTooltip && (
        <div
          className="absolute right-0 top-full mt-1 z-50 rounded-md p-3 min-w-[220px]"
          style={{
            backgroundColor: 'var(--bg-tertiary)',
            border: '1px solid var(--border)',
            fontSize: 10,
            color: 'var(--text-secondary)',
          }}
        >
          <div className="font-mono space-y-1">
            <div>Mode: <span style={{ color: config.color }}>{config.label}</span></div>
            {meta.generated_at && <div>Generated: {new Date(meta.generated_at).toLocaleTimeString()}</div>}
            {meta.cache_hit && <div style={{ color: 'var(--accent-amber)' }}>Cache hit</div>}
            {meta.elapsed_ms != null && <div>Elapsed: {meta.elapsed_ms.toFixed(0)}ms</div>}
            {meta.model_version && <div>Model: {meta.model_version}</div>}
            {meta.source_summary && <div>Source: {meta.source_summary}</div>}
            {(meta?.warnings?.length ?? 0) > 0 && (
              <div style={{ color: 'var(--accent-amber)', marginTop: 4 }}>
                {meta.warnings.map((w, i) => (
                  <div key={i}>Warning: {w}</div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
