import React from 'react'

interface SignificanceBadgeProps {
  /** p-value from statistical test */
  pValue: number | null | undefined
  /** Significance threshold (default 0.05) */
  alpha?: number
  /** Short label for the test */
  testName?: string
  /** Show the p-value number */
  showPValue?: boolean
}

/**
 * Badge showing statistical significance of a test result.
 * Green = significant (p < alpha), amber = marginal, red = not significant.
 */
export default function SignificanceBadge({
  pValue,
  alpha = 0.05,
  testName,
  showPValue = true,
}: SignificanceBadgeProps) {
  if (pValue == null) {
    return (
      <span
        className="font-mono inline-flex items-center gap-1 px-2 py-0.5 rounded"
        style={{
          fontSize: 10,
          backgroundColor: 'rgba(139,148,158,0.12)',
          color: 'var(--text-tertiary)',
        }}
      >
        {testName && <span>{testName}:</span>}
        <span>N/A</span>
      </span>
    )
  }

  const significant = pValue < alpha
  const marginal = pValue < alpha * 2 && !significant

  const bgColor = significant
    ? 'rgba(63,185,80,0.15)'
    : marginal
      ? 'rgba(210,153,34,0.15)'
      : 'rgba(248,81,73,0.12)'

  const textColor = significant
    ? 'var(--accent-green)'
    : marginal
      ? 'var(--accent-amber)'
      : 'var(--accent-red)'

  const label = significant ? 'SIG' : marginal ? 'MARG' : 'NS'

  return (
    <span
      className="font-mono inline-flex items-center gap-1 px-2 py-0.5 rounded"
      style={{ fontSize: 10, backgroundColor: bgColor, color: textColor, fontWeight: 600 }}
    >
      {testName && (
        <span style={{ fontWeight: 400, opacity: 0.8 }}>{testName}:</span>
      )}
      <span>{label}</span>
      {showPValue && (
        <span style={{ fontWeight: 400, opacity: 0.7 }}>
          p={pValue < 0.001 ? '<.001' : pValue.toFixed(3)}
        </span>
      )}
    </span>
  )
}

/**
 * Row of multiple significance badges for different tests.
 */
export function SignificanceBadgeRow({
  tests,
}: {
  tests: Array<{ name: string; pValue: number | null | undefined; alpha?: number }>
}) {
  return (
    <div className="flex items-center gap-2 flex-wrap">
      {tests.map((t) => (
        <SignificanceBadge key={t.name} testName={t.name} pValue={t.pValue} alpha={t.alpha} />
      ))}
    </div>
  )
}
