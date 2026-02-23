export function formatPercent(value: number | null | undefined, decimals = 2): string {
  if (value == null || isNaN(value)) return 'N/A'
  return `${(value * 100).toFixed(decimals)}%`
}

export function formatNumber(value: number | null | undefined, decimals = 2): string {
  if (value == null || isNaN(value)) return 'N/A'
  return value.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

export function colorForReturn(value: number): string {
  if (value > 0) return 'var(--accent-green)'
  if (value < 0) return 'var(--accent-red)'
  return 'var(--text-secondary)'
}

export function formatSignedPercent(value: number | null | undefined, decimals = 2): string {
  if (value == null || isNaN(value)) return 'N/A'
  const sign = value >= 0 ? '+' : ''
  return `${sign}${(value * 100).toFixed(decimals)}%`
}
