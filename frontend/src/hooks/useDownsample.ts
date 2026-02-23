import { useMemo } from 'react'

/**
 * Largest Triangle Three Buckets (LTTB) downsampling algorithm.
 * Preserves visual shape while limiting point count.
 */
function lttb(data: { x: number; y: number }[], threshold: number): { x: number; y: number }[] {
  if (data.length <= threshold) return data

  const sampled: { x: number; y: number }[] = []
  const bucketSize = (data.length - 2) / (threshold - 2)

  sampled.push(data[0])

  let a = 0
  for (let i = 0; i < threshold - 2; i++) {
    const avgRangeStart = Math.floor((i + 1) * bucketSize) + 1
    const avgRangeEnd = Math.min(Math.floor((i + 2) * bucketSize) + 1, data.length)

    let avgX = 0
    let avgY = 0
    const avgLen = avgRangeEnd - avgRangeStart
    for (let j = avgRangeStart; j < avgRangeEnd; j++) {
      avgX += data[j].x
      avgY += data[j].y
    }
    avgX /= avgLen
    avgY /= avgLen

    const rangeStart = Math.floor(i * bucketSize) + 1
    const rangeEnd = Math.min(Math.floor((i + 1) * bucketSize) + 1, data.length)

    let maxArea = -1
    let maxIdx = rangeStart

    for (let j = rangeStart; j < rangeEnd; j++) {
      const area = Math.abs(
        (data[a].x - avgX) * (data[j].y - data[a].y) -
        (data[a].x - data[j].x) * (avgY - data[a].y),
      )
      if (area > maxArea) {
        maxArea = area
        maxIdx = j
      }
    }

    sampled.push(data[maxIdx])
    a = maxIdx
  }

  sampled.push(data[data.length - 1])
  return sampled
}

export function useDownsample<T extends { x: number; y: number }>(
  data: T[],
  maxPoints = 2500,
): T[] {
  return useMemo(() => {
    if (!data || data.length <= maxPoints) return data
    return lttb(data, maxPoints) as T[]
  }, [data, maxPoints])
}
