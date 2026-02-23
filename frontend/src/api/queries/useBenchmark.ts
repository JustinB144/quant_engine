import { useQuery } from '@tanstack/react-query'
import { get } from '@/api/client'
import { BENCHMARK_COMPARISON } from '@/api/endpoints'

/** Matches actual /api/benchmark/comparison response.data */
export interface BenchmarkData {
  strategy: {
    annual_return: number
    annual_vol: number
    sharpe: number
    sortino: number
    max_drawdown: number
    var95: number
    cvar95: number
    var99: number
    cvar99: number
  }
  benchmark: {
    annual_return: number
    annual_vol: number
    sharpe: number
    sortino: number
    max_drawdown: number
    var95: number
    cvar95: number
    var99: number
    cvar99: number
  }
  strategy_points: number
  benchmark_points: number
}

export function useBenchmarkComparison() {
  return useQuery({
    queryKey: ['benchmark', 'comparison'],
    queryFn: () => get<BenchmarkData>(BENCHMARK_COMPARISON),
    staleTime: 120_000,
  })
}
