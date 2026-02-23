import { useQuery } from '@tanstack/react-query'
import { get } from '@/api/client'
import { BENCHMARK_COMPARISON, BENCHMARK_EQUITY_CURVES, BENCHMARK_ROLLING_METRICS } from '@/api/endpoints'
import type { TimeSeriesPoint } from '@/types/dashboard'

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

/** Matches /api/benchmark/equity-curves response.data */
export interface BenchmarkEquityCurves {
  strategy: TimeSeriesPoint[]
  benchmark: TimeSeriesPoint[]
  points: number
}

/** Matches /api/benchmark/rolling-metrics response.data */
export interface BenchmarkRollingMetrics {
  rolling_correlation: TimeSeriesPoint[]
  rolling_alpha: TimeSeriesPoint[]
  rolling_beta: TimeSeriesPoint[]
  relative_strength: TimeSeriesPoint[]
  drawdown_strategy: TimeSeriesPoint[]
  drawdown_benchmark: TimeSeriesPoint[]
  points: number
  window: number
}

export function useBenchmarkEquityCurves() {
  return useQuery({
    queryKey: ['benchmark', 'equity-curves'],
    queryFn: () => get<BenchmarkEquityCurves>(BENCHMARK_EQUITY_CURVES),
    staleTime: 120_000,
  })
}

export function useBenchmarkRollingMetrics() {
  return useQuery({
    queryKey: ['benchmark', 'rolling-metrics'],
    queryFn: () => get<BenchmarkRollingMetrics>(BENCHMARK_ROLLING_METRICS),
    staleTime: 120_000,
  })
}
