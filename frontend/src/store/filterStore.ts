import { create } from 'zustand'

interface FilterState {
  horizon: number
  setHorizon: (h: number) => void
  selectedTicker: string
  setSelectedTicker: (t: string) => void
  dateRange: [string, string] | null
  setDateRange: (r: [string, string] | null) => void
  selectedTimeframe: string
  setSelectedTimeframe: (tf: string) => void
  enabledIndicators: string[]
  setEnabledIndicators: (indicators: string[]) => void
  toggleIndicator: (indicator: string) => void
}

export const useFilterStore = create<FilterState>((set) => ({
  horizon: 10,
  setHorizon: (horizon) => set({ horizon }),
  selectedTicker: '',
  setSelectedTicker: (selectedTicker) => set({ selectedTicker }),
  dateRange: null,
  setDateRange: (dateRange) => set({ dateRange }),
  selectedTimeframe: '1d',
  setSelectedTimeframe: (selectedTimeframe) => set({ selectedTimeframe }),
  enabledIndicators: [],
  setEnabledIndicators: (enabledIndicators) => set({ enabledIndicators }),
  toggleIndicator: (indicator) =>
    set((state) => ({
      enabledIndicators: state.enabledIndicators.includes(indicator)
        ? state.enabledIndicators.filter((i) => i !== indicator)
        : [...state.enabledIndicators, indicator],
    })),
}))
