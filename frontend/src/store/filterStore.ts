import { create } from 'zustand'

interface FilterState {
  horizon: number
  setHorizon: (h: number) => void
  selectedTicker: string
  setSelectedTicker: (t: string) => void
  dateRange: [string, string] | null
  setDateRange: (r: [string, string] | null) => void
}

export const useFilterStore = create<FilterState>((set) => ({
  horizon: 10,
  setHorizon: (horizon) => set({ horizon }),
  selectedTicker: '',
  setSelectedTicker: (selectedTicker) => set({ selectedTicker }),
  dateRange: null,
  setDateRange: (dateRange) => set({ dateRange }),
}))
