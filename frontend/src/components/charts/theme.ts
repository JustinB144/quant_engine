/** Bloomberg-dark ECharts theme matching dash_ui/theme.py */

export const CHART_COLORS = [
  '#58a6ff', '#3fb950', '#f85149', '#d29922', '#bc8cff',
  '#39d2c0', '#f0883e', '#79c0ff', '#56d364', '#ff7b72',
  '#e3b341', '#d2a8ff', '#76e4cc', '#ffa657',
]

export const REGIME_COLORS: Record<number, string> = {
  0: '#3fb950',
  1: '#f85149',
  2: '#58a6ff',
  3: '#d29922',
}

export const REGIME_NAMES: Record<number, string> = {
  0: 'Trending Bull',
  1: 'Trending Bear',
  2: 'Mean Reverting',
  3: 'High Volatility',
}

export const bloombergDarkTheme = {
  color: CHART_COLORS,
  backgroundColor: 'transparent',
  textStyle: {
    color: '#c9d1d9',
    fontFamily: 'Menlo, monospace',
    fontSize: 11,
  },
  title: {
    textStyle: {
      color: '#ffffff',
      fontSize: 13,
      fontWeight: 500,
    },
    subtextStyle: {
      color: '#8b949e',
      fontSize: 10,
    },
  },
  legend: {
    textStyle: { color: '#c9d1d9', fontSize: 10 },
    inactiveColor: '#30363d',
  },
  tooltip: {
    backgroundColor: '#1c2028',
    borderColor: '#30363d',
    borderWidth: 1,
    textStyle: { color: '#ffffff', fontSize: 11 },
  },
  xAxis: {
    axisLine: { lineStyle: { color: '#30363d' } },
    axisTick: { lineStyle: { color: '#30363d' } },
    axisLabel: { color: '#8b949e', fontSize: 10 },
    splitLine: { lineStyle: { color: '#21262d', width: 0.5 } },
    nameTextStyle: { color: '#8b949e', fontSize: 11 },
  },
  yAxis: {
    axisLine: { lineStyle: { color: '#30363d' } },
    axisTick: { lineStyle: { color: '#30363d' } },
    axisLabel: { color: '#8b949e', fontSize: 10 },
    splitLine: { lineStyle: { color: '#21262d', width: 0.5 } },
    nameTextStyle: { color: '#8b949e', fontSize: 11 },
  },
  grid: {
    borderColor: '#21262d',
  },
  categoryAxis: {
    axisLine: { lineStyle: { color: '#30363d' } },
    axisTick: { lineStyle: { color: '#30363d' } },
    axisLabel: { color: '#8b949e', fontSize: 10 },
    splitLine: { lineStyle: { color: '#21262d' } },
  },
  valueAxis: {
    axisLine: { lineStyle: { color: '#30363d' } },
    axisTick: { lineStyle: { color: '#30363d' } },
    axisLabel: { color: '#8b949e', fontSize: 10 },
    splitLine: { lineStyle: { color: '#21262d', width: 0.5 } },
  },
}
