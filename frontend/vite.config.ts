import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'plotly': ['plotly.js', 'react-plotly.js'],
          'echarts': ['echarts', 'echarts-for-react'],
          'lightweight-charts': ['lightweight-charts'],
          'tanstack': ['@tanstack/react-query', '@tanstack/react-table', '@tanstack/react-virtual'],
        },
      },
    },
  },
})
