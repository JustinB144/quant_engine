import React, { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('ErrorBoundary caught:', error, info)
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback ?? (
          <div
            className="flex flex-col items-center justify-center p-12"
            style={{ minHeight: '40vh' }}
          >
            <div
              className="rounded-lg p-6 text-center max-w-md"
              style={{
                backgroundColor: 'rgba(248, 81, 73, 0.08)',
                border: '1px solid rgba(248, 81, 73, 0.3)',
              }}
            >
              <h2 className="text-lg font-semibold mb-2" style={{ color: 'var(--accent-red)' }}>
                Something went wrong
              </h2>
              <p className="font-mono mb-4" style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>
                {this.state.error?.message ?? 'An unexpected error occurred'}
              </p>
              <a
                href="/"
                className="inline-block px-4 py-2 rounded-md text-sm"
                style={{
                  backgroundColor: 'var(--accent-blue)',
                  color: 'var(--text-primary)',
                  textDecoration: 'none',
                }}
              >
                Go to Dashboard
              </a>
            </div>
          </div>
        )
      )
    }

    return this.props.children
  }
}
