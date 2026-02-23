import React from 'react'
import { Link } from 'react-router-dom'
import PageContainer from '@/components/layout/PageContainer'

export default function NotFoundPage() {
  return (
    <PageContainer>
      <div className="flex flex-col items-center justify-center" style={{ minHeight: '60vh' }}>
        <h1 className="text-4xl font-bold mb-2" style={{ color: 'var(--text-tertiary)' }}>404</h1>
        <p className="mb-4" style={{ color: 'var(--text-tertiary)' }}>Page not found</p>
        <Link
          to="/"
          className="px-4 py-2 rounded-md text-sm"
          style={{
            backgroundColor: 'var(--accent-blue)',
            color: 'var(--text-primary)',
            textDecoration: 'none',
          }}
        >
          Go to Dashboard
        </Link>
      </div>
    </PageContainer>
  )
}
