import type { ApiResponse, ResponseMeta } from '@/types/api'

const BASE_URL = '/api'

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public meta?: ResponseMeta,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<{ data: T; meta: ResponseMeta }> {
  const url = `${BASE_URL}${path}`
  const res = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  })

  if (!res.ok) {
    const text = await res.text().catch(() => 'Unknown error')
    throw new ApiError(text, res.status)
  }

  const json: ApiResponse<T> = await res.json()

  if (!json.ok) {
    throw new ApiError(json.error || 'Request failed', res.status, json.meta)
  }

  return { data: json.data as T, meta: json.meta }
}

export function get<T>(path: string) {
  return request<T>(path, { method: 'GET' })
}

export function post<T>(path: string, body?: unknown) {
  return request<T>(path, {
    method: 'POST',
    body: body ? JSON.stringify(body) : undefined,
  })
}

export function patch<T>(path: string, body?: unknown) {
  return request<T>(path, {
    method: 'PATCH',
    body: body ? JSON.stringify(body) : undefined,
  })
}
