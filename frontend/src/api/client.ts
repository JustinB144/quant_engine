import type { ApiResponse, ResponseMeta } from '@/types/api'

const BASE_URL = '/api'

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public meta?: ResponseMeta,
    public data?: unknown,
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
  const headers: Record<string, string> = {}
  if (options.body) {
    headers['Content-Type'] = 'application/json'
  }
  const res = await fetch(url, {
    ...options,
    headers: {
      ...headers,
      ...options.headers,
    },
  })

  if (!res.ok) {
    try {
      const errorJson = await res.json()
      throw new ApiError(
        errorJson.error || errorJson.message || res.statusText,
        res.status,
        errorJson.meta,
        errorJson,
      )
    } catch (e) {
      if (e instanceof ApiError) throw e
      const text = await res.text().catch(() => 'Unknown error')
      throw new ApiError(text, res.status)
    }
  }

  const json: ApiResponse<T> = await res.json()

  if (!json.ok) {
    throw new ApiError(json.error || 'Request failed', res.status, json.meta, json)
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
