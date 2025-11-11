import axios, { AxiosError } from 'axios'
import { API_BASE_URL } from '../utils/constants'
import { supabase } from '../lib/supabase'

export interface RegisterResponse {
  success: boolean
  user_id?: string
  samples_processed?: number
  valid_samples?: number
  avg_quality_score?: number
  avg_liveness_score?: number
  error?: string
}

export interface AuthenticateResponse {
  authenticated: boolean
  confidence: number
  threshold?: number
  liveness_score?: number
  reason: string
  similarities?: Record<string, number>
}

export interface IdentifyResponse {
  found: boolean
  liveness_score?: number
  matches: Array<{
    user_id: string
    confidence: number
  }>
  total_users_checked?: number
  reason?: string
}

export interface SystemInfo {
  version: string
  environment: string
  features: {
    liveness_detection: boolean
    depth_estimation: boolean
    temporal_liveness: boolean
    voice_authentication: boolean
    challenge_response: boolean
    adaptive_learning: boolean
    adaptive_threshold: boolean
    bias_monitoring: boolean
  }
  models: {
    face_size: number
    verification_threshold: number
    liveness_threshold: number
    min_registration_samples: number
  }
}

export interface HealthResponse {
  status: string
  timestamp: number
}

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true,
})

// Request interceptor for JWT token (if needed in future)
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      // Handle unauthorized - clear token and redirect
      localStorage.removeItem('auth_token')
    }
    return Promise.reject(error)
  }
)

// System Info
export const getSystemInfo = async (): Promise<SystemInfo> => {
  const response = await apiClient.get<SystemInfo>('/api/v1/system/info')
  return response.data
}

export const getHealth = async (): Promise<HealthResponse> => {
  const response = await apiClient.get<HealthResponse>('/health')
  return response.data
}

// Registration
export const registerUser = async (
  userId: string,
  images: File[]
): Promise<RegisterResponse> => {
  const formData = new FormData()
  formData.append('user_id', userId)
  images.forEach((image) => {
    formData.append('images', image)
  })

  const response = await apiClient.post<RegisterResponse>(
    '/api/v1/register',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  )
  return response.data
}

// Authentication
export const authenticateUser = async (
  userId: string,
  image: File
): Promise<AuthenticateResponse> => {
  const formData = new FormData()
  formData.append('user_id', userId)
  formData.append('image', image)

  const response = await apiClient.post<AuthenticateResponse>(
    '/api/v1/authenticate',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  )
  return response.data
}

// Identification
export const identifyUser = async (
  image: File,
  topK: number = 3
): Promise<IdentifyResponse> => {
  const formData = new FormData()
  formData.append('image', image)

  const response = await apiClient.post<IdentifyResponse>(
    `/api/v1/identify?top_k=${topK}`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  )
  return response.data
}

// User Management
export const deleteUser = async (userId: string): Promise<{ success: boolean; message: string }> => {
  const formData = new FormData()
  formData.append('user_id', userId)

  const response = await apiClient.post<{ success: boolean; message: string }>(
    '/api/v1/delete_user',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  )
  return response.data
}

// WebSocket connection
export const createWebSocketConnection = (clientId: string): WebSocket => {
  const wsUrl = API_BASE_URL.replace('http', 'ws')
  return new WebSocket(`${wsUrl}/ws/${clientId}`)
}

export default apiClient

