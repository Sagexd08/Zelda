// API Configuration
export const API_BASE_URL =
  (typeof window !== 'undefined' && (window as any).__ENV__?.VITE_API_URL) ||
  (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_API_URL) ||
  (typeof window !== 'undefined' && window.location.hostname.includes('vercel.app')
    ? 'https://zelda-facial-auth-api.onrender.com'
    : 'http://localhost:8000')

// Theme Colors
export const THEME_COLORS = {
  background: '#0A0F1C',
  accent: '#1E90FF',
  accentSky: '#00BFFF',
  neonBlue: '#00BFFF',
  neonLight: '#87CEEB',
  text: '#FFFFFF',
} as const

// User ID Validation
export const USER_ID_PATTERN = /^[A-Za-z0-9_\-]+$/
export const USER_ID_MAX_LENGTH = 64

// File Upload Limits
export const MAX_FILE_SIZE = 10 * 1024 * 1024 // 10MB
export const ALLOWED_IMAGE_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']

// Event Types for Logs
export const EVENT_TYPES = {
  LOGIN: 'login',
  REGISTER: 'register',
  DELETE: 'delete',
  AUTHENTICATE: 'authenticate',
  IDENTIFY: 'identify',
} as const

export type EventType = typeof EVENT_TYPES[keyof typeof EVENT_TYPES]

// Local Storage Keys
export const STORAGE_KEYS = {
  THEME: 'facial-auth-theme',
  API_URL: 'facial-auth-api-url',
} as const

