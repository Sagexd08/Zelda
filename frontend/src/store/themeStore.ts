import { create } from 'zustand'

type Theme = 'dark' | 'light'

interface ThemeState {
  theme: Theme
  setTheme: (theme: Theme) => void
  toggleTheme: () => void
}

export const useThemeStore = create<ThemeState>((set) => {
  // Initialize theme from localStorage
  const storedTheme = localStorage.getItem('facial-auth-theme') as Theme | null
  const initialTheme = storedTheme || 'dark'

  // Apply theme to document
  if (typeof document !== 'undefined') {
    document.documentElement.classList.remove('light', 'dark')
    document.documentElement.classList.add(initialTheme)
  }

  return {
    theme: initialTheme,
    setTheme: (theme: Theme) => {
      set({ theme })
      if (typeof document !== 'undefined') {
        document.documentElement.classList.remove('light', 'dark')
        document.documentElement.classList.add(theme)
        localStorage.setItem('facial-auth-theme', theme)
      }
    },
    toggleTheme: () => {
      set((state) => {
        const newTheme = state.theme === 'dark' ? 'light' : 'dark'
        if (typeof document !== 'undefined') {
          document.documentElement.classList.remove('light', 'dark')
          document.documentElement.classList.add(newTheme)
          localStorage.setItem('facial-auth-theme', newTheme)
        }
        return { theme: newTheme }
      })
    },
  }
})

