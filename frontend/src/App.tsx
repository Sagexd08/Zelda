import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { Toaster } from 'sonner'
import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useThemeStore } from './store/themeStore'
import Navbar from './components/layout/Navbar'
import LoadingScreen from './components/ui/LoadingScreen'
import Landing from './pages/Landing'
import Register from './pages/Register'
import Authenticate from './pages/Authenticate'
import Dashboard from './pages/Dashboard'
import Users from './pages/Users'
import Logs from './pages/Logs'
import Settings from './pages/Settings'

function App() {
  const [loading, setLoading] = useState(true)
  const { theme } = useThemeStore()

  useEffect(() => {
    const timer = setTimeout(() => setLoading(false), 1500)
    return () => clearTimeout(timer)
  }, [])

  return (
    <div className={theme}>
      <Router>
        <AnimatePresence mode="wait">
          {loading ? (
            <LoadingScreen key="loading" />
          ) : (
            <motion.div
              key="app"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="min-h-screen bg-dark-950 text-white"
            >
              <Navbar />
              <div className="pt-24">
                <Routes>
                  <Route path="/" element={<Landing />} />
                  <Route path="/register" element={<div className="container mx-auto px-4 py-8"><Register /></div>} />
                  <Route path="/authenticate" element={<Authenticate />} />
                  <Route path="/dashboard" element={<div className="container mx-auto px-4 py-8"><Dashboard /></div>} />
                  <Route path="/users" element={<div className="container mx-auto px-4 py-8"><Users /></div>} />
                  <Route path="/logs" element={<div className="container mx-auto px-4 py-8"><Logs /></div>} />
                  <Route path="/settings" element={<div className="container mx-auto px-4 py-8"><Settings /></div>} />
                </Routes>
              </div>
              <Toaster 
                position="top-right" 
                richColors 
                toastOptions={{
                  style: {
                    background: 'rgba(10, 15, 28, 0.95)',
                    border: '1px solid rgba(30, 144, 255, 0.2)',
                    color: '#fff',
                  },
                }}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </Router>
    </div>
  )
}

export default App

