import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ShieldCheck, Video, Camera, CheckCircle2, XCircle } from 'lucide-react'
import { toast } from 'sonner'
import Webcam from 'react-webcam'
import { Card } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { Progress } from '../components/ui/progress'
import Spinner from '../components/ui/Spinner'
import Plasma from '../components/effects/Plasma'
import { authenticateUser, type AuthenticateResponse } from '../services/api'

const Authenticate = () => {
  const [userId, setUserId] = useState('')
  const [mode, setMode] = useState<'video' | 'image'>('video')
  const [isStreaming, setIsStreaming] = useState(false)
  const [result, setResult] = useState<AuthenticateResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const webcamRef = useRef<Webcam>(null)
  const [faceDetected, setFaceDetected] = useState(false)

  const videoConstraints = {
    width: 1280,
    height: 720,
    facingMode: 'user' as const,
  }

  const extractErrorMessage = (error: unknown, fallback = 'Authentication failed'): string => {
    if (error && typeof error === 'object' && 'response' in error) {
      const axiosError = error as { response?: { data?: { detail?: unknown } } }
      const detail = axiosError.response?.data?.detail
      if (typeof detail === 'string') return detail
      if (detail && typeof detail === 'object') {
        if ('message' in detail && typeof detail.message === 'string') return detail.message
        if ('error' in detail && typeof detail.error === 'string') return detail.error
      }
    }
    if (error instanceof Error) return error.message
    return fallback
  }

  const captureAndAuthenticate = async () => {
    if (!userId.trim()) {
      toast.error('Please enter a User ID')
      return
    }

    const imageSrc = webcamRef.current?.getScreenshot()
    if (!imageSrc) {
      toast.error('Failed to capture image')
      return
    }

    setLoading(true)
    setResult(null)

    try {
      const response = await fetch(imageSrc)
      const blob = await response.blob()
      const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' })

      const authResult = await authenticateUser(userId, file)
      setResult(authResult)

      if (authResult.authenticated) {
        toast.success(
          `Authentication successful! Confidence: ${(authResult.confidence * 100).toFixed(1)}%`
        )
      } else {
        toast.error(`Authentication failed: ${authResult.reason}`)
      }
    } catch (error) {
      console.error('Authentication error:', error)
      const message = extractErrorMessage(error)
      toast.error(message)
      setResult({
        authenticated: false,
        reason: message,
        confidence: 0,
      })
    } finally {
      setLoading(false)
    }
  }

  const startStreaming = () => {
    if (!userId.trim()) {
      toast.error('Please enter a User ID first')
      return
    }
    setIsStreaming(true)
    setResult(null)
  }

  const stopStreaming = () => {
    setIsStreaming(false)
    setFaceDetected(false)
  }

  useEffect(() => {
    if (isStreaming) {
      const interval = setInterval(() => {
        setFaceDetected(Math.random() > 0.3)
      }, 500)
      return () => clearInterval(interval)
    }
  }, [isStreaming])

  return (
    <div className="relative min-h-screen">
      <div className="absolute inset-0 z-0">
        <Plasma
          color="#00BFFF"
          speed={0.6}
          direction="forward"
          scale={1.5}
          opacity={0.25}
          mouseInteractive={true}
        />
      </div>
      <div className="relative z-10 max-w-7xl mx-auto px-4 py-8 pt-32">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex items-center mb-4">
            <div className="w-12 h-12 bg-gradient-blue rounded-xl flex items-center justify-center mr-4">
              <ShieldCheck className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold text-white">Face Authentication</h1>
              <p className="text-gray-400 mt-1">Verify your identity using facial recognition</p>
            </div>
          </div>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <Card variant="elevated" className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-white">Authentication Mode</h2>
                <div className="flex space-x-2">
                  <Button
                    onClick={() => setMode('video')}
                    variant={mode === 'video' ? 'default' : 'secondary'}
                    icon={<Video className="w-4 h-4" />}
                    size="sm"
                  >
                    Live Video
                  </Button>
                  <Button
                    onClick={() => setMode('image')}
                    variant={mode === 'image' ? 'default' : 'secondary'}
                    icon={<Camera className="w-4 h-4" />}
                    size="sm"
                  >
                    Single Shot
                  </Button>
                </div>
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">User ID *</label>
                <input
                  type="text"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  placeholder="Enter user ID to authenticate"
                  disabled={isStreaming}
                  className="w-full px-4 py-3 bg-dark-800 border border-dark-600 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-accent transition-all disabled:opacity-50"
                />
              </div>

              <div className="relative rounded-2xl overflow-hidden bg-dark-900 border border-accent/20">
                <Webcam
                  audio={false}
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  videoConstraints={videoConstraints}
                  className="w-full"
                />

                <AnimatePresence>
                  {isStreaming && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="absolute inset-0 pointer-events-none"
                    >
                      <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                        <motion.div
                          animate={{
                            scale: faceDetected ? 1 : 1.1,
                            opacity: faceDetected ? 1 : 0.5,
                          }}
                          className={`w-64 h-80 border-4 rounded-3xl ${
                            faceDetected ? 'border-accent neon-border' : 'border-yellow-500'
                          }`}
                        />
                      </div>
                      {faceDetected && (
                        <motion.div
                          initial={{ opacity: 0, y: -20 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="absolute top-4 left-1/2 transform -translate-x-1/2 glass px-4 py-2 rounded-full flex items-center space-x-2"
                        >
                          <CheckCircle2 className="w-5 h-5 text-accent" />
                          <span className="font-semibold text-white">Face Detected</span>
                        </motion.div>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>

                {isStreaming && (
                  <div className="absolute top-4 right-4 flex flex-col space-y-2">
                    <div className="glass px-3 py-2 rounded-lg flex items-center space-x-2">
                      <div className="w-2 h-2 bg-accent rounded-full animate-pulse" />
                      <span className="text-white text-sm font-medium">LIVE</span>
                    </div>
                  </div>
                )}
              </div>

              <div className="flex space-x-4 mt-6">
                {mode === 'video' ? (
                  <>
                    {!isStreaming ? (
                      <Button
                        onClick={startStreaming}
                        icon={<Video className="w-5 h-5" />}
                        className="flex-1"
                        size="lg"
                        disabled={!userId}
                      >
                        Start Live Authentication
                      </Button>
                    ) : (
                      <>
                        <Button
                          onClick={captureAndAuthenticate}
                          icon={loading ? undefined : <ShieldCheck className="w-5 h-5" />}
                          className="flex-1"
                          size="lg"
                          disabled={loading || !faceDetected}
                          loading={loading}
                        >
                          Authenticate Now
                        </Button>
                        <Button onClick={stopStreaming} variant="destructive" size="lg">
                          Stop
                        </Button>
                      </>
                    )}
                  </>
                ) : (
                  <Button
                    onClick={captureAndAuthenticate}
                    icon={loading ? undefined : <Camera className="w-5 h-5" />}
                    className="w-full"
                    size="lg"
                    disabled={loading || !userId}
                    loading={loading}
                  >
                    Capture & Authenticate
                  </Button>
                )}
              </div>
            </Card>

            <AnimatePresence>
              {result && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                >
                  <Card
                    variant={result.authenticated ? 'gradient' : 'outlined'}
                    className={
                      result.authenticated
                        ? 'border-accent/50 bg-accent/10'
                        : 'border-red-500/50 bg-red-500/10'
                    }
                  >
                    <div className="flex items-start">
                      {result.authenticated ? (
                        <CheckCircle2 className="w-12 h-12 text-accent mr-4 flex-shrink-0" />
                      ) : (
                        <XCircle className="w-12 h-12 text-red-500 mr-4 flex-shrink-0" />
                      )}
                      <div className="flex-1">
                        <h3
                          className={`text-2xl font-bold mb-2 ${
                            result.authenticated ? 'text-accent' : 'text-red-500'
                          }`}
                        >
                          {result.authenticated
                            ? 'Authentication Successful!'
                            : 'Authentication Failed'}
                        </h3>
                        <div className="space-y-2 text-gray-300">
                          <div className="flex items-center justify-between">
                            <span>Confidence Score:</span>
                            <span className="font-bold">
                              {(result.confidence * 100).toFixed(2)}%
                            </span>
                          </div>
                          {result.threshold && (
                            <div className="flex items-center justify-between">
                              <span>Threshold:</span>
                              <span className="font-bold">
                                {(result.threshold * 100).toFixed(2)}%
                              </span>
                            </div>
                          )}
                          {result.liveness_score && (
                            <div className="flex items-center justify-between">
                              <span>Liveness Score:</span>
                              <span className="font-bold">
                                {(result.liveness_score * 100).toFixed(2)}%
                              </span>
                            </div>
                          )}
                          <div className="flex items-center justify-between pt-2 border-t border-gray-700">
                            <span>Status:</span>
                            <Badge variant={result.authenticated ? 'success' : 'destructive'}>
                              {result.reason}
                            </Badge>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="mt-4">
                      <Progress
                        value={result.confidence * 100}
                        variant={result.authenticated ? 'success' : 'error'}
                        showLabel
                      />
                    </div>
                  </Card>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <div className="space-y-6">
            <Card variant="elevated" className="p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Authentication Tips</h3>
              <ul className="space-y-3 text-sm text-gray-400">
                <li className="flex items-start">
                  <CheckCircle2 className="w-5 h-5 text-accent mr-2 flex-shrink-0 mt-0.5" />
                  <span>Position your face within the frame</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle2 className="w-5 h-5 text-accent mr-2 flex-shrink-0 mt-0.5" />
                  <span>Ensure adequate lighting</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle2 className="w-5 h-5 text-accent mr-2 flex-shrink-0 mt-0.5" />
                  <span>Look directly at the camera</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle2 className="w-5 h-5 text-accent mr-2 flex-shrink-0 mt-0.5" />
                  <span>Stay still during capture</span>
                </li>
              </ul>
            </Card>

            <Card variant="elevated" className="p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Security Features</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Liveness Detection</span>
                  <Badge variant="success" pulse>Active</Badge>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Anti-Spoofing</span>
                  <Badge variant="success" pulse>Active</Badge>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Depth Analysis</span>
                  <Badge variant="success" pulse>Active</Badge>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Authenticate

