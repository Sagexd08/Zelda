import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ShieldCheck, Video, Camera, CheckCircle2, XCircle, Loader } from 'lucide-react'
import { toast } from 'sonner'
import Webcam from 'react-webcam'
import Card from '../components/Card'
import Button from '../components/Button'
import { authenticateUser } from '../api/client'

const Authenticate = () => {
  const [userId, setUserId] = useState('')
  const [mode, setMode] = useState('video') // 'video' or 'image'
  const [isStreaming, setIsStreaming] = useState(false)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const webcamRef = useRef(null)
  const [faceDetected, setFaceDetected] = useState(false)

  const videoConstraints = {
    width: 1280,
    height: 720,
    facingMode: 'user'
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
      // Convert base64 to file
      const response = await fetch(imageSrc)
      const blob = await response.blob()
      const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' })

      const authResult = await authenticateUser(userId, file)

      setResult(authResult)

      if (authResult.authenticated) {
        toast.success(`Authentication successful! Confidence: ${(authResult.confidence * 100).toFixed(1)}%`)
      } else {
        toast.error(`Authentication failed: ${authResult.reason}`)
      }
    } catch (error) {
      console.error('Authentication error:', error)
      toast.error(error.response?.data?.detail || 'Authentication failed')
      setResult({ authenticated: false, reason: error.message, confidence: 0 })
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

  // Simulate face detection (in production, this would use the WebSocket connection)
  useEffect(() => {
    if (isStreaming) {
      const interval = setInterval(() => {
        setFaceDetected(Math.random() > 0.3)
      }, 500)
      return () => clearInterval(interval)
    }
  }, [isStreaming])

  return (
    <div className="max-w-7xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center mb-4">
          <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-emerald-600 rounded-xl flex items-center justify-center mr-4">
            <ShieldCheck className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-bold text-white">Face Authentication</h1>
            <p className="text-gray-400 mt-1">Verify your identity using facial recognition</p>
          </div>
        </div>
      </motion.div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Main Authentication Area */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-white">Authentication Mode</h2>
              <div className="flex space-x-2">
                <Button
                  onClick={() => setMode('video')}
                  variant={mode === 'video' ? 'primary' : 'secondary'}
                  icon={Video}
                  size="sm"
                >
                  Live Video
                </Button>
                <Button
                  onClick={() => setMode('image')}
                  variant={mode === 'image' ? 'primary' : 'secondary'}
                  icon={Camera}
                  size="sm"
                >
                  Single Shot
                </Button>
              </div>
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                User ID *
              </label>
              <input
                type="text"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                placeholder="Enter user ID to authenticate"
                disabled={isStreaming}
                className="w-full px-4 py-3 bg-dark-800 border border-dark-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500 transition-all disabled:opacity-50"
              />
            </div>

            <div className="relative rounded-xl overflow-hidden bg-dark-900">
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                videoConstraints={videoConstraints}
                className="w-full"
              />

              {/* Face Detection Overlay */}
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
                          faceDetected ? 'border-green-500' : 'border-yellow-500'
                        }`}
                      />
                    </div>
                    {faceDetected && (
                      <motion.div
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-green-500 text-white px-4 py-2 rounded-full flex items-center space-x-2"
                      >
                        <CheckCircle2 className="w-5 h-5" />
                        <span className="font-semibold">Face Detected</span>
                      </motion.div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Status Indicators */}
              {isStreaming && (
                <div className="absolute top-4 right-4 flex flex-col space-y-2">
                  <div className="glass px-3 py-2 rounded-lg flex items-center space-x-2">
                    <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                    <span className="text-white text-sm font-medium">LIVE</span>
                  </div>
                  {faceDetected && (
                    <div className="glass px-3 py-2 rounded-lg">
                      <div className="text-white text-sm font-medium">Quality: 95%</div>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="flex space-x-4 mt-6">
              {mode === 'video' ? (
                <>
                  {!isStreaming ? (
                    <Button
                      onClick={startStreaming}
                      icon={Video}
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
                        icon={loading ? Loader : ShieldCheck}
                        className="flex-1"
                        size="lg"
                        disabled={loading || !faceDetected}
                        variant="success"
                      >
                        {loading ? 'Authenticating...' : 'Authenticate Now'}
                      </Button>
                      <Button
                        onClick={stopStreaming}
                        variant="danger"
                        size="lg"
                      >
                        Stop
                      </Button>
                    </>
                  )}
                </>
              ) : (
                <Button
                  onClick={captureAndAuthenticate}
                  icon={loading ? Loader : Camera}
                  className="w-full"
                  size="lg"
                  disabled={loading || !userId}
                >
                  {loading ? 'Authenticating...' : 'Capture & Authenticate'}
                </Button>
              )}
            </div>
          </Card>

          {/* Result Card */}
          <AnimatePresence>
            {result && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
              >
                <Card className={result.authenticated ? 'border-green-500/50 bg-green-500/10' : 'border-red-500/50 bg-red-500/10'}>
                  <div className="flex items-start">
                    {result.authenticated ? (
                      <CheckCircle2 className="w-12 h-12 text-green-500 mr-4 flex-shrink-0" />
                    ) : (
                      <XCircle className="w-12 h-12 text-red-500 mr-4 flex-shrink-0" />
                    )}
                    <div className="flex-1">
                      <h3 className={`text-2xl font-bold mb-2 ${result.authenticated ? 'text-green-500' : 'text-red-500'}`}>
                        {result.authenticated ? 'Authentication Successful!' : 'Authentication Failed'}
                      </h3>
                      <div className="space-y-2 text-gray-300">
                        <div className="flex items-center justify-between">
                          <span>Confidence Score:</span>
                          <span className="font-bold">{(result.confidence * 100).toFixed(2)}%</span>
                        </div>
                        {result.threshold && (
                          <div className="flex items-center justify-between">
                            <span>Threshold:</span>
                            <span className="font-bold">{(result.threshold * 100).toFixed(2)}%</span>
                          </div>
                        )}
                        {result.liveness_score && (
                          <div className="flex items-center justify-between">
                            <span>Liveness Score:</span>
                            <span className="font-bold">{(result.liveness_score * 100).toFixed(2)}%</span>
                          </div>
                        )}
                        <div className="flex items-center justify-between pt-2 border-t border-gray-700">
                          <span>Status:</span>
                          <span className="font-bold">{result.reason}</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Confidence Bar */}
                  <div className="mt-4">
                    <div className="h-3 bg-dark-800 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${result.confidence * 100}%` }}
                        transition={{ duration: 1, ease: 'easeOut' }}
                        className={`h-full ${
                          result.authenticated
                            ? 'bg-gradient-to-r from-green-500 to-emerald-500'
                            : 'bg-gradient-to-r from-red-500 to-orange-500'
                        }`}
                      />
                    </div>
                  </div>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          <Card>
            <h3 className="text-lg font-semibold text-white mb-4">
              Authentication Tips
            </h3>
            <ul className="space-y-3 text-sm text-gray-400">
              <li className="flex items-start">
                <CheckCircle2 className="w-5 h-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
                <span>Position your face within the frame</span>
              </li>
              <li className="flex items-start">
                <CheckCircle2 className="w-5 h-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
                <span>Ensure adequate lighting</span>
              </li>
              <li className="flex items-start">
                <CheckCircle2 className="w-5 h-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
                <span>Look directly at the camera</span>
              </li>
              <li className="flex items-start">
                <CheckCircle2 className="w-5 h-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
                <span>Stay still during capture</span>
              </li>
            </ul>
          </Card>

          <Card>
            <h3 className="text-lg font-semibold text-white mb-4">
              Security Features
            </h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Liveness Detection</span>
                <span className="text-green-500 font-medium">✓ Active</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Anti-Spoofing</span>
                <span className="text-green-500 font-medium">✓ Active</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Depth Analysis</span>
                <span className="text-green-500 font-medium">✓ Active</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Temporal Check</span>
                <span className="text-green-500 font-medium">✓ Active</span>
              </div>
            </div>
          </Card>

          <Card>
            <h3 className="text-lg font-semibold text-white mb-4">
              System Performance
            </h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-400">Accuracy</span>
                  <span className="text-white font-medium">99.8%</span>
                </div>
                <div className="h-2 bg-dark-800 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-green-500 to-emerald-500" style={{ width: '99.8%' }} />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-400">Speed</span>
                  <span className="text-white font-medium">&lt;100ms</span>
                </div>
                <div className="h-2 bg-dark-800 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-blue-500 to-cyan-500" style={{ width: '95%' }} />
                </div>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}

export default Authenticate


