import { useState } from 'react'
import { motion } from 'framer-motion'
import { UserPlus, Upload, Camera, CheckCircle2, AlertCircle, Activity } from 'lucide-react'
import { toast } from 'sonner'
import { Card } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import WebcamCapture from '../components/features/WebcamCapture'
import { registerUser } from '../services/api'

const Register = () => {
  const extractErrorMessage = (error: any, fallback = 'Registration failed') => {
    const detail = error?.response?.data?.detail ?? error?.response?.data
    if (typeof detail === 'string') {
      return detail
    }
    if (detail && typeof detail === 'object') {
      if (detail.error) return detail.error
      if (detail.message) return detail.message
      if (detail.detail) return detail.detail
      return JSON.stringify(detail)
    }
    if (error?.message) return error.message
    return fallback
  }

  const [userId, setUserId] = useState('')
  const [capturedImages, setCapturedImages] = useState<string[]>([])
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [captureMode, setCaptureMode] = useState<'camera' | 'upload'>('camera')

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    setUploadedFiles(files)
    
    const previews = files.map(file => URL.createObjectURL(file))
    setCapturedImages(previews)
  }

  const handleWebcamCapture = (images: string | string[]) => {
    setCapturedImages(Array.isArray(images) ? images : [images])
  }

  const dataURLtoFile = (dataurl: string, filename: string): File => {
    const arr = dataurl.split(',')
    const mime = arr[0].match(/:(.*?);/)?.[1] || 'image/jpeg'
    const bstr = atob(arr[1])
    let n = bstr.length
    const u8arr = new Uint8Array(n)
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n)
    }
    return new File([u8arr], filename, { type: mime })
  }

  const handleRegister = async () => {
    if (!userId.trim()) {
      toast.error('Please enter a User ID')
      return
    }

    if (capturedImages.length === 0) {
      toast.error('Please capture or upload at least one image')
      return
    }

    setLoading(true)
    setResult(null)

    try {
      let imageFiles: File[] = []
      
      if (captureMode === 'camera') {
        imageFiles = capturedImages.map((img, i) => 
          dataURLtoFile(img, `capture_${i}.jpg`)
        )
      } else {
        imageFiles = uploadedFiles
      }

      const response = await registerUser(userId, imageFiles)

      if (response.success) {
        setResult(response)
        toast.success(`Successfully registered ${userId}!`)
        
        setTimeout(() => {
          setUserId('')
          setCapturedImages([])
          setUploadedFiles([])
          setResult(null)
        }, 3000)
      } else {
        toast.error(response.error || 'Registration failed')
        setResult({ success: false, error: response.error })
      }
    } catch (error) {
      console.error('Registration error:', error)
      const message = extractErrorMessage(error)
      toast.error(message)
      setResult({ success: false, error: message })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-6xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center mb-4">
          <div className="w-12 h-12 bg-gradient-blue rounded-xl flex items-center justify-center mr-4">
            <UserPlus className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-bold text-white">Face Registration</h1>
            <p className="text-gray-400 mt-1">Register a new user to the system</p>
          </div>
        </div>
      </motion.div>

      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <Card variant="elevated" className="p-6">
            <h2 className="text-2xl font-bold text-white mb-6">User Information</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  User ID *
                </label>
                <input
                  type="text"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  placeholder="Enter unique user identifier"
                  className="w-full px-4 py-3 bg-dark-800 border border-dark-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-accent transition-all"
                />
                <p className="text-xs text-gray-500 mt-1">
                  This will be used to identify the user during authentication
                </p>
              </div>
            </div>
          </Card>

          <Card variant="elevated" className="p-6 sm:p-8">
            <h2 className="text-xl sm:text-2xl font-bold text-white mb-6">Capture Face Images</h2>
            
            <div className="flex flex-wrap gap-2 sm:gap-3 mb-6">
              <Button
                onClick={() => setCaptureMode('camera')}
                variant={captureMode === 'camera' ? 'default' : 'secondary'}
                icon={<Camera className="w-4 h-4" />}
                size="sm"
              >
                Camera
              </Button>
              <Button
                onClick={() => setCaptureMode('upload')}
                variant={captureMode === 'upload' ? 'default' : 'secondary'}
                icon={<Upload className="w-4 h-4" />}
                size="sm"
              >
                Upload
              </Button>
            </div>

            {captureMode === 'camera' ? (
              <WebcamCapture 
                onCapture={handleWebcamCapture}
                multipleCaptures={true}
              />
            ) : (
              <div>
                <div className="border-2 border-dashed border-dark-600 rounded-lg p-12 text-center hover:border-accent transition-all">
                  <Upload className="w-12 h-12 mx-auto text-gray-500 mb-4" />
                  <p className="text-gray-400 mb-2">
                    Drop images here or click to browse
                  </p>
                  <p className="text-sm text-gray-500 mb-4">
                    Upload multiple images for better accuracy
                  </p>
                  <input
                    type="file"
                    multiple
                    accept="image/*"
                    onChange={handleFileUpload}
                    className="hidden"
                    id="file-upload"
                  />
                  <label htmlFor="file-upload">
                    <Button icon={<Upload className="w-4 h-4" />}>
                      Select Files
                    </Button>
                  </label>
                </div>

                {capturedImages.length > 0 && (
                  <div className="grid grid-cols-3 gap-4 mt-6">
                    {capturedImages.map((img, index) => (
                      <div key={index} className="relative group">
                        <img
                          src={img}
                          alt={`Upload ${index + 1}`}
                          className="w-full h-32 object-cover rounded-lg"
                        />
                        <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg flex items-center justify-center">
                          <CheckCircle2 className="w-8 h-8 text-green-500" />
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </Card>

          <Button
            onClick={handleRegister}
            disabled={loading || !userId || capturedImages.length === 0}
            icon={<UserPlus className="w-5 h-5" />}
            className="w-full"
            size="lg"
            loading={loading}
          >
            Register User
          </Button>
        </div>

        <div className="space-y-6">
          <Card variant="elevated" className="p-6">
            <h3 className="text-lg font-semibold text-white mb-4">
              Registration Tips
            </h3>
            <ul className="space-y-3 text-sm text-gray-400">
              <li className="flex items-start">
                <CheckCircle2 className="w-5 h-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
                <span>Capture multiple images from different angles</span>
              </li>
              <li className="flex items-start">
                <CheckCircle2 className="w-5 h-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
                <span>Ensure good lighting conditions</span>
              </li>
              <li className="flex items-start">
                <CheckCircle2 className="w-5 h-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
                <span>Look directly at the camera</span>
              </li>
              <li className="flex items-start">
                <CheckCircle2 className="w-5 h-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
                <span>Remove glasses or accessories if possible</span>
              </li>
              <li className="flex items-start">
                <CheckCircle2 className="w-5 h-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
                <span>Maintain a neutral expression</span>
              </li>
            </ul>
          </Card>

          {result && (
            <motion.div
              initial={{ opacity: 0, y: -20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.3, type: 'spring', stiffness: 300 }}
              className="w-full"
            >
              <Card 
                variant={result.success ? 'gradient' : 'outlined'}
                className={`
                  w-full p-6 sm:p-8
                  ${result.success 
                    ? 'border-green-500/50 bg-gradient-to-br from-green-500/20 via-green-500/10 to-transparent backdrop-blur-xl shadow-lg shadow-green-500/20' 
                    : 'border-red-500/50 bg-gradient-to-br from-red-500/20 via-red-500/10 to-transparent backdrop-blur-xl shadow-lg shadow-red-500/20'
                  }
                  relative overflow-hidden
                `}
              >
                {/* Animated background glow */}
                <div className={`absolute inset-0 opacity-20 ${result.success ? 'bg-green-500' : 'bg-red-500'} blur-2xl animate-pulse`}></div>
                
                <div className="relative flex items-start gap-4">
                  <motion.div
                    initial={{ scale: 0, rotate: -180 }}
                    animate={{ scale: 1, rotate: 0 }}
                    transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
                    className={`
                      flex-shrink-0 w-12 h-12 sm:w-14 sm:h-14 rounded-full flex items-center justify-center
                      ${result.success 
                        ? 'bg-green-500/20 border-2 border-green-500/50' 
                        : 'bg-red-500/20 border-2 border-red-500/50'
                      }
                    `}
                  >
                    {result.success ? (
                      <CheckCircle2 className="w-7 h-7 text-green-400" />
                    ) : (
                      <AlertCircle className="w-7 h-7 text-red-400" />
                    )}
                  </motion.div>
                  <div className="flex-1 min-w-0">
                    <motion.h3 
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.1 }}
                      className={`
                        text-lg sm:text-xl font-bold mb-3 sm:mb-4
                        ${result.success ? 'text-green-400' : 'text-red-400'}
                      `}
                    >
                      {result.success ? 'Registration Successful!' : 'Registration Failed'}
                    </motion.h3>
                    {result.success ? (
                      <motion.div 
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3 }}
                        className="space-y-2 text-sm sm:text-base"
                      >
                        <div className="flex items-center justify-between py-2 px-3 sm:py-2.5 sm:px-4 rounded-lg bg-dark-800/50 border border-dark-600/50">
                          <span className="text-gray-400">User ID:</span>
                          <span className="text-white font-semibold">{result.user_id}</span>
                        </div>
                        <div className="flex items-center justify-between py-2 px-3 sm:py-2.5 sm:px-4 rounded-lg bg-dark-800/50 border border-dark-600/50">
                          <span className="text-gray-400">Samples:</span>
                          <span className="text-white font-semibold">{result.valid_samples}/{result.samples_processed}</span>
                        </div>
                        <div className="flex items-center justify-between py-2 px-3 sm:py-2.5 sm:px-4 rounded-lg bg-dark-800/50 border border-dark-600/50">
                          <span className="text-gray-400">Quality Score:</span>
                          <span className="text-green-400 font-semibold">{(result.avg_quality_score * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex items-center justify-between py-2 px-3 sm:py-2.5 sm:px-4 rounded-lg bg-dark-800/50 border border-dark-600/50">
                          <span className="text-gray-400">Liveness Score:</span>
                          <span className="text-green-400 font-semibold">{(result.avg_liveness_score * 100).toFixed(1)}%</span>
                        </div>
                      </motion.div>
                    ) : (
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3 }}
                        className="p-4 sm:p-5 rounded-lg bg-dark-800/50 border border-red-500/20"
                      >
                        <p className="text-sm sm:text-base text-gray-300 leading-relaxed whitespace-pre-wrap break-words">
                          {result.error}
                        </p>
                      </motion.div>
                    )}
                  </div>
                </div>
              </Card>
            </motion.div>
          )}

          <Card variant="elevated" className="p-6 sm:p-8">
            <h3 className="text-lg sm:text-xl font-semibold text-white mb-5 sm:mb-6 flex items-center gap-2">
              <Activity className="w-5 h-5 text-accent" />
              System Status
            </h3>
            <div className="space-y-3 sm:space-y-4">
              <div className="flex items-center justify-between py-2 px-3 rounded-lg bg-dark-800/30 border border-dark-600/50">
                <span className="text-sm sm:text-base text-gray-400">Face Detection</span>
                <Badge variant="success">Active</Badge>
              </div>
              <div className="flex items-center justify-between py-2 px-3 rounded-lg bg-dark-800/30 border border-dark-600/50">
                <span className="text-sm sm:text-base text-gray-400">Liveness Check</span>
                <Badge variant="success">Active</Badge>
              </div>
              <div className="flex items-center justify-between py-2 px-3 rounded-lg bg-dark-800/30 border border-dark-600/50">
                <span className="text-sm sm:text-base text-gray-400">Quality Assessment</span>
                <Badge variant="success">Active</Badge>
              </div>
              <div className="flex items-center justify-between py-2 px-3 rounded-lg bg-dark-800/30 border border-dark-600/50">
                <span className="text-sm sm:text-base text-gray-400">Min. Samples</span>
                <span className="text-white font-semibold">1</span>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}

export default Register

