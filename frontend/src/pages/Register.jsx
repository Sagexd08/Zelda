import { useState } from 'react'
import { motion } from 'framer-motion'
import { UserPlus, Upload, Camera, CheckCircle2, AlertCircle } from 'lucide-react'
import { toast } from 'sonner'
import Card from '../components/Card'
import Button from '../components/Button'
import WebcamCapture from '../components/WebcamCapture'
import { registerUser } from '../api/client'

const Register = () => {
  const [userId, setUserId] = useState('')
  const [capturedImages, setCapturedImages] = useState([])
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [captureMode, setCaptureMode] = useState('camera') // 'camera' or 'upload'

  const handleFileUpload = (e) => {
    const files = Array.from(e.target.files)
    setUploadedFiles(files)
    
    // Preview images
    const previews = files.map(file => URL.createObjectURL(file))
    setCapturedImages(previews)
  }

  const handleWebcamCapture = (images) => {
    setCapturedImages(Array.isArray(images) ? images : [images])
  }

  const dataURLtoFile = (dataurl, filename) => {
    const arr = dataurl.split(',')
    const mime = arr[0].match(/:(.*?);/)[1]
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
      let imageFiles = []
      
      if (captureMode === 'camera') {
        // Convert base64 images to files
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
        
        // Reset form after 2 seconds
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
      toast.error(error.response?.data?.detail || 'Registration failed')
      setResult({ success: false, error: error.message })
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
          <div className="w-12 h-12 bg-gradient-to-br from-primary-500 to-purple-600 rounded-xl flex items-center justify-center mr-4">
            <UserPlus className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-bold text-white">Face Registration</h1>
            <p className="text-gray-400 mt-1">Register a new user to the system</p>
          </div>
        </div>
      </motion.div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Main Registration Form */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
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
                  className="w-full px-4 py-3 bg-dark-800 border border-dark-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500 transition-all"
                />
                <p className="text-xs text-gray-500 mt-1">
                  This will be used to identify the user during authentication
                </p>
              </div>
            </div>
          </Card>

          <Card>
            <h2 className="text-2xl font-bold text-white mb-6">Capture Face Images</h2>
            
            {/* Mode Toggle */}
            <div className="flex space-x-2 mb-6">
              <Button
                onClick={() => setCaptureMode('camera')}
                variant={captureMode === 'camera' ? 'primary' : 'secondary'}
                icon={Camera}
                size="sm"
              >
                Camera
              </Button>
              <Button
                onClick={() => setCaptureMode('upload')}
                variant={captureMode === 'upload' ? 'primary' : 'secondary'}
                icon={Upload}
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
                <div className="border-2 border-dashed border-dark-600 rounded-lg p-12 text-center hover:border-primary-500 transition-all">
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
                    <Button as="span" icon={Upload}>
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
            icon={UserPlus}
            className="w-full"
            size="lg"
          >
            {loading ? 'Registering...' : 'Register User'}
          </Button>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          <Card>
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
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
            >
              <Card className={result.success ? 'border-green-500/50 bg-green-500/10' : 'border-red-500/50 bg-red-500/10'}>
                <div className="flex items-start">
                  {result.success ? (
                    <CheckCircle2 className="w-6 h-6 text-green-500 mr-3 flex-shrink-0" />
                  ) : (
                    <AlertCircle className="w-6 h-6 text-red-500 mr-3 flex-shrink-0" />
                  )}
                  <div className="flex-1">
                    <h3 className={`font-semibold mb-2 ${result.success ? 'text-green-500' : 'text-red-500'}`}>
                      {result.success ? 'Registration Successful!' : 'Registration Failed'}
                    </h3>
                    {result.success ? (
                      <div className="text-sm text-gray-300 space-y-1">
                        <p>User ID: {result.user_id}</p>
                        <p>Samples: {result.valid_samples}/{result.samples_processed}</p>
                        <p>Quality: {(result.avg_quality_score * 100).toFixed(1)}%</p>
                        <p>Liveness: {(result.avg_liveness_score * 100).toFixed(1)}%</p>
                      </div>
                    ) : (
                      <p className="text-sm text-gray-300">{result.error}</p>
                    )}
                  </div>
                </div>
              </Card>
            </motion.div>
          )}

          <Card>
            <h3 className="text-lg font-semibold text-white mb-4">
              System Status
            </h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Face Detection</span>
                <span className="text-green-500 font-medium">Active</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Liveness Check</span>
                <span className="text-green-500 font-medium">Active</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Quality Assessment</span>
                <span className="text-green-500 font-medium">Active</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Min. Samples</span>
                <span className="text-white font-medium">3</span>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}

export default Register


