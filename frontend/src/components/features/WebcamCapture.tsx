import { useRef, useState, useCallback } from 'react'
import Webcam from 'react-webcam'
import { Camera, RotateCcw, Check, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { motion, AnimatePresence } from 'framer-motion'

interface WebcamCaptureProps {
  onCapture?: (images: string | string[]) => void
  multipleCaptures?: boolean
}

const WebcamCapture = ({ onCapture, multipleCaptures = false }: WebcamCaptureProps) => {
  const webcamRef = useRef<Webcam>(null)
  const [capturedImages, setCapturedImages] = useState<string[]>([])
  const [showCamera, setShowCamera] = useState(false)

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot()
    if (imageSrc) {
      if (multipleCaptures) {
        const newImages = [...capturedImages, imageSrc]
        setCapturedImages(newImages)
        onCapture?.(newImages)
      } else {
        setCapturedImages([imageSrc])
        onCapture?.(imageSrc)
        setShowCamera(false)
      }
    }
  }, [webcamRef, capturedImages, multipleCaptures, onCapture])

  const reset = () => {
    setCapturedImages([])
    setShowCamera(true)
    onCapture?.(multipleCaptures ? [] : null as any)
  }

  const removeImage = (index: number) => {
    const newImages = capturedImages.filter((_, i) => i !== index)
    setCapturedImages(newImages)
    onCapture?.(multipleCaptures ? newImages : null as any)
  }

  const videoConstraints = {
    width: 1280,
    height: 720,
    facingMode: 'user' as const
  }

  return (
    <div className="space-y-4">
      <AnimatePresence mode="wait">
        {showCamera ? (
          <motion.div
            key="camera"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="relative rounded-xl overflow-hidden"
          >
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              videoConstraints={videoConstraints}
              className="w-full rounded-xl"
            />
            <div className="absolute inset-0 pointer-events-none">
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                <motion.div 
                  className="w-64 h-80 border-4 border-accent rounded-3xl"
                  initial={{ opacity: 0.5, scale: 0.95 }}
                  animate={{ 
                    opacity: [0.5, 0.8, 0.5],
                    scale: [0.95, 1, 0.95],
                  }}
                  transition={{ 
                    duration: 2,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                />
                <div className="absolute top-0 left-0 w-full h-full flex items-center justify-center text-white text-sm font-medium">
                  <div className="glass px-3 py-1 rounded-full">
                    Position your face in the frame
                  </div>
                </div>
              </div>
            </div>
            <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex space-x-4">
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Button
                  onClick={capture}
                  icon={<Camera className="w-5 h-5" />}
                  size="lg"
                  variant="gradient"
                >
                  Capture
                </Button>
              </motion.div>
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Button
                  onClick={() => setShowCamera(false)}
                  variant="secondary"
                  size="lg"
                >
                  {multipleCaptures && capturedImages.length > 0 ? 'Done' : 'Cancel'}
                </Button>
              </motion.div>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="gallery"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-4"
          >
            {capturedImages.length > 0 ? (
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {capturedImages.map((img, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="relative group"
                  >
                    <img
                      src={img}
                      alt={`Captured ${index + 1}`}
                      className="w-full h-48 object-cover rounded-lg"
                    />
                    <div className="absolute top-2 right-2">
                      <button
                        onClick={() => removeImage(index)}
                        className="bg-red-600 hover:bg-red-700 text-white p-2 rounded-full transition-colors"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                    <div className="absolute bottom-2 left-2 glass px-2 py-1 rounded text-sm text-white">
                      Image {index + 1}
                    </div>
                  </motion.div>
                ))}
              </div>
            ) : (
              <div className="glass rounded-xl p-12 text-center">
                <Camera className="w-16 h-16 mx-auto text-gray-500 mb-4" />
                <p className="text-gray-400 mb-4">No image captured yet</p>
                <Button onClick={() => setShowCamera(true)} icon={<Camera className="w-5 h-5" />}>
                  Open Camera
                </Button>
              </div>
            )}

            {capturedImages.length > 0 && (
              <div className="flex justify-center space-x-4 flex-wrap gap-2">
                {multipleCaptures && (
                  <Button
                    onClick={() => setShowCamera(true)}
                    icon={<Camera className="w-4 h-4" />}
                    variant="secondary"
                  >
                    Capture More
                  </Button>
                )}
                <Button onClick={reset} icon={<RotateCcw className="w-4 h-4" />} variant="secondary">
                  Reset
                </Button>
                <Button icon={<Check className="w-4 h-4" />} variant="default">
                  {multipleCaptures 
                    ? `Use ${capturedImages.length} Image${capturedImages.length > 1 ? 's' : ''}`
                    : 'Use This Image'
                  }
                </Button>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {!showCamera && capturedImages.length === 0 && (
        <Button
          onClick={() => setShowCamera(true)}
          icon={<Camera className="w-5 h-5" />}
          className="w-full"
          size="lg"
        >
          Open Camera
        </Button>
      )}
    </div>
  )
}

export default WebcamCapture

