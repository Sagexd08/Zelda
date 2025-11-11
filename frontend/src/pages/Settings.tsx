import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Settings as SettingsIcon,
  Save,
  RotateCcw,
  Shield,
  Sliders,
  Cpu,
  Database,
  CheckCircle2
} from 'lucide-react'
import { toast } from 'sonner'
import { Card } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { getSystemInfo } from '../services/api'

const Settings = () => {
  const [systemInfo, setSystemInfo] = useState<any>(null)
  const [settings, setSettings] = useState({
    faceSize: 112,
    verificationThreshold: 0.6,
    livenessThreshold: 0.5,
    minRegistrationSamples: 3,
    enableDepthEstimation: false,
    enableTemporalLiveness: false,
    enableVoiceAuth: false,
    enableChallengeResponse: false,
    enableOnlineLearning: false,
    enableAdaptiveThreshold: false,
    enableBiasMonitoring: false,
    apiWorkers: 4,
    maxConcurrentRequests: 100,
    cacheEnabled: true,
    logLevel: 'INFO',
  })

  useEffect(() => {
    const fetchSystemInfo = async () => {
      try {
        const info = await getSystemInfo()
        setSystemInfo(info)
        
        if (info.features) {
          setSettings(prev => ({
            ...prev,
            enableDepthEstimation: info.features.depth_estimation,
            enableTemporalLiveness: info.features.temporal_liveness,
            enableVoiceAuth: info.features.voice_authentication,
            enableChallengeResponse: info.features.challenge_response,
            enableOnlineLearning: info.features.adaptive_learning,
            enableAdaptiveThreshold: info.features.adaptive_threshold,
            enableBiasMonitoring: info.features.bias_monitoring,
          }))
        }
        
        if (info.models) {
          setSettings(prev => ({
            ...prev,
            faceSize: info.models.face_size,
            verificationThreshold: info.models.verification_threshold,
            livenessThreshold: info.models.liveness_threshold,
            minRegistrationSamples: info.models.min_registration_samples,
          }))
        }
      } catch (error) {
        console.error('Error fetching system info:', error)
      }
    }

    fetchSystemInfo()
  }, [])

  const handleSave = () => {
    toast.success('Settings saved successfully!')
  }

  const handleReset = () => {
    toast.info('Settings reset to defaults')
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6 pb-16">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div className="flex items-center">
          <div className="w-12 h-12 bg-gradient-blue rounded-xl flex items-center justify-center mr-4">
            <SettingsIcon className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-bold text-white">System Settings</h1>
            <p className="text-gray-400 mt-1">Configure system parameters and features</p>
          </div>
        </div>
      </motion.div>

      {systemInfo && (
        <Card variant="gradient" className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-white mb-1">System Information</h3>
              <p className="text-gray-400 text-sm">
                Version {systemInfo.version} • Environment: {systemInfo.environment}
              </p>
            </div>
            <div className="flex items-center space-x-2 glass px-4 py-2 rounded-lg">
              <CheckCircle2 className="w-5 h-5 text-green-500" />
              <span className="text-white font-medium">Operational</span>
            </div>
          </div>
        </Card>
      )}

      <Card variant="elevated" className="p-6">
        <div className="flex items-center mb-6">
          <Sliders className="w-6 h-6 text-accent mr-3" />
          <h2 className="text-2xl font-bold text-white">Detection Parameters</h2>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Face Size (pixels)
            </label>
            <input
              type="number"
              value={settings.faceSize}
              onChange={(e) => setSettings({ ...settings, faceSize: parseInt(e.target.value) })}
              className="w-full px-4 py-2 bg-dark-800 border border-dark-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-accent"
            />
            <p className="text-xs text-gray-500 mt-1">Minimum face size for detection</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Verification Threshold: {settings.verificationThreshold.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={settings.verificationThreshold}
              onChange={(e) => setSettings({ ...settings, verificationThreshold: parseFloat(e.target.value) })}
              className="w-full"
            />
            <p className="text-xs text-gray-500 mt-1">Similarity threshold for authentication</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Liveness Threshold: {settings.livenessThreshold.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={settings.livenessThreshold}
              onChange={(e) => setSettings({ ...settings, livenessThreshold: parseFloat(e.target.value) })}
              className="w-full"
            />
            <p className="text-xs text-gray-500 mt-1">Threshold for liveness detection</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Min Registration Samples
            </label>
            <input
              type="number"
              min="1"
              max="10"
              value={settings.minRegistrationSamples}
              onChange={(e) => setSettings({ ...settings, minRegistrationSamples: parseInt(e.target.value) })}
              className="w-full px-4 py-2 bg-dark-800 border border-dark-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-accent"
            />
            <p className="text-xs text-gray-500 mt-1">Minimum images required for registration</p>
          </div>
        </div>
      </Card>

      <Card variant="elevated" className="p-6">
        <div className="flex items-center mb-6">
          <Shield className="w-6 h-6 text-green-500 mr-3" />
          <h2 className="text-2xl font-bold text-white">Security Features</h2>
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          {[
            { key: 'enableDepthEstimation', label: 'Depth Estimation', desc: 'Use depth maps for improved liveness' },
            { key: 'enableTemporalLiveness', label: 'Temporal Liveness', desc: 'Analyze video sequences for anti-spoofing' },
            { key: 'enableVoiceAuth', label: 'Voice Authentication', desc: 'Enable multimodal voice verification' },
            { key: 'enableChallengeResponse', label: 'Challenge Response', desc: 'Interactive liveness challenges' },
            { key: 'enableOnlineLearning', label: 'Online Learning', desc: 'Continuous model improvement' },
            { key: 'enableAdaptiveThreshold', label: 'Adaptive Threshold', desc: 'Per-user threshold calibration' },
            { key: 'enableBiasMonitoring', label: 'Bias Monitoring', desc: 'Track and mitigate algorithmic bias' },
          ].map((feature) => (
            <div key={feature.key} className="flex items-center justify-between p-4 bg-dark-800 rounded-lg">
              <div className="flex-1">
                <p className="text-white font-medium">{feature.label}</p>
                <p className="text-xs text-gray-500">{feature.desc}</p>
              </div>
              <Badge variant={settings[feature.key as keyof typeof settings] ? 'success' : 'secondary'}>
                {settings[feature.key as keyof typeof settings] ? 'ON' : 'OFF'}
              </Badge>
            </div>
          ))}
        </div>
      </Card>

      <Card variant="elevated" className="p-6">
        <div className="flex items-center mb-6">
          <Cpu className="w-6 h-6 text-yellow-500 mr-3" />
          <h2 className="text-2xl font-bold text-white">Performance Settings</h2>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              API Workers
            </label>
            <input
              type="number"
              min="1"
              max="16"
              value={settings.apiWorkers}
              onChange={(e) => setSettings({ ...settings, apiWorkers: parseInt(e.target.value) })}
              className="w-full px-4 py-2 bg-dark-800 border border-dark-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-accent"
            />
            <p className="text-xs text-gray-500 mt-1">Number of worker processes</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Max Concurrent Requests
            </label>
            <input
              type="number"
              min="10"
              max="1000"
              step="10"
              value={settings.maxConcurrentRequests}
              onChange={(e) => setSettings({ ...settings, maxConcurrentRequests: parseInt(e.target.value) })}
              className="w-full px-4 py-2 bg-dark-800 border border-dark-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-accent"
            />
            <p className="text-xs text-gray-500 mt-1">Maximum simultaneous requests</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Log Level
            </label>
            <select
              value={settings.logLevel}
              onChange={(e) => setSettings({ ...settings, logLevel: e.target.value })}
              className="w-full px-4 py-2 bg-dark-800 border border-dark-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-accent"
            >
              <option value="DEBUG">DEBUG</option>
              <option value="INFO">INFO</option>
              <option value="WARNING">WARNING</option>
              <option value="ERROR">ERROR</option>
            </select>
            <p className="text-xs text-gray-500 mt-1">Logging verbosity level</p>
          </div>

          <div className="flex items-center justify-between p-4 bg-dark-800 rounded-lg">
            <div>
              <p className="text-white font-medium">Enable Caching</p>
              <p className="text-xs text-gray-500">Cache embeddings for faster lookup</p>
            </div>
            <Badge variant={settings.cacheEnabled ? 'success' : 'secondary'}>
              {settings.cacheEnabled ? 'ON' : 'OFF'}
            </Badge>
          </div>
        </div>
      </Card>

      <Card variant="elevated" className="p-6">
        <div className="flex items-center mb-6">
          <Database className="w-6 h-6 text-blue-500 mr-3" />
          <h2 className="text-2xl font-bold text-white">Database Information</h2>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          <div className="text-center p-4 bg-dark-800 rounded-lg">
            <p className="text-gray-400 text-sm mb-1">Database Type</p>
            <p className="text-white font-semibold text-lg">SQLite</p>
          </div>
          <div className="text-center p-4 bg-dark-800 rounded-lg">
            <p className="text-gray-400 text-sm mb-1">Size</p>
            <p className="text-white font-semibold text-lg">24.5 MB</p>
          </div>
          <div className="text-center p-4 bg-dark-800 rounded-lg">
            <p className="text-gray-400 text-sm mb-1">Backup Status</p>
            <Badge variant="success">Up to date</Badge>
          </div>
        </div>
      </Card>

      <div className="flex space-x-4">
        <Button onClick={handleSave} icon={<Save className="w-5 h-5" />} size="lg" className="flex-1">
          Save Settings
        </Button>
        <Button onClick={handleReset} icon={<RotateCcw className="w-5 h-5" />} variant="secondary" size="lg">
          Reset to Defaults
        </Button>
      </div>
    </div>
  )
}

export default Settings

