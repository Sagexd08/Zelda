import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Users, 
  ShieldCheck, 
  Activity, 
  TrendingUp,
  Clock,
  AlertCircle,
  CheckCircle2,
  Zap
} from 'lucide-react'
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts'
import Card from '../components/Card'
import { getSystemInfo, getHealth } from '../api/client'

const Dashboard = () => {
  const [systemInfo, setSystemInfo] = useState(null)
  const [health, setHealth] = useState(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [sysInfo, healthData] = await Promise.all([
          getSystemInfo(),
          getHealth()
        ])
        setSystemInfo(sysInfo)
        setHealth(healthData)
      } catch (error) {
        console.error('Error fetching system data:', error)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 10000) // Refresh every 10 seconds
    return () => clearInterval(interval)
  }, [])

  // Sample data for charts
  const authData = Array.from({ length: 24 }, (_, i) => ({
    hour: `${i}:00`,
    successful: Math.floor(Math.random() * 100) + 50,
    failed: Math.floor(Math.random() * 20) + 5,
  }))

  const performanceData = Array.from({ length: 30 }, (_, i) => ({
    day: `Day ${i + 1}`,
    accuracy: 98 + Math.random() * 2,
    latency: 80 + Math.random() * 40,
  }))

  const modelData = [
    { name: 'ArcFace', accuracy: 99.8, color: '#667eea' },
    { name: 'FaceNet', accuracy: 99.6, color: '#764ba2' },
    { name: 'MobileFaceNet', accuracy: 99.2, color: '#f093fb' },
    { name: 'Liveness CNN', accuracy: 98.5, color: '#4facfe' },
  ]

  const livenessData = [
    { name: 'Genuine', value: 850, color: '#10b981' },
    { name: 'Spoof Detected', value: 45, color: '#ef4444' },
    { name: 'Uncertain', value: 15, color: '#f59e0b' },
  ]

  const stats = [
    {
      label: 'Total Users',
      value: '1,284',
      change: '+12.5%',
      icon: Users,
      color: 'from-blue-500 to-cyan-500',
      positive: true
    },
    {
      label: 'Auth Today',
      value: '856',
      change: '+8.2%',
      icon: ShieldCheck,
      color: 'from-green-500 to-emerald-500',
      positive: true
    },
    {
      label: 'Success Rate',
      value: '98.5%',
      change: '+0.3%',
      icon: Activity,
      color: 'from-purple-500 to-pink-500',
      positive: true
    },
    {
      label: 'Avg Latency',
      value: '87ms',
      change: '-12ms',
      icon: Zap,
      color: 'from-yellow-500 to-orange-500',
      positive: true
    },
  ]

  return (
    <div className="space-y-6 pb-16">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-4xl font-bold text-white">Dashboard</h1>
          <p className="text-gray-400 mt-1">Real-time system metrics and analytics</p>
        </div>
        {health && (
          <div className="flex items-center space-x-2 glass px-4 py-2 rounded-lg">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <span className="text-white font-medium">System Healthy</span>
          </div>
        )}
      </motion.div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => {
          const Icon = stat.icon
          return (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <Card hover={false}>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <p className="text-gray-400 text-sm mb-1">{stat.label}</p>
                    <p className="text-3xl font-bold text-white mb-2">{stat.value}</p>
                    <div className={`flex items-center text-sm ${stat.positive ? 'text-green-500' : 'text-red-500'}`}>
                      <TrendingUp className="w-4 h-4 mr-1" />
                      {stat.change}
                    </div>
                  </div>
                  <div className={`w-12 h-12 bg-gradient-to-br ${stat.color} rounded-xl flex items-center justify-center`}>
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                </div>
              </Card>
            </motion.div>
          )
        })}
      </div>

      {/* Charts Row 1 */}
      <div className="grid lg:grid-cols-2 gap-6">
        <Card>
          <h2 className="text-xl font-bold text-white mb-4">Authentication Activity (24h)</h2>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={authData}>
              <defs>
                <linearGradient id="colorSuccess" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="colorFailed" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="hour" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                labelStyle={{ color: '#fff' }}
              />
              <Legend />
              <Area type="monotone" dataKey="successful" stroke="#10b981" fillOpacity={1} fill="url(#colorSuccess)" />
              <Area type="monotone" dataKey="failed" stroke="#ef4444" fillOpacity={1} fill="url(#colorFailed)" />
            </AreaChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <h2 className="text-xl font-bold text-white mb-4">Model Performance</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={modelData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="name" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" domain={[95, 100]} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                labelStyle={{ color: '#fff' }}
              />
              <Bar dataKey="accuracy" radius={[8, 8, 0, 0]}>
                {modelData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Charts Row 2 */}
      <div className="grid lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2">
          <h2 className="text-xl font-bold text-white mb-4">System Performance Trends</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="day" stroke="#94a3b8" />
              <YAxis yAxisId="left" stroke="#94a3b8" />
              <YAxis yAxisId="right" orientation="right" stroke="#94a3b8" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                labelStyle={{ color: '#fff' }}
              />
              <Legend />
              <Line 
                yAxisId="left"
                type="monotone" 
                dataKey="accuracy" 
                stroke="#10b981" 
                strokeWidth={2}
                dot={{ fill: '#10b981', r: 4 }}
              />
              <Line 
                yAxisId="right"
                type="monotone" 
                dataKey="latency" 
                stroke="#f59e0b" 
                strokeWidth={2}
                dot={{ fill: '#f59e0b', r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <h2 className="text-xl font-bold text-white mb-4">Liveness Detection</h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={livenessData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {livenessData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
              />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* System Status */}
      <div className="grid lg:grid-cols-3 gap-6">
        <Card>
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <CheckCircle2 className="w-5 h-5 text-green-500 mr-2" />
            Active Services
          </h3>
          <div className="space-y-3">
            {systemInfo?.features && Object.entries(systemInfo.features).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between text-sm">
                <span className="text-gray-400 capitalize">{key.replace(/_/g, ' ')}</span>
                <span className={`font-medium ${value ? 'text-green-500' : 'text-gray-500'}`}>
                  {value ? '✓ Active' : '○ Disabled'}
                </span>
              </div>
            ))}
          </div>
        </Card>

        <Card>
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <Activity className="w-5 h-5 text-blue-500 mr-2" />
            Model Configuration
          </h3>
          <div className="space-y-3">
            {systemInfo?.models && Object.entries(systemInfo.models).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between text-sm">
                <span className="text-gray-400 capitalize">{key.replace(/_/g, ' ')}</span>
                <span className="text-white font-medium">{value}</span>
              </div>
            ))}
          </div>
        </Card>

        <Card>
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <Clock className="w-5 h-5 text-yellow-500 mr-2" />
            Recent Activity
          </h3>
          <div className="space-y-3">
            <div className="flex items-start">
              <CheckCircle2 className="w-4 h-4 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
              <div className="text-sm">
                <p className="text-white">User authenticated</p>
                <p className="text-gray-500">2 minutes ago</p>
              </div>
            </div>
            <div className="flex items-start">
              <CheckCircle2 className="w-4 h-4 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
              <div className="text-sm">
                <p className="text-white">New user registered</p>
                <p className="text-gray-500">15 minutes ago</p>
              </div>
            </div>
            <div className="flex items-start">
              <AlertCircle className="w-4 h-4 text-yellow-500 mr-2 mt-0.5 flex-shrink-0" />
              <div className="text-sm">
                <p className="text-white">Failed authentication</p>
                <p className="text-gray-500">1 hour ago</p>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  )
}

export default Dashboard


