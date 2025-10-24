import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { 
  Shield, 
  Zap, 
  Eye, 
  Lock, 
  UserPlus, 
  ShieldCheck,
  BarChart3,
  ArrowRight,
  CheckCircle2,
  Scan
} from 'lucide-react'
import Card from '../components/Card'

const Home = () => {
  const features = [
    {
      icon: Shield,
      title: 'Enterprise Security',
      description: 'Military-grade encryption and secure storage of biometric data',
      color: 'from-blue-500 to-cyan-500'
    },
    {
      icon: Zap,
      title: 'Lightning Fast',
      description: 'Authentication in under 100ms with real-time processing',
      color: 'from-yellow-500 to-orange-500'
    },
    {
      icon: Eye,
      title: 'Liveness Detection',
      description: 'Advanced anti-spoofing with temporal and depth analysis',
      color: 'from-purple-500 to-pink-500'
    },
    {
      icon: Lock,
      title: '99.8% Accuracy',
      description: 'State-of-the-art deep learning models with fusion architecture',
      color: 'from-green-500 to-emerald-500'
    }
  ]

  const stats = [
    { label: 'Accuracy Rate', value: '99.8%' },
    { label: 'Response Time', value: '<100ms' },
    { label: 'Active Models', value: '5+' },
    { label: 'Security Level', value: 'Military' }
  ]

  const steps = [
    { icon: UserPlus, title: 'Register', description: 'Capture multiple face samples' },
    { icon: ShieldCheck, title: 'Authenticate', description: 'Verify identity in real-time' },
    { icon: BarChart3, title: 'Monitor', description: 'Track analytics and performance' }
  ]

  return (
    <div className="space-y-16 pb-20">
      {/* Hero Section */}
      <section className="relative text-center space-y-6 pt-10 pb-16 overflow-hidden">
        {/* Background Elements */}
        <div className="absolute inset-0 -z-10 overflow-hidden">
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.5 }}
            transition={{ duration: 1 }}
            className="absolute top-20 -left-20 w-96 h-96 bg-indigo-600 rounded-full filter blur-[100px] opacity-20"
          />
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.5 }}
            transition={{ duration: 1, delay: 0.3 }}
            className="absolute bottom-0 -right-20 w-96 h-96 bg-purple-600 rounded-full filter blur-[100px] opacity-20"
          />
        </div>
        
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="w-20 h-20 mx-auto mb-6 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center shadow-lg"
        >
          <Scan className="w-10 h-10 text-white" />
        </motion.div>
        
        <motion.h1 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-4xl md:text-6xl font-bold text-white"
        >
          Next-Gen <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-500">Facial Authentication</span>
        </motion.h1>
        
        <motion.p 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="text-xl text-gray-300 max-w-3xl mx-auto"
        >
          Enterprise-grade biometric authentication with advanced liveness detection
          and military-grade security for your applications.
        </motion.p>
        
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="flex flex-wrap justify-center gap-4 pt-6"
        >
          <Link to="/register">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-6 py-3 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-lg text-white font-medium flex items-center gap-2 shadow-lg"
            >
              <UserPlus size={20} />
              Register Now
              <motion.span
                animate={{ x: [0, 5, 0] }}
                transition={{ repeat: Infinity, duration: 1.5, ease: "easeInOut" }}
              >
                <ArrowRight size={16} />
              </motion.span>
            </motion.button>
          </Link>
          
          <Link to="/authenticate">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-6 py-3 bg-dark-800 border border-gray-700 rounded-lg text-white font-medium flex items-center gap-2 hover:bg-dark-700 transition-colors"
            >
              <ShieldCheck size={20} />
              Authenticate
            </motion.button>
          </Link>
        </motion.div>
      </section>

      {/* Stats */}
      <section className="py-16 bg-gradient-to-b from-dark-900/80 to-dark-800/30 backdrop-blur-sm border border-white/5 rounded-2xl my-16 shadow-2xl">
        <div className="text-center mb-12">
          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="text-3xl font-bold text-white mb-4"
          >
            Trusted by <span className="text-indigo-400">Organizations</span>
          </motion.h2>
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="text-gray-300 max-w-2xl mx-auto"
          >
            Our system is deployed in enterprises worldwide, securing millions of authentications daily
          </motion.p>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          {stats.map((stat, index) => (
            <motion.div 
              key={index} 
              className="text-center p-4"
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: index * 0.1 }}
            >
              <motion.p 
                className="text-4xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-purple-400 mb-2"
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: 0.2 + index * 0.1 }}
              >
                {stat.value}
              </motion.p>
              <p className="text-gray-300 font-medium">{stat.label}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="py-12">
        <div className="text-center mb-12">
          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="text-3xl font-bold text-white mb-4"
          >
            Advanced <span className="text-indigo-400">Features</span>
          </motion.h2>
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="text-gray-300 max-w-2xl mx-auto"
          >
            Our facial authentication system combines cutting-edge AI with enterprise security
          </motion.p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              whileHover={{ y: -5, transition: { duration: 0.2 } }}
              className="bg-dark-800/50 backdrop-blur-sm border border-white/10 rounded-xl p-6 shadow-xl"
            >
              <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${feature.color} flex items-center justify-center mb-4 shadow-lg`}>
                <feature.icon className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-bold text-white mb-2">{feature.title}</h3>
              <p className="text-gray-300">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* How It Works */}
      <section>
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-4xl font-bold text-white mb-4">
            How It Works
          </h2>
          <p className="text-gray-400 text-lg">
            Get started in three simple steps
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          {steps.map((step, index) => {
            const Icon = step.icon
            return (
              <motion.div
                key={step.title}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.2 * index }}
                className="relative"
              >
                <Card className="text-center h-full">
                  <div className="absolute -top-4 left-1/2 transform -translate-x-1/2 w-8 h-8 bg-primary-600 rounded-full flex items-center justify-center text-white font-bold">
                    {index + 1}
                  </div>
                  <div className="w-16 h-16 mx-auto bg-gradient-to-br from-primary-500 to-purple-600 rounded-2xl flex items-center justify-center mb-4 mt-4">
                    <Icon className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-2xl font-bold text-white mb-2">
                    {step.title}
                  </h3>
                  <p className="text-gray-400">
                    {step.description}
                  </p>
                </Card>
                {index < steps.length - 1 && (
                  <div className="hidden md:block absolute top-1/2 -right-4 transform -translate-y-1/2 z-10">
                    <ArrowRight className="w-8 h-8 text-primary-500" />
                  </div>
                )}
              </motion.div>
            )
          })}
        </div>
      </section>

      {/* Technology Stack */}
      <section>
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
        >
          <Card className="text-center">
            <h2 className="text-4xl font-bold text-white mb-6">
              Advanced AI Technology
            </h2>
            <div className="grid md:grid-cols-3 gap-8 mt-8">
              <div>
                <h3 className="text-xl font-semibold text-white mb-3">Face Detection</h3>
                <ul className="text-gray-400 space-y-2">
                  <li className="flex items-center justify-center">
                    <CheckCircle2 className="w-4 h-4 text-green-500 mr-2" />
                    RetinaFace
                  </li>
                  <li className="flex items-center justify-center">
                    <CheckCircle2 className="w-4 h-4 text-green-500 mr-2" />
                    MTCNN
                  </li>
                </ul>
              </div>
              <div>
                <h3 className="text-xl font-semibold text-white mb-3">Recognition Models</h3>
                <ul className="text-gray-400 space-y-2">
                  <li className="flex items-center justify-center">
                    <CheckCircle2 className="w-4 h-4 text-green-500 mr-2" />
                    ArcFace (ResNet100)
                  </li>
                  <li className="flex items-center justify-center">
                    <CheckCircle2 className="w-4 h-4 text-green-500 mr-2" />
                    FaceNet
                  </li>
                  <li className="flex items-center justify-center">
                    <CheckCircle2 className="w-4 h-4 text-green-500 mr-2" />
                    MobileFaceNet
                  </li>
                </ul>
              </div>
              <div>
                <h3 className="text-xl font-semibold text-white mb-3">Anti-Spoofing</h3>
                <ul className="text-gray-400 space-y-2">
                  <li className="flex items-center justify-center">
                    <CheckCircle2 className="w-4 h-4 text-green-500 mr-2" />
                    CNN Liveness
                  </li>
                  <li className="flex items-center justify-center">
                    <CheckCircle2 className="w-4 h-4 text-green-500 mr-2" />
                    Temporal LSTM
                  </li>
                  <li className="flex items-center justify-center">
                    <CheckCircle2 className="w-4 h-4 text-green-500 mr-2" />
                    Depth Estimation
                  </li>
                </ul>
              </div>
            </div>
          </Card>
        </motion.div>
      </section>

      {/* CTA */}
      <section className="py-16 text-center relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/20 to-purple-500/20 rounded-3xl blur-3xl opacity-30"></div>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="relative z-10"
        >
          <h2 className="text-4xl font-bold text-white mb-6">Ready to get started?</h2>
          <p className="text-gray-300 max-w-2xl mx-auto mb-8 text-lg">
            Join thousands of organizations that trust our facial authentication system
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.98 }}>
              <Link 
                to="/register" 
                className="px-8 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-medium rounded-lg shadow-lg shadow-indigo-500/25 hover:shadow-indigo-500/40 transition-all duration-200"
              >
                Create Account
              </Link>
            </motion.div>
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.98 }}>
              <Link 
                to="/authenticate" 
                className="px-8 py-3 bg-dark-800 text-white font-medium rounded-lg border border-white/10 hover:bg-dark-700 transition-all duration-200"
              >
                Try Demo
              </Link>
            </motion.div>
          </div>
        </motion.div>
      </section>
    </div>
  )
}

export default Home


