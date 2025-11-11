import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { 
  ShieldCheck, 
  ArrowRight, 
  Sparkles,
  Zap,
  Eye,
  Lock,
  UserPlus,
  BarChart3,
  CheckCircle2,
  Scan,
  Brain,
  Cpu,
  Database,
  Shield,
  TrendingUp,
  Clock,
  Users,
  Globe,
  Code,
  Layers
} from 'lucide-react'
import Plasma from '../components/effects/Plasma'
import { Card } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '../components/ui/tooltip'

const Landing = () => {
  const features = [
    {
      icon: Shield,
      title: '99.8% Accuracy',
      description: 'State-of-the-art deep learning models with multi-model fusion',
      color: 'from-blue-500 to-cyan-500',
      delay: 0.1
    },
    {
      icon: Zap,
      title: '<100ms Response',
      description: 'Lightning-fast authentication with real-time processing',
      color: 'from-yellow-500 to-orange-500',
      delay: 0.2
    },
    {
      icon: Eye,
      title: 'Advanced Liveness',
      description: 'Multi-layer anti-spoofing with temporal and depth analysis',
      color: 'from-purple-500 to-pink-500',
      delay: 0.3
    },
    {
      icon: Lock,
      title: 'Military-Grade Security',
      description: 'AES-256 encryption and secure biometric storage',
      color: 'from-green-500 to-emerald-500',
      delay: 0.4
    },
    {
      icon: Brain,
      title: 'AI-Powered',
      description: 'Advanced ML models: ArcFace, FaceNet, MobileFaceNet',
      color: 'from-indigo-500 to-purple-500',
      delay: 0.5
    },
    {
      icon: Database,
      title: 'Enterprise Ready',
      description: 'Scalable architecture for millions of authentications',
      color: 'from-red-500 to-pink-500',
      delay: 0.6
    }
  ]

  const stats = [
    { label: 'Accuracy Rate', value: '99.8%', icon: TrendingUp },
    { label: 'Response Time', value: '<100ms', icon: Clock },
    { label: 'Active Models', value: '5+', icon: Cpu },
    { label: 'Security Level', value: 'Military', icon: ShieldCheck }
  ]

  const steps = [
    { 
      icon: UserPlus, 
      title: 'Register', 
      description: 'Capture multiple face samples with quality validation',
      color: 'from-blue-500 to-cyan-500'
    },
    { 
      icon: ShieldCheck, 
      title: 'Authenticate', 
      description: 'Real-time verification with liveness detection',
      color: 'from-purple-500 to-pink-500'
    },
    { 
      icon: BarChart3, 
      title: 'Monitor', 
      description: 'Track analytics, performance, and security metrics',
      color: 'from-green-500 to-emerald-500'
    }
  ]

  const technologies = [
    {
      category: 'Face Detection',
      items: ['RetinaFace', 'MTCNN'],
      icon: Eye
    },
    {
      category: 'Recognition Models',
      items: ['ArcFace (ResNet100)', 'FaceNet', 'MobileFaceNet'],
      icon: Brain
    },
    {
      category: 'Anti-Spoofing',
      items: ['CNN Liveness', 'Temporal LSTM', 'Depth Estimation'],
      icon: Shield
    },
    {
      category: 'Infrastructure',
      items: ['FastAPI', 'React', 'WebGL', 'Supabase'],
      icon: Layers
    }
  ]

  return (
    <div className="relative min-h-screen overflow-hidden bg-dark-950">
      {/* Plasma WebGL Background */}
      <div className="fixed inset-0 z-0">
        <Plasma
          color="#00BFFF"
          speed={0.8}
          direction="forward"
          scale={1.2}
          opacity={0.3}
          mouseInteractive={true}
        />
      </div>

      {/* Gradient Overlay */}
      <div className="fixed inset-0 z-[1] bg-gradient-to-b from-dark-950/80 via-dark-950/40 to-dark-950/80 pointer-events-none"></div>

      {/* Content */}
      <div className="relative z-10">
        {/* Hero Section */}
        <section className="min-h-screen flex items-center justify-center px-4 sm:px-6 lg:px-8 py-20 pt-40">
          <div className="max-w-7xl mx-auto text-center w-full">
            {/* Logo/Icon */}
            <motion.div
              initial={{ scale: 0, rotate: -180 }}
              animate={{ scale: 1, rotate: 0 }}
              transition={{ delay: 0.2, type: 'spring', stiffness: 200, damping: 15 }}
              className="mb-10 flex justify-center"
            >
              <motion.div
                animate={{ 
                  rotate: [0, 5, -5, 0],
                  scale: [1, 1.05, 1]
                }}
                transition={{ 
                  duration: 4,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
                className="relative w-32 h-32 sm:w-36 sm:h-36 md:w-40 md:h-40"
              >
                <div className="absolute inset-0 rounded-3xl bg-gradient-blue opacity-75 blur-2xl animate-pulse"></div>
                <div className="relative w-full h-full rounded-3xl bg-gradient-to-br from-blue-500 via-cyan-500 to-blue-600 flex items-center justify-center shadow-2xl border-2 border-cyan-400/30">
                  <ShieldCheck className="w-16 h-16 sm:w-20 sm:h-20 text-white z-10 drop-shadow-lg" />
                  <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-white/20 to-transparent"></div>
                </div>
                {/* Orbiting particles */}
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                  className="absolute inset-0"
                >
                  <div className="absolute top-0 left-1/2 w-2 h-2 bg-cyan-400 rounded-full blur-sm"></div>
                  <div className="absolute bottom-0 left-1/2 w-2 h-2 bg-blue-400 rounded-full blur-sm"></div>
                  <div className="absolute left-0 top-1/2 w-2 h-2 bg-blue-300 rounded-full blur-sm"></div>
                  <div className="absolute right-0 top-1/2 w-2 h-2 bg-cyan-300 rounded-full blur-sm"></div>
                </motion.div>
              </motion.div>
            </motion.div>

            {/* Main Heading */}
            <motion.h1
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.8 }}
              className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl xl:text-8xl font-bold mb-8 gradient-text-neon leading-tight tracking-tight"
            >
              <span className="block mb-2">Enterprise Facial</span>
              <span className="block text-white drop-shadow-2xl">Authentication</span>
            </motion.h1>

            {/* Subheading */}
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5, duration: 0.8 }}
              className="text-lg sm:text-xl md:text-2xl lg:text-3xl text-gray-300 mb-14 max-w-4xl mx-auto leading-relaxed px-4"
            >
              Secure, real-time facial recognition powered by advanced AI/ML technologies.
              <br className="hidden sm:block" />
              <span className="text-accent font-semibold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">Military-grade encryption</span> meets cutting-edge design.
            </motion.p>

            {/* CTA Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7, duration: 0.8 }}
              className="flex flex-col sm:flex-row gap-4 sm:gap-6 justify-center items-center mb-20 px-4"
            >
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Link to="/authenticate">
                      <Button
                        size="lg"
                        variant="gradient"
                        icon={<ArrowRight className="w-6 h-6" />}
                        iconPosition="right"
                        className="text-lg px-10 py-5"
                      >
                        Authenticate Now
                      </Button>
                    </Link>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Start authenticating users instantly</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>

              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Link to="/register">
                      <Button
                        size="lg"
                        variant="outline"
                        className="text-lg px-10 py-5"
                      >
                        Register User
                      </Button>
                    </Link>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Add new users to the system</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </motion.div>

            {/* Stats Bar */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.9, duration: 0.8 }}
              className="grid grid-cols-2 md:grid-cols-4 gap-4 sm:gap-6 max-w-5xl mx-auto px-4"
            >
              <TooltipProvider>
                {stats.map((stat, index) => {
                  const Icon = stat.icon
                  return (
                    <Tooltip key={stat.label}>
                      <TooltipTrigger asChild>
                        <motion.div
                          initial={{ opacity: 0, scale: 0.8 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: 1 + index * 0.1, type: 'spring' }}
                          whileHover={{ scale: 1.05, y: -5 }}
                          className="w-full"
                        >
                          <Card variant="elevated" hover glow className="p-4 sm:p-6 text-center h-full min-h-[140px] sm:min-h-[160px] flex flex-col justify-center">
                            <Icon className="w-7 h-7 sm:w-8 sm:h-8 text-accent mb-3 mx-auto" />
                            <div className="text-2xl sm:text-3xl md:text-4xl font-bold gradient-text-neon mb-2">
                              {stat.value}
                            </div>
                            <div className="text-xs sm:text-sm text-gray-400">{stat.label}</div>
                          </Card>
                        </motion.div>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Industry-leading {stat.label.toLowerCase()}</p>
                      </TooltipContent>
                    </Tooltip>
                  )
                })}
              </TooltipProvider>
            </motion.div>
          </div>
        </section>

        {/* Features Section */}
        <section className="py-16 sm:py-20 lg:py-24 px-4 sm:px-6 lg:px-8">
          <div className="max-w-7xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              className="text-center mb-12 sm:mb-16"
            >
              <h2 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-4 px-4">
                Advanced <span className="gradient-text-neon">Features</span>
              </h2>
              <p className="text-lg sm:text-xl text-gray-400 max-w-3xl mx-auto px-4">
                Our facial authentication system combines cutting-edge AI with enterprise security
              </p>
            </motion.div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8">
              <TooltipProvider>
                {features.map((feature, index) => {
                  const Icon = feature.icon
                  return (
                    <Tooltip key={feature.title}>
                      <TooltipTrigger asChild>
                        <motion.div
                          initial={{ opacity: 0, y: 30 }}
                          whileInView={{ opacity: 1, y: 0 }}
                          viewport={{ once: true }}
                          transition={{ delay: feature.delay, duration: 0.6 }}
                          whileHover={{ y: -10, scale: 1.02 }}
                          className="h-full"
                        >
                          <Card variant="elevated" hover glow delay={feature.delay} className="p-6 sm:p-8 h-full flex flex-col">
                            <div className={`w-14 h-14 sm:w-16 sm:h-16 rounded-2xl bg-gradient-to-br ${feature.color} flex items-center justify-center mb-5 sm:mb-6 shadow-lg group-hover:scale-110 transition-transform`}>
                              <Icon className="w-7 h-7 sm:w-8 sm:h-8 text-white" />
                            </div>
                            <h3 className="text-xl sm:text-2xl font-bold text-white mb-3">{feature.title}</h3>
                            <p className="text-gray-400 leading-relaxed text-sm sm:text-base flex-grow">{feature.description}</p>
                            <div className="mt-4 sm:mt-6">
                              <Badge variant="secondary" pulse>{feature.title.split(' ')[0]}</Badge>
                            </div>
                          </Card>
                        </motion.div>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>{feature.description}</p>
                      </TooltipContent>
                    </Tooltip>
                  )
                })}
              </TooltipProvider>
            </div>
          </div>
        </section>

        {/* How It Works */}
        <section className="py-16 sm:py-20 lg:py-24 px-4 sm:px-6 lg:px-8 bg-dark-900/50 backdrop-blur-sm">
          <div className="max-w-6xl mx-auto">
            <motion.div
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              className="text-center mb-12 sm:mb-16"
            >
              <h2 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-4 px-4">
                How It <span className="gradient-text-neon">Works</span>
              </h2>
              <p className="text-lg sm:text-xl text-gray-400 px-4">
                Get started in three simple steps
              </p>
            </motion.div>

            <div className="grid sm:grid-cols-1 md:grid-cols-3 gap-6 sm:gap-8 relative">
              {steps.map((step, index) => {
                const Icon = step.icon
                return (
                  <motion.div
                    key={step.title}
                    initial={{ opacity: 0, x: -30 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: index * 0.2, duration: 0.6 }}
                    className="relative"
                  >
                    <Card variant="elevated" hover glow className="text-center h-full p-6 sm:p-8 relative min-h-[280px] sm:min-h-[320px]">
                      <div className={`absolute -top-6 left-1/2 transform -translate-x-1/2 w-12 h-12 sm:w-14 sm:h-14 bg-gradient-to-br ${step.color} rounded-full flex items-center justify-center text-white font-bold text-lg sm:text-xl shadow-lg border-2 border-white/20`}>
                        {index + 1}
                      </div>
                      <div className={`w-16 h-16 sm:w-20 sm:h-20 mx-auto bg-gradient-to-br ${step.color} rounded-2xl flex items-center justify-center mb-5 sm:mb-6 mt-6 sm:mt-4 shadow-lg`}>
                        <Icon className="w-8 h-8 sm:w-10 sm:h-10 text-white" />
                      </div>
                      <h3 className="text-xl sm:text-2xl font-bold text-white mb-3">
                        {step.title}
                      </h3>
                      <p className="text-gray-400 leading-relaxed text-sm sm:text-base mb-4">
                        {step.description}
                      </p>
                      <div className="mt-auto">
                        <Badge variant="info">{step.title}</Badge>
                      </div>
                    </Card>
                    {index < steps.length - 1 && (
                      <div className="hidden md:block absolute top-1/2 -right-4 transform -translate-y-1/2 z-10">
                        <motion.div
                          animate={{ x: [0, 10, 0] }}
                          transition={{ repeat: Infinity, duration: 2 }}
                        >
                          <ArrowRight className="w-10 h-10 text-accent" />
                        </motion.div>
                      </div>
                    )}
                  </motion.div>
                )
              })}
            </div>
          </div>
        </section>

        {/* Technology Stack */}
        <section className="py-16 sm:py-20 lg:py-24 px-4 sm:px-6 lg:px-8">
          <div className="max-w-6xl mx-auto">
            <motion.div
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              className="text-center mb-12 sm:mb-16"
            >
              <h2 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-4 px-4">
                Advanced <span className="gradient-text-neon">Technology</span>
              </h2>
              <p className="text-lg sm:text-xl text-gray-400 px-4">
                Powered by state-of-the-art AI and ML models
              </p>
            </motion.div>

            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
              {technologies.map((tech, index) => {
                const Icon = tech.icon
                return (
                  <motion.div
                    key={tech.category}
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: index * 0.1 }}
                    whileHover={{ scale: 1.05 }}
                  >
                    <Card variant="elevated" hover glow className="p-5 sm:p-6 h-full min-h-[200px]">
                      <Icon className="w-9 h-9 sm:w-10 sm:h-10 text-accent mb-4" />
                      <h3 className="text-lg sm:text-xl font-bold text-white mb-4">{tech.category}</h3>
                      <ul className="space-y-2">
                        {tech.items.map((item) => (
                          <li key={item} className="flex items-start text-gray-400">
                            <CheckCircle2 className="w-4 h-4 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
                            <span className="text-sm leading-relaxed">{item}</span>
                          </li>
                        ))}
                      </ul>
                    </Card>
                  </motion.div>
                )
              })}
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="py-16 sm:py-20 lg:py-24 px-4 sm:px-6 lg:px-8">
          <div className="max-w-5xl mx-auto">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              className="relative overflow-hidden rounded-3xl"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-accent/20 via-blue-500/20 to-accent/20 blur-3xl"></div>
              <div className="relative glass-strong rounded-3xl p-8 sm:p-12 md:p-16 lg:p-20 text-center border border-cyan-400/20">
                <motion.div
                  animate={{ rotate: [0, 5, -5, 0] }}
                  transition={{ duration: 4, repeat: Infinity }}
                  className="w-20 h-20 mx-auto mb-6 bg-gradient-blue rounded-2xl flex items-center justify-center"
                >
                  <Sparkles className="w-10 h-10 text-white" />
                </motion.div>
                <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
                  Ready to get started?
                </h2>
                <p className="text-xl text-gray-300 max-w-2xl mx-auto mb-10">
                  Join organizations worldwide that trust our facial authentication system
                </p>
                <div className="flex flex-col sm:flex-row gap-6 justify-center">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Link to="/register">
                          <Button
                            size="lg"
                            variant="gradient"
                            icon={<UserPlus className="w-6 h-6" />}
                            className="text-lg px-10 py-5"
                          >
                            Create Account
                          </Button>
                        </Link>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Start using our facial authentication system</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Link to="/authenticate">
                          <Button
                            size="lg"
                            variant="outline"
                            icon={<ShieldCheck className="w-6 h-6" />}
                            className="text-lg px-10 py-5"
                          >
                            Try Demo
                          </Button>
                        </Link>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Experience the authentication flow</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
              </div>
            </motion.div>
          </div>
        </section>
      </div>
    </div>
  )
}

export default Landing
