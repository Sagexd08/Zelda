import { motion } from 'framer-motion'
import Spinner from './Spinner'

export default function LoadingScreen() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 flex items-center justify-center bg-dark-950 z-50"
    >
      <div className="text-center">
        <Spinner size="lg" className="mb-8" />
        <motion.h1
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="text-3xl font-bold gradient-text-neon mb-2"
        >
          Zelda
        </motion.h1>
        <p className="text-gray-300">Advanced Facial Authentication</p>
      </div>
    </motion.div>
  )
}

