import { motion } from 'framer-motion';

export default function LoadingScreen() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 flex items-center justify-center bg-gradient-to-br from-dark-950 via-dark-900 to-dark-950 z-50"
    >
      <div className="text-center">
        <motion.div
          animate={{
            scale: [1, 1.2, 1],
            rotate: [0, 180, 360],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut"
          }}
          className="w-24 h-24 mb-8 mx-auto"
        >
          <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
            <polygon points="50,10 90,90 10,90" fill="none" stroke="#6366F1" strokeWidth="4" />
            <circle cx="50" cy="50" r="30" fill="none" stroke="#8B5CF6" strokeWidth="4" />
          </svg>
        </motion.div>
        <motion.h1
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="text-3xl font-bold text-white"
        >
          Zelda
        </motion.h1>
        <p className="text-gray-300 mt-2">Advanced Facial Authentication</p>
      </div>
    </motion.div>
  );
}