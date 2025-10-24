import { motion } from 'framer-motion'

const Card = ({ children, className = '', hover = true }) => {
  return (
    <motion.div
      whileHover={hover ? { y: -4, scale: 1.02 } : {}}
      transition={{ type: 'spring', stiffness: 300 }}
      className={`glass rounded-xl p-6 ${className}`}
    >
      {children}
    </motion.div>
  )
}

export default Card


