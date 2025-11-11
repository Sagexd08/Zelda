import * as React from 'react'
import { motion } from 'framer-motion'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '../../lib/utils'

const cardVariants = cva(
  'rounded-2xl transition-all duration-300 border',
  {
    variants: {
      variant: {
        default: 'glass border-dark-600/50',
        elevated: 'glass-strong shadow-2xl border-dark-600/30 backdrop-blur-xl',
        outlined: 'border-2 border-accent/20 bg-dark-900/50 backdrop-blur-sm',
        gradient: 'bg-gradient-to-br from-accent/10 via-blue-500/10 to-accent/10 border border-accent/20 backdrop-blur-xl',
      },
      hover: {
        true: 'hover:scale-[1.02] hover:shadow-xl hover:shadow-accent/20 hover:border-accent/40 cursor-pointer transition-all duration-300',
        false: '',
      },
    },
    defaultVariants: {
      variant: 'default',
      hover: true,
    },
  }
)

export interface CardProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof cardVariants> {
  glow?: boolean
  delay?: number
}

const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, variant, hover, glow, delay = 0, children, ...props }, ref) => {
    return (
      <motion.div
        ref={ref}
        className={cn(cardVariants({ variant, hover }), glow && 'hover-glow', className)}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay, duration: 0.5 }}
        whileHover={hover ? { y: -4 } : {}}
        {...(props as any)}
      >
        {children}
      </motion.div>
    )
  }
)
Card.displayName = 'Card'

export { Card, cardVariants }

