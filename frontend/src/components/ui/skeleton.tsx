import * as React from 'react'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'

export interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'circular' | 'text' | 'rectangular'
  animation?: 'pulse' | 'wave' | 'none'
}

const Skeleton = React.forwardRef<HTMLDivElement, SkeletonProps>(
  ({ className, variant = 'default', animation = 'pulse', ...props }, ref) => {
    const baseClasses = 'bg-dark-800'
    
    const variantClasses = {
      default: 'rounded-lg',
      circular: 'rounded-full',
      text: 'rounded',
      rectangular: 'rounded-none',
    }

    const animationClasses = {
      pulse: 'animate-pulse',
      wave: 'animate-shimmer',
      none: '',
    }

    return (
      <motion.div
        ref={ref}
        className={cn(
          baseClasses,
          variantClasses[variant],
          animationClasses[animation],
          className
        )}
        initial={{ opacity: 0.5 }}
        animate={{ opacity: [0.5, 1, 0.5] }}
        transition={{ duration: 1.5, repeat: Infinity }}
        {...(props as any)}
      />
    )
  }
)
Skeleton.displayName = 'Skeleton'

export { Skeleton }

