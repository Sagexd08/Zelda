import * as React from 'react'
import * as ProgressPrimitive from '@radix-ui/react-progress'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'

const Progress = React.forwardRef<
  React.ElementRef<typeof ProgressPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof ProgressPrimitive.Root> & {
    variant?: 'default' | 'success' | 'warning' | 'error'
    showLabel?: boolean
  }
>(({ className, value, variant = 'default', showLabel = false, ...props }, ref) => {
  const variantClasses = {
    default: 'bg-gradient-blue',
    success: 'bg-gradient-to-r from-green-500 to-emerald-500',
    warning: 'bg-gradient-to-r from-yellow-500 to-orange-500',
    error: 'bg-gradient-to-r from-red-500 to-pink-500',
  }

  return (
    <ProgressPrimitive.Root
      ref={ref}
      className={cn('relative h-4 w-full overflow-hidden rounded-full bg-dark-800', className)}
      {...props}
    >
      <ProgressPrimitive.Indicator
        className={cn('h-full w-full flex-1 transition-all', variantClasses[variant])}
        style={{ transform: `translateX(-${100 - (value || 0)}%)` }}
      >
        {showLabel && value !== undefined && value !== null && (
          <motion.div
            className="flex h-full items-center justify-center text-xs font-medium text-white"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            {Math.round(value)}%
          </motion.div>
        )}
      </ProgressPrimitive.Indicator>
    </ProgressPrimitive.Root>
  )
})
Progress.displayName = ProgressPrimitive.Root.displayName

export { Progress }

