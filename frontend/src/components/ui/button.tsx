import * as React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'
import { Loader2 } from 'lucide-react'

const buttonVariants = cva(
  'inline-flex items-center justify-center whitespace-nowrap rounded-xl text-sm font-semibold ring-offset-background transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 relative overflow-hidden group',
  {
    variants: {
      variant: {
        default: 'bg-gradient-blue text-white hover:shadow-lg hover:shadow-accent/50 hover:scale-105',
        destructive: 'bg-red-600 text-white hover:bg-red-700 hover:shadow-lg hover:shadow-red-500/50',
        outline: 'border-2 border-accent/30 bg-transparent text-white hover:border-accent/60 hover:bg-accent/10',
        secondary: 'glass text-white hover:bg-dark-800',
        ghost: 'hover:bg-accent/10 text-gray-400 hover:text-white',
        link: 'text-accent underline-offset-4 hover:underline',
        gradient: 'bg-gradient-to-r from-blue-500 via-cyan-500 to-blue-600 text-white hover:shadow-2xl hover:shadow-cyan-500/50',
        neon: 'bg-transparent border-2 border-accent text-accent hover:bg-accent hover:text-white neon-border',
        success: 'bg-green-600 text-white hover:bg-green-700 hover:shadow-lg hover:shadow-green-500/50',
      },
      size: {
        default: 'h-11 px-6 py-2',
        sm: 'h-9 px-4 text-xs',
        lg: 'h-14 px-10 text-lg',
        icon: 'h-10 w-10',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  loading?: boolean
  icon?: React.ReactNode
  iconPosition?: 'left' | 'right'
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, loading, icon, iconPosition = 'left', children, disabled, ...props }, ref) => {
    return (
      <motion.button
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        disabled={disabled || loading}
        whileHover={{ scale: disabled || loading ? 1 : 1.02 }}
        whileTap={{ scale: disabled || loading ? 1 : 0.98 }}
        {...(props as any)}
      >
        {loading && (
          <Loader2 className="mr-2 h-4 w-4 animate-spin absolute" />
        )}
        <span className={cn('flex items-center gap-2', loading && 'opacity-0')}>
          {icon && iconPosition === 'left' && !loading && icon}
          {children}
          {icon && iconPosition === 'right' && !loading && icon}
        </span>
        <span className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000"></span>
      </motion.button>
    )
  }
)
Button.displayName = 'Button'

export { Button, buttonVariants }

