import { ReactNode } from 'react'
import { TrendingUp, TrendingDown } from 'lucide-react'

interface MetricCardProps {
  title: string
  value: string | number
  change?: number
  icon?: ReactNode
  format?: 'number' | 'currency' | 'percent'
}

export default function MetricCard({
  title,
  value,
  change,
  icon,
  format = 'number',
}: MetricCardProps) {
  const formatValue = (val: string | number) => {
    if (typeof val === 'string') return val

    switch (format) {
      case 'currency':
        return new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: 'USD',
          minimumFractionDigits: 0,
          maximumFractionDigits: 0,
        }).format(val)
      case 'percent':
        return `${(val * 100).toFixed(2)}%`
      default:
        return val.toLocaleString()
    }
  }

  const isPositive = change !== undefined && change >= 0

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
      <div className="flex items-center justify-between mb-2">
        <p className="text-sm font-medium text-slate-400">{title}</p>
        {icon && <div className="text-slate-400">{icon}</div>}
      </div>

      <div className="flex items-baseline justify-between">
        <p className="text-2xl font-bold text-white">{formatValue(value)}</p>

        {change !== undefined && (
          <div
            className={`flex items-center text-sm font-medium ${
              isPositive ? 'text-green-400' : 'text-red-400'
            }`}
          >
            {isPositive ? (
              <TrendingUp size={16} className="mr-1" />
            ) : (
              <TrendingDown size={16} className="mr-1" />
            )}
            {Math.abs(change).toFixed(2)}%
          </div>
        )}
      </div>
    </div>
  )
}
