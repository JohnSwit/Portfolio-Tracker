import { ReactNode } from 'react'

interface CardProps {
  children: ReactNode
  title?: string
  className?: string
}

export default function Card({ children, title, className = '' }: CardProps) {
  return (
    <div className={`bg-slate-800 rounded-lg border border-slate-700 ${className}`}>
      {title && (
        <div className="px-6 py-4 border-b border-slate-700">
          <h3 className="text-lg font-semibold text-white">{title}</h3>
        </div>
      )}
      <div className="p-6">{children}</div>
    </div>
  )
}
