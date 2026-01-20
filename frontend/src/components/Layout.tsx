import { ReactNode } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { BarChart3, TrendingUp, AlertTriangle, PieChart, Upload } from 'lucide-react'

interface LayoutProps {
  children: ReactNode
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation()

  const navigation = [
    { name: 'Dashboard', path: '/', icon: BarChart3 },
    { name: 'Performance', path: '/performance', icon: TrendingUp },
    { name: 'Risk', path: '/risk', icon: AlertTriangle },
    { name: 'Holdings', path: '/holdings', icon: PieChart },
    { name: 'Import Data', path: '/import', icon: Upload },
  ]

  return (
    <div className="min-h-screen bg-slate-900">
      {/* Sidebar */}
      <div className="fixed inset-y-0 left-0 z-50 w-64 bg-slate-800 border-r border-slate-700">
        <div className="flex h-16 items-center px-6 border-b border-slate-700">
          <h1 className="text-xl font-bold text-white">Portfolio Analytics</h1>
        </div>

        <nav className="mt-6 px-3">
          {navigation.map((item) => {
            const Icon = item.icon
            const isActive = location.pathname === item.path

            return (
              <Link
                key={item.name}
                to={item.path}
                className={`
                  flex items-center gap-3 px-3 py-2 mb-1 rounded-lg text-sm font-medium transition-colors
                  ${
                    isActive
                      ? 'bg-primary-600 text-white'
                      : 'text-slate-300 hover:bg-slate-700 hover:text-white'
                  }
                `}
              >
                <Icon size={20} />
                {item.name}
              </Link>
            )
          })}
        </nav>
      </div>

      {/* Main content */}
      <div className="pl-64">
        <main className="p-8">{children}</main>
      </div>
    </div>
  )
}
