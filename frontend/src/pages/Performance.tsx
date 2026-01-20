import { useEffect, useState } from 'react'
import { portfolioApi, PerformanceMetrics, AttributionResult } from '@/services/api'
import Card from '@/components/Card'
import MetricCard from '@/components/MetricCard'
import { TrendingUp, Target, Activity } from 'lucide-react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'

const ACCOUNT_ID = 'default'

export default function Performance() {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null)
  const [attribution, setAttribution] = useState<AttributionResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [period, setPeriod] = useState('1Y')

  useEffect(() => {
    loadPerformance()
  }, [period])

  const loadPerformance = async () => {
    try {
      const endDate = new Date()
      const startDate = new Date()

      switch (period) {
        case '1M':
          startDate.setMonth(startDate.getMonth() - 1)
          break
        case '3M':
          startDate.setMonth(startDate.getMonth() - 3)
          break
        case '6M':
          startDate.setMonth(startDate.getMonth() - 6)
          break
        case '1Y':
          startDate.setFullYear(startDate.getFullYear() - 1)
          break
        case 'YTD':
          startDate.setMonth(0)
          startDate.setDate(1)
          break
      }

      const dateRange = {
        start_date: startDate.toISOString(),
        end_date: endDate.toISOString(),
        benchmark: '^GSPC',
      }

      const [performanceRes, attributionRes] = await Promise.all([
        portfolioApi.getPerformance(ACCOUNT_ID, dateRange),
        portfolioApi.getAttribution(ACCOUNT_ID, dateRange),
      ])

      setMetrics(performanceRes.data)
      setAttribution(attributionRes.data)
    } catch (error) {
      console.error('Error loading performance:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading || !metrics || !attribution) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-slate-400">Loading performance data...</div>
      </div>
    )
  }

  // Prepare sector attribution data
  const sectorAttributionData = Object.entries(attribution.sector_attribution).map(
    ([sector, data]) => ({
      sector,
      contribution: (data.contribution || 0) * 100,
      return: (data.return || 0) * 100,
    })
  )

  return (
    <div>
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Performance Analysis</h1>
          <p className="text-slate-400">{metrics.period}</p>
        </div>

        {/* Period Selector */}
        <div className="flex gap-2">
          {['1M', '3M', '6M', '1Y', 'YTD'].map((p) => (
            <button
              key={p}
              onClick={() => setPeriod(p)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                period === p
                  ? 'bg-primary-600 text-white'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <MetricCard
          title="Time-Weighted Return"
          value={metrics.twr}
          format="percent"
          icon={<TrendingUp size={20} />}
        />
        <MetricCard
          title="Money-Weighted Return (IRR)"
          value={metrics.mwr}
          format="percent"
          icon={<Activity size={20} />}
        />
        <MetricCard
          title="Benchmark Return"
          value={metrics.benchmark_return || 0}
          format="percent"
          icon={<Target size={20} />}
        />
        <MetricCard
          title="Active Return"
          value={metrics.active_return || 0}
          format="percent"
          change={(metrics.active_return || 0) * 100}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <Card title="Risk-Adjusted Returns" className="lg:col-span-1">
          <div className="space-y-4">
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm text-slate-400">Annualized Return</span>
                <span className="text-sm font-medium text-white">
                  {((metrics.annualized_return || 0) * 100).toFixed(2)}%
                </span>
              </div>
            </div>
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm text-slate-400">Tracking Error</span>
                <span className="text-sm font-medium text-white">
                  {((metrics.tracking_error || 0) * 100).toFixed(2)}%
                </span>
              </div>
            </div>
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm text-slate-400">Information Ratio</span>
                <span className="text-sm font-medium text-white">
                  {(metrics.information_ratio || 0).toFixed(2)}
                </span>
              </div>
            </div>
          </div>
        </Card>

        {/* Sector Attribution */}
        <Card title="Sector Attribution" className="lg:col-span-2">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={sectorAttributionData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="sector" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '0.5rem',
                }}
                formatter={(value: number) => `${value.toFixed(2)}%`}
              />
              <Legend wrapperStyle={{ color: '#cbd5e1' }} />
              <Bar dataKey="contribution" fill="#0ea5e9" name="Contribution %" />
              <Bar dataKey="return" fill="#8b5cf6" name="Return %" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Stock Attribution */}
      <Card title="Top Contributors">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left py-3 px-2 text-sm font-medium text-slate-400">Stock</th>
                <th className="text-right py-3 px-2 text-sm font-medium text-slate-400">
                  Return
                </th>
                <th className="text-right py-3 px-2 text-sm font-medium text-slate-400">
                  Weight
                </th>
                <th className="text-right py-3 px-2 text-sm font-medium text-slate-400">
                  Contribution
                </th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(attribution.stock_attribution)
                .sort(([, a], [, b]) => (b.contribution || 0) - (a.contribution || 0))
                .slice(0, 10)
                .map(([symbol, data]) => (
                  <tr key={symbol} className="border-b border-slate-700/50">
                    <td className="py-3 px-2 text-sm font-medium text-white">{symbol}</td>
                    <td
                      className={`py-3 px-2 text-sm text-right ${
                        (data.return || 0) >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}
                    >
                      {((data.return || 0) * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-2 text-sm text-right text-slate-300">
                      {((data.weight || 0) * 100).toFixed(2)}%
                    </td>
                    <td
                      className={`py-3 px-2 text-sm text-right ${
                        (data.contribution || 0) >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}
                    >
                      {((data.contribution || 0) * 100).toFixed(2)}%
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}
