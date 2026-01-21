import { useEffect, useState } from 'react'
import axios from 'axios'
import Card from '@/components/Card'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import { Info, TrendingUp } from 'lucide-react'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const ACCOUNT_ID = 'default'

const PERIODS = ['1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', '10Y']
const METRIC_TYPES = [
  { key: 'simple_return', label: 'Simple Return', color: '#3b82f6' },
  { key: 'twr', label: 'TWR', color: '#8b5cf6' },
  { key: 'mwr', label: 'MWR (IRR)', color: '#10b981' },
]

interface PerformanceData {
  portfolio: Record<string, any>
  securities: Record<string, any>
  time_series: Record<string, any>
}

export default function PerformanceV2() {
  const [data, setData] = useState<PerformanceData | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedPeriod, setSelectedPeriod] = useState('1Y')
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['twr'])
  const [showTooltip, setShowTooltip] = useState<string | null>(null)

  useEffect(() => {
    loadPerformance()
  }, [])

  const loadPerformance = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/performance-v2/${ACCOUNT_ID}`)
      setData(response.data)
    } catch (error) {
      console.error('Error loading performance:', error)
    } finally {
      setLoading(false)
    }
  }

  const toggleMetric = (metricKey: string) => {
    if (selectedMetrics.includes(metricKey)) {
      if (selectedMetrics.length > 1) {
        setSelectedMetrics(selectedMetrics.filter((m) => m !== metricKey))
      }
    } else {
      setSelectedMetrics([...selectedMetrics, metricKey])
    }
  }

  const formatPercent = (value: number | null | undefined) => {
    if (value === null || value === undefined) return 'N/A'
    return `${(value * 100).toFixed(2)}%`
  }

  const formatCurrency = (value: number | null | undefined) => {
    if (value === null || value === undefined) return 'N/A'
    return `$${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  }

  if (loading || !data) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-slate-400">Loading performance data...</div>
      </div>
    )
  }

  // Prepare chart data
  const chartData = data.time_series[selectedPeriod]
    ? data.time_series[selectedPeriod].dates.map((date: string, i: number) => ({
        date,
        simple_return: data.time_series[selectedPeriod].simple_return[i] * 100,
        twr: data.time_series[selectedPeriod].twr[i] * 100,
        mwr: data.time_series[selectedPeriod].mwr[i] * 100,
      }))
    : []

  // Prepare securities table data
  const securitiesData = Object.entries(data.securities).map(([symbol, periods]: [string, any]) => ({
    symbol,
    ...periods[selectedPeriod],
  }))

  const portfolioMetrics = data.portfolio[selectedPeriod] || {}

  return (
    <div>
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Performance Analysis</h1>
        <p className="text-slate-400">Comprehensive return metrics across all time periods</p>
      </div>

      {/* Period Selector */}
      <Card title="Time Period" className="mb-6">
        <div className="flex gap-2 flex-wrap">
          {PERIODS.map((period) => (
            <button
              key={period}
              onClick={() => setSelectedPeriod(period)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                selectedPeriod === period
                  ? 'bg-primary-600 text-white'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              {period}
            </button>
          ))}
        </div>
      </Card>

      {/* Portfolio Summary */}
      <Card title="Portfolio Performance" icon={<TrendingUp size={20} />} className="mb-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-sm text-slate-400">Simple Return</span>
              <button
                onClick={() => setShowTooltip(showTooltip === 'simple' ? null : 'simple')}
                className="text-slate-500 hover:text-slate-300"
              >
                <Info size={14} />
              </button>
            </div>
            {showTooltip === 'simple' && (
              <div className="text-xs text-slate-400 bg-slate-800 p-2 rounded mb-2">
                Modified Dietz method: (Ending - Beginning - Net Flows) / (Beginning + Weighted Flows)
              </div>
            )}
            <div className="text-2xl font-bold text-white">
              {formatPercent(portfolioMetrics.simple_return)}
            </div>
            <div className="text-xs text-slate-500 mt-1">
              {formatCurrency(portfolioMetrics.start_value)} → {formatCurrency(portfolioMetrics.end_value)}
            </div>
          </div>

          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-sm text-slate-400">TWR</span>
              <button
                onClick={() => setShowTooltip(showTooltip === 'twr' ? null : 'twr')}
                className="text-slate-500 hover:text-slate-300"
              >
                <Info size={14} />
              </button>
            </div>
            {showTooltip === 'twr' && (
              <div className="text-xs text-slate-400 bg-slate-800 p-2 rounded mb-2">
                Time-Weighted Return: Chain-links sub-period returns between cash flows. Eliminates impact
                of contribution timing.
              </div>
            )}
            <div className="text-2xl font-bold text-white">{formatPercent(portfolioMetrics.twr)}</div>
          </div>

          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-sm text-slate-400">MWR (IRR)</span>
              <button
                onClick={() => setShowTooltip(showTooltip === 'mwr' ? null : 'mwr')}
                className="text-slate-500 hover:text-slate-300"
              >
                <Info size={14} />
              </button>
            </div>
            {showTooltip === 'mwr' && (
              <div className="text-xs text-slate-400 bg-slate-800 p-2 rounded mb-2">
                Money-Weighted Return / Internal Rate of Return: Accounts for timing and size of cash
                flows. Your personal return.
              </div>
            )}
            <div className="text-2xl font-bold text-white">{formatPercent(portfolioMetrics.mwr)}</div>
          </div>
        </div>
      </Card>

      {/* Chart */}
      <Card title="Return Time Series" className="mb-6">
        <div className="mb-4 flex gap-2 flex-wrap">
          {METRIC_TYPES.map((metric) => (
            <button
              key={metric.key}
              onClick={() => toggleMetric(metric.key)}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                selectedMetrics.includes(metric.key)
                  ? 'bg-slate-700 text-white'
                  : 'bg-slate-800/50 text-slate-400 hover:bg-slate-800'
              }`}
              style={{
                borderLeft: selectedMetrics.includes(metric.key) ? `3px solid ${metric.color}` : 'none',
              }}
            >
              {metric.label}
            </button>
          ))}
        </div>

        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="date" stroke="#94a3b8" tick={{ fontSize: 12 }} />
            <YAxis
              stroke="#94a3b8"
              tick={{ fontSize: 12 }}
              tickFormatter={(value) => `${value.toFixed(1)}%`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1e293b',
                border: '1px solid #334155',
                borderRadius: '0.5rem',
              }}
              formatter={(value: number) => `${value.toFixed(2)}%`}
            />
            <Legend wrapperStyle={{ color: '#cbd5e1' }} />

            {selectedMetrics.includes('simple_return') && (
              <Line
                type="monotone"
                dataKey="simple_return"
                stroke={METRIC_TYPES[0].color}
                name="Simple Return"
                strokeWidth={2}
                dot={false}
              />
            )}
            {selectedMetrics.includes('twr') && (
              <Line
                type="monotone"
                dataKey="twr"
                stroke={METRIC_TYPES[1].color}
                name="TWR"
                strokeWidth={2}
                dot={false}
              />
            )}
            {selectedMetrics.includes('mwr') && (
              <Line
                type="monotone"
                dataKey="mwr"
                stroke={METRIC_TYPES[2].color}
                name="MWR (IRR)"
                strokeWidth={2}
                dot={false}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Securities Table */}
      <Card title="Security-Level Performance" className="mb-6">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left py-3 px-2 text-sm font-medium text-slate-400">Symbol</th>
                <th className="text-right py-3 px-2 text-sm font-medium text-slate-400">
                  Simple Return
                </th>
                <th className="text-right py-3 px-2 text-sm font-medium text-slate-400">TWR</th>
                <th className="text-right py-3 px-2 text-sm font-medium text-slate-400">
                  MWR (IRR)
                </th>
                <th className="text-right py-3 px-2 text-sm font-medium text-slate-400">
                  Start Value
                </th>
                <th className="text-right py-3 px-2 text-sm font-medium text-slate-400">End Value</th>
              </tr>
            </thead>
            <tbody>
              {securitiesData.map((security) => (
                <tr key={security.symbol} className="border-b border-slate-700/50">
                  <td className="py-3 px-2 text-sm font-medium text-white">{security.symbol}</td>
                  <td
                    className={`py-3 px-2 text-sm text-right ${
                      (security.simple_return || 0) >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}
                  >
                    {formatPercent(security.simple_return)}
                  </td>
                  <td
                    className={`py-3 px-2 text-sm text-right ${
                      (security.twr || 0) >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}
                  >
                    {formatPercent(security.twr)}
                  </td>
                  <td
                    className={`py-3 px-2 text-sm text-right ${
                      (security.mwr || 0) >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}
                  >
                    {formatPercent(security.mwr)}
                  </td>
                  <td className="py-3 px-2 text-sm text-right text-slate-300">
                    {formatCurrency(security.start_value)}
                  </td>
                  <td className="py-3 px-2 text-sm text-right text-slate-300">
                    {formatCurrency(security.end_value)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Methodology Note */}
      <Card title="Methodology" className="bg-slate-800/50">
        <div className="text-sm text-slate-400 space-y-2">
          <p>
            <strong className="text-slate-300">Simple Return:</strong> Modified Dietz method accounts for
            weighted cash flows throughout the period.
          </p>
          <p>
            <strong className="text-slate-300">TWR (Time-Weighted Return):</strong> Chain-links sub-period
            returns between cash flow dates. Best for comparing to benchmarks as it eliminates the effect of
            contribution timing.
          </p>
          <p>
            <strong className="text-slate-300">MWR (Money-Weighted Return / IRR):</strong> Internal rate of
            return considering timing and magnitude of all cash flows. Represents your personal return
            including investment timing decisions.
          </p>
        </div>
      </Card>
    </div>
  )
}
