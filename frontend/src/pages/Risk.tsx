import { useEffect, useState } from 'react'
import { portfolioApi, RiskMetrics, StressTestResult } from '@/services/api'
import Card from '@/components/Card'
import MetricCard from '@/components/MetricCard'
import { AlertTriangle, TrendingDown, Activity, Target } from 'lucide-react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts'

const ACCOUNT_ID = 'default'

export default function Risk() {
  const [metrics, setMetrics] = useState<RiskMetrics | null>(null)
  const [stressTests, setStressTests] = useState<StressTestResult[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadRiskAnalytics()
  }, [])

  const loadRiskAnalytics = async () => {
    try {
      const endDate = new Date()
      const startDate = new Date()
      startDate.setFullYear(startDate.getFullYear() - 1)

      const dateRange = {
        start_date: startDate.toISOString(),
        end_date: endDate.toISOString(),
        benchmark: '^GSPC',
      }

      const [riskRes, stressRes] = await Promise.all([
        portfolioApi.getRiskMetrics(ACCOUNT_ID, dateRange),
        portfolioApi.runStressTests(ACCOUNT_ID),
      ])

      setMetrics(riskRes.data)
      setStressTests(stressRes.data)
    } catch (error) {
      console.error('Error loading risk analytics:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading || !metrics) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-slate-400">Loading risk analytics...</div>
      </div>
    )
  }

  // Prepare factor exposure data for radar chart
  const factorData = Object.entries(metrics.factor_exposures).map(([factor, value]) => ({
    factor,
    exposure: value,
  }))

  // Prepare stress test data
  const stressTestData = stressTests.map((test) => ({
    scenario: test.scenario,
    loss: Math.abs(test.estimated_loss_pct) * 100,
  }))

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Risk Analytics</h1>
        <p className="text-slate-400">{metrics.period}</p>
      </div>

      {/* Risk Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <MetricCard
          title="Annual Volatility"
          value={metrics.annual_volatility}
          format="percent"
          icon={<Activity size={20} />}
        />
        <MetricCard
          title="Sharpe Ratio"
          value={metrics.sharpe_ratio.toFixed(2)}
          icon={<Target size={20} />}
        />
        <MetricCard
          title="Max Drawdown"
          value={metrics.max_drawdown}
          format="percent"
          icon={<TrendingDown size={20} />}
        />
        <MetricCard
          title="95% VaR"
          value={metrics.var_95}
          format="percent"
          icon={<AlertTriangle size={20} />}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Risk Statistics */}
        <Card title="Risk Statistics">
          <div className="space-y-4">
            <div className="flex justify-between items-center pb-3 border-b border-slate-700">
              <span className="text-sm text-slate-400">Beta</span>
              <span className="text-sm font-medium text-white">{metrics.beta.toFixed(2)}</span>
            </div>
            <div className="flex justify-between items-center pb-3 border-b border-slate-700">
              <span className="text-sm text-slate-400">Alpha</span>
              <span className="text-sm font-medium text-white">
                {((metrics.alpha || 0) * 100).toFixed(2)}%
              </span>
            </div>
            <div className="flex justify-between items-center pb-3 border-b border-slate-700">
              <span className="text-sm text-slate-400">Correlation</span>
              <span className="text-sm font-medium text-white">
                {(metrics.correlation || 0).toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between items-center pb-3 border-b border-slate-700">
              <span className="text-sm text-slate-400">Sortino Ratio</span>
              <span className="text-sm font-medium text-white">
                {metrics.sortino_ratio.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between items-center pb-3 border-b border-slate-700">
              <span className="text-sm text-slate-400">Calmar Ratio</span>
              <span className="text-sm font-medium text-white">
                {(metrics.calmar_ratio || 0).toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between items-center pb-3 border-b border-slate-700">
              <span className="text-sm text-slate-400">Downside Volatility</span>
              <span className="text-sm font-medium text-white">
                {(metrics.downside_volatility * 100).toFixed(2)}%
              </span>
            </div>
            <div className="flex justify-between items-center pb-3 border-b border-slate-700">
              <span className="text-sm text-slate-400">99% VaR</span>
              <span className="text-sm font-medium text-white">
                {(metrics.var_99 * 100).toFixed(2)}%
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-slate-400">95% CVaR</span>
              <span className="text-sm font-medium text-white">
                {(metrics.cvar_95 * 100).toFixed(2)}%
              </span>
            </div>
          </div>
        </Card>

        {/* Factor Exposures */}
        <Card title="Factor Exposures">
          {factorData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={factorData}>
                <PolarGrid stroke="#334155" />
                <PolarAngleAxis dataKey="factor" stroke="#94a3b8" />
                <PolarRadiusAxis stroke="#94a3b8" />
                <Radar
                  name="Exposure"
                  dataKey="exposure"
                  stroke="#0ea5e9"
                  fill="#0ea5e9"
                  fillOpacity={0.6}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: '1px solid #334155',
                    borderRadius: '0.5rem',
                  }}
                />
              </RadarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-64 text-slate-400">
              No factor exposure data available
            </div>
          )}
        </Card>
      </div>

      {/* Stress Tests */}
      <Card title="Stress Test Scenarios">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={stressTestData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="scenario" stroke="#94a3b8" angle={-15} textAnchor="end" height={100} />
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
            <Bar dataKey="loss" fill="#ef4444" name="Estimated Loss %" />
          </BarChart>
        </ResponsiveContainer>

        <div className="mt-6 overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left py-3 px-2 text-sm font-medium text-slate-400">
                  Scenario
                </th>
                <th className="text-right py-3 px-2 text-sm font-medium text-slate-400">
                  Est. Loss
                </th>
                <th className="text-right py-3 px-2 text-sm font-medium text-slate-400">
                  Loss %
                </th>
                <th className="text-right py-3 px-2 text-sm font-medium text-slate-400">
                  Portfolio After
                </th>
                <th className="text-center py-3 px-2 text-sm font-medium text-slate-400">
                  VaR Breach
                </th>
              </tr>
            </thead>
            <tbody>
              {stressTests.map((test, index) => (
                <tr key={index} className="border-b border-slate-700/50">
                  <td className="py-3 px-2 text-sm font-medium text-white">{test.scenario}</td>
                  <td className="py-3 px-2 text-sm text-right text-red-400">
                    ${test.estimated_loss.toLocaleString()}
                  </td>
                  <td className="py-3 px-2 text-sm text-right text-red-400">
                    {(test.estimated_loss_pct * 100).toFixed(2)}%
                  </td>
                  <td className="py-3 px-2 text-sm text-right text-slate-300">
                    ${test.portfolio_value_after.toLocaleString()}
                  </td>
                  <td className="py-3 px-2 text-center">
                    {test.var_breach ? (
                      <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-500/10 text-red-400">
                        Yes
                      </span>
                    ) : (
                      <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-500/10 text-green-400">
                        No
                      </span>
                    )}
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
