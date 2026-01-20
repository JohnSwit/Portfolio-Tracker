import { useEffect, useState } from 'react'
import { portfolioApi, Portfolio } from '@/services/api'
import MetricCard from '@/components/MetricCard'
import Card from '@/components/Card'
import { DollarSign, TrendingUp, PieChart, Activity } from 'lucide-react'
import { PieChart as RechartsPie, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts'

const ACCOUNT_ID = 'default'

export default function Dashboard() {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadPortfolio()
  }, [])

  const loadPortfolio = async () => {
    try {
      const response = await portfolioApi.getPortfolio(ACCOUNT_ID)
      setPortfolio(response.data)
    } catch (error) {
      console.error('Error loading portfolio:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-slate-400">Loading portfolio...</div>
      </div>
    )
  }

  if (!portfolio) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-slate-400">No portfolio data available</div>
      </div>
    )
  }

  // Calculate metrics
  const totalValue = portfolio.total_value || 0
  const totalCost = portfolio.holdings.reduce((sum, h) => sum + h.cost_basis, 0)
  const totalGain = totalValue - totalCost
  const totalGainPct = totalCost > 0 ? (totalGain / totalCost) * 100 : 0

  // Sector allocation
  const sectorData: Record<string, number> = {}
  portfolio.holdings.forEach((h) => {
    const sector = h.sector || 'Unknown'
    sectorData[sector] = (sectorData[sector] || 0) + (h.market_value || 0)
  })

  const sectorChartData = Object.entries(sectorData).map(([sector, value]) => ({
    name: sector,
    value,
  }))

  const COLORS = ['#0ea5e9', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#6366f1']

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Portfolio Overview</h1>
        <p className="text-slate-400">
          {portfolio.account_name || `Account ${portfolio.account_id}`}
        </p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <MetricCard
          title="Total Value"
          value={totalValue}
          format="currency"
          icon={<DollarSign size={20} />}
        />
        <MetricCard
          title="Total Gain/Loss"
          value={totalGain}
          change={totalGainPct}
          format="currency"
          icon={<TrendingUp size={20} />}
        />
        <MetricCard
          title="Number of Holdings"
          value={portfolio.holdings.length}
          icon={<PieChart size={20} />}
        />
        <MetricCard
          title="Cash Balance"
          value={portfolio.cash_balance}
          format="currency"
          icon={<Activity size={20} />}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Holdings Table */}
        <Card title="Top Holdings">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-slate-700">
                  <th className="text-left py-3 px-2 text-sm font-medium text-slate-400">
                    Symbol
                  </th>
                  <th className="text-right py-3 px-2 text-sm font-medium text-slate-400">
                    Value
                  </th>
                  <th className="text-right py-3 px-2 text-sm font-medium text-slate-400">
                    Gain/Loss
                  </th>
                  <th className="text-right py-3 px-2 text-sm font-medium text-slate-400">
                    Weight
                  </th>
                </tr>
              </thead>
              <tbody>
                {portfolio.holdings
                  .sort((a, b) => (b.market_value || 0) - (a.market_value || 0))
                  .slice(0, 10)
                  .map((holding) => {
                    const gain = (holding.market_value || 0) - holding.cost_basis
                    const gainPct = (gain / holding.cost_basis) * 100
                    const weight = ((holding.market_value || 0) / totalValue) * 100

                    return (
                      <tr key={holding.symbol} className="border-b border-slate-700/50">
                        <td className="py-3 px-2 text-sm font-medium text-white">
                          {holding.symbol}
                        </td>
                        <td className="py-3 px-2 text-sm text-right text-slate-300">
                          ${(holding.market_value || 0).toLocaleString()}
                        </td>
                        <td
                          className={`py-3 px-2 text-sm text-right ${
                            gain >= 0 ? 'text-green-400' : 'text-red-400'
                          }`}
                        >
                          {gain >= 0 ? '+' : ''}
                          {gainPct.toFixed(2)}%
                        </td>
                        <td className="py-3 px-2 text-sm text-right text-slate-300">
                          {weight.toFixed(1)}%
                        </td>
                      </tr>
                    )
                  })}
              </tbody>
            </table>
          </div>
        </Card>

        {/* Sector Allocation */}
        <Card title="Sector Allocation">
          <ResponsiveContainer width="100%" height={300}>
            <RechartsPie>
              <Pie
                data={sectorChartData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {sectorChartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                formatter={(value: number) => `$${value.toLocaleString()}`}
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '0.5rem',
                }}
              />
              <Legend
                verticalAlign="bottom"
                height={36}
                iconType="circle"
                wrapperStyle={{ color: '#cbd5e1' }}
              />
            </RechartsPie>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  )
}
