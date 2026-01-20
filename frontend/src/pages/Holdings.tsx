import { useEffect, useState } from 'react'
import { portfolioApi, HoldingsAnalysis, Portfolio } from '@/services/api'
import Card from '@/components/Card'
import MetricCard from '@/components/MetricCard'
import { PieChart as PieIcon, TrendingUp, BarChart3, Activity } from 'lucide-react'
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Legend,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts'

const ACCOUNT_ID = 'default'

export default function Holdings() {
  const [analysis, setAnalysis] = useState<HoldingsAnalysis | null>(null)
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadHoldingsAnalysis()
  }, [])

  const loadHoldingsAnalysis = async () => {
    try {
      const [analysisRes, portfolioRes] = await Promise.all([
        portfolioApi.getHoldingsAnalysis(ACCOUNT_ID),
        portfolioApi.getPortfolio(ACCOUNT_ID),
      ])

      setAnalysis(analysisRes.data)
      setPortfolio(portfolioRes.data)
    } catch (error) {
      console.error('Error loading holdings analysis:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading || !analysis || !portfolio) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-slate-400">Loading holdings analysis...</div>
      </div>
    )
  }

  const COLORS = ['#0ea5e9', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#6366f1', '#ef4444']

  // Prepare sector exposure data
  const sectorData = Object.entries(analysis.sector_exposure).map(([sector, weight]) => ({
    name: sector,
    value: weight * 100,
  }))

  // Prepare country exposure data
  const countryData = Object.entries(analysis.country_exposure).map(([country, weight]) => ({
    name: country,
    value: weight * 100,
  }))

  // Prepare industry exposure data
  const industryData = Object.entries(analysis.industry_exposure)
    .map(([industry, weight]) => ({
      industry,
      weight: weight * 100,
    }))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 10)

  // Factor scores
  const factorScores = [
    { name: 'Value', score: analysis.value_score || 0 },
    { name: 'Growth', score: analysis.growth_score || 0 },
    { name: 'Momentum', score: analysis.momentum_score || 0 },
    { name: 'Quality', score: analysis.quality_score || 0 },
  ].filter((f) => f.score !== 0)

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Holdings Analysis</h1>
        <p className="text-slate-400">Portfolio composition and exposure metrics</p>
      </div>

      {/* Concentration Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <MetricCard
          title="Top 10 Concentration"
          value={analysis.top_10_concentration}
          format="percent"
          icon={<PieIcon size={20} />}
        />
        <MetricCard
          title="Top 20 Concentration"
          value={analysis.top_20_concentration}
          format="percent"
          icon={<BarChart3 size={20} />}
        />
        <MetricCard
          title="Herfindahl Index"
          value={analysis.herfindahl_index.toFixed(4)}
          icon={<Activity size={20} />}
        />
        <MetricCard
          title="Weighted Beta"
          value={analysis.weighted_beta.toFixed(2)}
          icon={<TrendingUp size={20} />}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Sector Exposure */}
        <Card title="Sector Exposure">
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={sectorData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, value }) => `${name} ${value.toFixed(1)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {sectorData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                formatter={(value: number) => `${value.toFixed(2)}%`}
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
            </PieChart>
          </ResponsiveContainer>
        </Card>

        {/* Country Exposure */}
        <Card title="Country Exposure">
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={countryData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, value }) => `${name} ${value.toFixed(1)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {countryData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                formatter={(value: number) => `${value.toFixed(2)}%`}
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
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Industry Exposure */}
      <Card title="Top 10 Industry Exposure" className="mb-8">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={industryData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="industry" stroke="#94a3b8" angle={-15} textAnchor="end" height={100} />
            <YAxis stroke="#94a3b8" />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1e293b',
                border: '1px solid #334155',
                borderRadius: '0.5rem',
              }}
              formatter={(value: number) => `${value.toFixed(2)}%`}
            />
            <Bar dataKey="weight" fill="#0ea5e9" name="Weight %" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Portfolio Characteristics */}
        <Card title="Portfolio Characteristics">
          <div className="space-y-4">
            <div className="flex justify-between items-center pb-3 border-b border-slate-700">
              <span className="text-sm text-slate-400">Avg Market Cap</span>
              <span className="text-sm font-medium text-white">
                ${(analysis.avg_market_cap / 1e9).toFixed(2)}B
              </span>
            </div>
            <div className="flex justify-between items-center pb-3 border-b border-slate-700">
              <span className="text-sm text-slate-400">Median Market Cap</span>
              <span className="text-sm font-medium text-white">
                ${(analysis.median_market_cap / 1e9).toFixed(2)}B
              </span>
            </div>
            <div className="flex justify-between items-center pb-3 border-b border-slate-700">
              <span className="text-sm text-slate-400">Avg P/E Ratio</span>
              <span className="text-sm font-medium text-white">
                {analysis.avg_pe_ratio?.toFixed(2) || 'N/A'}
              </span>
            </div>
            <div className="flex justify-between items-center pb-3 border-b border-slate-700">
              <span className="text-sm text-slate-400">Avg P/B Ratio</span>
              <span className="text-sm font-medium text-white">
                {analysis.avg_pb_ratio?.toFixed(2) || 'N/A'}
              </span>
            </div>
            <div className="flex justify-between items-center pb-3 border-b border-slate-700">
              <span className="text-sm text-slate-400">Weighted Beta</span>
              <span className="text-sm font-medium text-white">
                {analysis.weighted_beta.toFixed(2)}
              </span>
            </div>
          </div>
        </Card>

        {/* Liquidity Metrics */}
        <Card title="Liquidity Analysis">
          <div className="space-y-4">
            <div className="flex justify-between items-center pb-3 border-b border-slate-700">
              <span className="text-sm text-slate-400">Avg Daily Volume</span>
              <span className="text-sm font-medium text-white">
                {(analysis.avg_daily_volume / 1e6).toFixed(2)}M
              </span>
            </div>
            <div className="flex justify-between items-center pb-3 border-b border-slate-700">
              <span className="text-sm text-slate-400">Liquidity Score</span>
              <span className="text-sm font-medium text-white">
                {(analysis.weighted_liquidity_score * 100).toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between items-center pb-3 border-b border-slate-700">
              <span className="text-sm text-slate-400">Illiquid Holdings %</span>
              <span className="text-sm font-medium text-white">
                {(analysis.illiquid_holdings_pct * 100).toFixed(2)}%
              </span>
            </div>
            {factorScores.length > 0 && (
              <>
                <div className="pt-4">
                  <h4 className="text-sm font-medium text-slate-400 mb-3">Factor Scores</h4>
                </div>
                {factorScores.map((factor) => (
                  <div
                    key={factor.name}
                    className="flex justify-between items-center pb-3 border-b border-slate-700"
                  >
                    <span className="text-sm text-slate-400">{factor.name} Score</span>
                    <span className="text-sm font-medium text-white">
                      {factor.score.toFixed(2)}
                    </span>
                  </div>
                ))}
              </>
            )}
          </div>
        </Card>
      </div>
    </div>
  )
}
