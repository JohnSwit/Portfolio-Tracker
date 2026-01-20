import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export interface Portfolio {
  account_id: string
  account_name?: string
  holdings: Holding[]
  transactions: Transaction[]
  cash_balance: number
  total_value?: number
  inception_date?: string
  last_updated?: string
}

export interface Holding {
  symbol: string
  quantity: number
  cost_basis: number
  current_price?: number
  market_value?: number
  sector?: string
  industry?: string
  country?: string
  asset_class: string
  market_cap?: number
  pe_ratio?: number
  pb_ratio?: number
  dividend_yield?: number
  beta?: number
  avg_daily_volume?: number
}

export interface Transaction {
  date: string
  symbol: string
  transaction_type: string
  quantity: number
  price: number
  amount: number
  fees: number
  notes?: string
}

export interface PerformanceMetrics {
  period: string
  start_date: string
  end_date: string
  twr: number
  mwr: number
  total_return: number
  annualized_return?: number
  benchmark_return?: number
  active_return?: number
  tracking_error?: number
  information_ratio?: number
}

export interface AttributionResult {
  period: string
  total_return: number
  sector_attribution: Record<string, Record<string, number>>
  country_attribution: Record<string, Record<string, number>>
  stock_attribution: Record<string, Record<string, number>>
  factor_attribution: Record<string, number>
}

export interface RiskMetrics {
  period: string
  daily_volatility: number
  annual_volatility: number
  downside_volatility: number
  sharpe_ratio: number
  sortino_ratio: number
  calmar_ratio?: number
  beta: number
  alpha?: number
  correlation?: number
  max_drawdown: number
  max_drawdown_duration?: number
  current_drawdown: number
  var_95: number
  var_99: number
  cvar_95: number
  factor_exposures: Record<string, number>
}

export interface HoldingsAnalysis {
  top_10_concentration: number
  top_20_concentration: number
  herfindahl_index: number
  active_share?: number
  sector_exposure: Record<string, number>
  country_exposure: Record<string, number>
  industry_exposure: Record<string, number>
  asset_class_exposure: Record<string, number>
  avg_daily_volume: number
  weighted_liquidity_score: number
  illiquid_holdings_pct: number
  avg_market_cap: number
  median_market_cap: number
  avg_pe_ratio?: number
  avg_pb_ratio?: number
  weighted_beta: number
  value_score?: number
  growth_score?: number
  momentum_score?: number
  quality_score?: number
}

export interface StressTestResult {
  scenario: string
  estimated_loss: number
  estimated_loss_pct: number
  portfolio_value_after: number
  var_breach: boolean
}

export interface DateRange {
  start_date: string
  end_date?: string
  benchmark?: string
}

// API functions
export const portfolioApi = {
  getPortfolio: (accountId: string) =>
    api.get<Portfolio>(`/api/portfolio/${accountId}`),

  getTransactions: (accountId: string, days: number = 365) =>
    api.get(`/api/transactions/${accountId}`, { params: { days } }),

  getPerformance: (accountId: string, dateRange: DateRange) =>
    api.post<PerformanceMetrics>(`/api/performance/${accountId}`, dateRange),

  getAttribution: (accountId: string, dateRange: DateRange) =>
    api.post<AttributionResult>(`/api/attribution/${accountId}`, dateRange),

  getRiskMetrics: (accountId: string, dateRange: DateRange) =>
    api.post<RiskMetrics>(`/api/risk/${accountId}`, dateRange),

  getHoldingsAnalysis: (accountId: string) =>
    api.get<HoldingsAnalysis>(`/api/holdings/${accountId}`),

  runStressTests: (accountId: string) =>
    api.post<StressTestResult[]>(`/api/stress-test/${accountId}`),

  getStockInfo: (symbol: string) =>
    api.get(`/api/market/info/${symbol}`),

  getCurrentPrices: (symbols: string[]) =>
    api.get(`/api/market/prices`, { params: { symbols: symbols.join(',') } }),
}

export default api
