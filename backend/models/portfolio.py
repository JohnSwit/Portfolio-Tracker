from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class Holding(BaseModel):
    """Individual position in the portfolio"""
    symbol: str
    quantity: float
    cost_basis: float
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    asset_class: str = "equity"

    # Factor exposures
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None

    # Liquidity metrics
    avg_daily_volume: Optional[float] = None
    last_updated: Optional[datetime] = None


class Transaction(BaseModel):
    """Portfolio transaction"""
    date: datetime
    symbol: str
    transaction_type: str  # buy, sell, dividend, split
    quantity: float
    price: float
    amount: float
    fees: float = 0.0
    notes: Optional[str] = None


class Portfolio(BaseModel):
    """Complete portfolio information"""
    account_id: str
    account_name: Optional[str] = None
    holdings: List[Holding] = []
    transactions: List[Transaction] = []
    cash_balance: float = 0.0
    total_value: Optional[float] = None
    inception_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None


class PerformanceMetrics(BaseModel):
    """Performance measurement results"""
    period: str
    start_date: datetime
    end_date: datetime

    # Returns
    twr: float = Field(..., description="Time-Weighted Return")
    mwr: float = Field(..., description="Money-Weighted Return (IRR)")
    total_return: float
    annualized_return: Optional[float] = None

    # Benchmark comparison
    benchmark_return: Optional[float] = None
    active_return: Optional[float] = None
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None


class AttributionResult(BaseModel):
    """Attribution analysis results"""
    period: str
    total_return: float

    # Sector attribution
    sector_attribution: Dict[str, Dict[str, float]] = {}

    # Country attribution
    country_attribution: Dict[str, Dict[str, float]] = {}

    # Stock-level attribution
    stock_attribution: Dict[str, Dict[str, float]] = {}

    # Factor attribution
    factor_attribution: Dict[str, float] = {}


class RiskMetrics(BaseModel):
    """Risk analytics results"""
    period: str

    # Volatility measures
    daily_volatility: float
    annual_volatility: float
    downside_volatility: float

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: Optional[float] = None

    # Market risk
    beta: float
    alpha: Optional[float] = None
    correlation: Optional[float] = None

    # Drawdown
    max_drawdown: float
    max_drawdown_duration: Optional[int] = None
    current_drawdown: float

    # Value at Risk
    var_95: float = Field(..., description="95% VaR")
    var_99: float = Field(..., description="99% VaR")
    cvar_95: float = Field(..., description="95% Conditional VaR")

    # Factor exposures
    factor_exposures: Dict[str, float] = {}


class HoldingsAnalysis(BaseModel):
    """Holdings analysis results"""

    # Concentration
    top_10_concentration: float
    top_20_concentration: float
    herfindahl_index: float
    active_share: Optional[float] = None

    # Exposure breakdown
    sector_exposure: Dict[str, float] = {}
    country_exposure: Dict[str, float] = {}
    industry_exposure: Dict[str, float] = {}
    asset_class_exposure: Dict[str, float] = {}

    # Liquidity
    avg_daily_volume: float
    weighted_liquidity_score: float
    illiquid_holdings_pct: float

    # Factor exposures
    avg_market_cap: float
    median_market_cap: float
    avg_pe_ratio: Optional[float] = None
    avg_pb_ratio: Optional[float] = None
    weighted_beta: float
    value_score: Optional[float] = None
    growth_score: Optional[float] = None
    momentum_score: Optional[float] = None
    quality_score: Optional[float] = None


class StressTestScenario(BaseModel):
    """Stress test scenario definition"""
    name: str
    description: str
    factor_shocks: Dict[str, float]  # factor -> percentage change
    market_shock: Optional[float] = None


class StressTestResult(BaseModel):
    """Results from stress testing"""
    scenario: str
    estimated_loss: float
    estimated_loss_pct: float
    portfolio_value_after: float
    var_breach: bool
