"""
Data models for institutional analytics

New models for long-only active management analytics
"""

from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class BenchmarkConfig(BaseModel):
    """Benchmark configuration"""
    ticker: str = Field(..., description="Benchmark ticker symbol (e.g., SPY, ^GSPC)")
    name: str = Field(..., description="Benchmark name (e.g., S&P 500)")
    composition_source: str = Field(default="etf", description="Source: 'etf', 'index', or 'manual'")
    rebalance_frequency: str = Field(default="quarterly", description="Rebalance frequency")


class ActiveMetrics(BaseModel):
    """Active management metrics relative to benchmark"""
    active_share: float = Field(..., description="Active share (0.0 to 1.0)")
    tracking_error: float = Field(..., description="Realized tracking error (annualized)")
    information_ratio: Optional[float] = Field(None, description="Active return / tracking error")
    active_risk_contributors: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Top contributors to active risk"
    )


class ConcentrationMetrics(BaseModel):
    """Portfolio concentration measures"""
    top5_weight: float = Field(..., description="Sum of top 5 holdings weights")
    top10_weight: float = Field(..., description="Sum of top 10 holdings weights")
    hhi_holdings: float = Field(..., description="HHI for individual holdings")
    hhi_sectors: float = Field(..., description="HHI for sector weights")
    num_holdings: int = Field(..., description="Total number of holdings")
    num_sectors: int = Field(..., description="Number of distinct sectors")
    avg_correlation_60d: Optional[float] = Field(None, description="60-day avg pairwise correlation")
    avg_correlation_252d: Optional[float] = Field(None, description="252-day avg pairwise correlation")


class LiquidityMetrics(BaseModel):
    """Liquidity and implementation metrics"""
    days_to_exit_percentiles: Dict[str, float] = Field(
        default_factory=dict,
        description="Days-to-exit distribution (p25, p50, p75)"
    )
    pct_illiquid_3d: float = Field(..., description="% of portfolio in positions > 3 days to exit")
    pct_illiquid_5d: float = Field(..., description="% of portfolio in positions > 5 days to exit")
    pct_illiquid_10d: float = Field(..., description="% of portfolio in positions > 10 days to exit")
    warnings: List[Dict[str, any]] = Field(
        default_factory=list,
        description="Liquidity warnings for specific positions"
    )


class FactorExposures(BaseModel):
    """Factor exposures (portfolio and active vs benchmark)"""
    # Portfolio factor betas
    market_beta: float = Field(..., description="Market factor beta")
    size_tilt: float = Field(..., description="Size factor tilt (SMB)")
    value_tilt: float = Field(..., description="Value factor tilt (HML)")
    momentum_tilt: float = Field(..., description="Momentum factor tilt")
    quality_tilt: float = Field(..., description="Quality factor tilt")
    low_vol_tilt: float = Field(..., description="Low volatility tilt")

    # Active tilts (portfolio - benchmark)
    active_tilts: Dict[str, float] = Field(
        default_factory=dict,
        description="Active factor tilts vs benchmark"
    )

    # Regression R-squared
    r_squared: Optional[float] = Field(None, description="Factor model R-squared")


class AttributionDetail(BaseModel):
    """Detailed attribution breakdown"""
    # Brinson attribution
    allocation_effect: float = Field(..., description="Sector allocation effect")
    selection_effect: float = Field(..., description="Security selection effect")
    interaction_effect: float = Field(..., description="Interaction effect")

    # Sector breakdown
    sector_attribution: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Attribution by sector {sector: {allocation, selection}}"
    )

    # Top contributors/detractors
    top_contributors: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Top positive contributors [{symbol, contribution}]"
    )
    top_detractors: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Top negative contributors [{symbol, contribution}]"
    )


class DrawdownMetrics(BaseModel):
    """Drawdown analysis"""
    max_drawdown: float = Field(..., description="Maximum drawdown (negative)")
    max_drawdown_start: Optional[datetime] = Field(None, description="Start date of max drawdown")
    max_drawdown_end: Optional[datetime] = Field(None, description="End date of max drawdown")
    max_drawdown_duration_days: Optional[int] = Field(None, description="Duration in days")

    current_drawdown: float = Field(..., description="Current drawdown from peak")
    current_drawdown_duration_days: Optional[int] = Field(None, description="Current drawdown duration")

    # Relative drawdown (portfolio - benchmark)
    relative_drawdown_max: Optional[float] = Field(None, description="Max relative drawdown")
    relative_drawdown_current: Optional[float] = Field(None, description="Current relative drawdown")


class CaptureRatios(BaseModel):
    """Capture ratios vs benchmark"""
    upside_capture: float = Field(..., description="Upside capture ratio (%)  ")
    downside_capture: float = Field(..., description="Downside capture ratio (%)")
    up_months: int = Field(..., description="Number of up months in period")
    down_months: int = Field(..., description="Number of down months in period")


class CIODashboard(BaseModel):
    """
    CIO Summary Dashboard

    One-page overview of institutional analytics for active long-only portfolio
    """
    # Metadata
    period: str = Field(..., description="Time period (e.g., '1Y', '3Y', 'ITD')")
    as_of_date: datetime = Field(..., description="Report date")
    portfolio_value: float = Field(..., description="Total portfolio value")
    benchmark_ticker: str = Field(..., description="Benchmark ticker")

    # Key Active Metrics
    active_share: float = Field(..., description="Active share vs benchmark")
    tracking_error: float = Field(..., description="Realized tracking error (ann.)")
    information_ratio: Optional[float] = Field(None, description="IR")

    # Performance
    portfolio_return: float = Field(..., description="Portfolio return for period")
    benchmark_return: float = Field(..., description="Benchmark return for period")
    active_return: float = Field(..., description="Active return (portfolio - benchmark)")

    # Attribution
    attribution: AttributionDetail = Field(..., description="Detailed attribution")

    # Risk
    active_risk_contributors: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Top contributors to active risk"
    )
    drawdown_metrics: DrawdownMetrics = Field(..., description="Drawdown analysis")
    capture_ratios: CaptureRatios = Field(..., description="Up/down capture ratios")

    # Positioning
    sector_active_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Active weights by sector"
    )
    factor_exposures: FactorExposures = Field(..., description="Factor tilts")

    # Concentration & Liquidity
    concentration: ConcentrationMetrics = Field(..., description="Concentration metrics")
    liquidity: LiquidityMetrics = Field(..., description="Liquidity metrics")


# Legacy model updates (add fields to existing models in portfolio.py)
# These would be added to models/portfolio.py but kept here for reference

class PerformanceMetricsExtended(BaseModel):
    """Extended performance metrics (add to existing PerformanceMetrics)"""
    # Active metrics
    active_share: Optional[float] = None
    active_risk_contribution_top5: Optional[List[Dict[str, float]]] = None

    # Capture ratios
    upside_capture: Optional[float] = None
    downside_capture: Optional[float] = None


class AttributionResultExtended(BaseModel):
    """Extended attribution (add to existing AttributionResult)"""
    # Brinson components
    allocation_effect: Optional[float] = None
    selection_effect: Optional[float] = None
    interaction_effect: Optional[float] = None

    # Contribution to return
    contribution_by_position: Optional[Dict[str, float]] = None
