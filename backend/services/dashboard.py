"""
CIO Dashboard orchestration service

Aggregates all institutional analytics into a comprehensive executive dashboard
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

from models.portfolio import Holding, Portfolio
from models.analytics_models import (
    CIODashboard,
    AttributionDetail,
    DrawdownMetrics,
    CaptureRatios,
    FactorExposures,
    ConcentrationMetrics,
    LiquidityMetrics
)

# Import analytics services
from services import analytics
from services import tracking_error
from services import attribution
from services import drawdown
from services import liquidity
from services import factor_analysis

logger = logging.getLogger(__name__)


def build_cio_dashboard(
    portfolio: Portfolio,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    benchmark_holdings: Optional[List[Holding]],
    volume_data: Dict[str, float],
    factor_returns: Dict[str, pd.Series],
    period: str = "1Y",
    benchmark_ticker: str = "SPY"
) -> CIODashboard:
    """
    Build comprehensive CIO dashboard aggregating all analytics

    Args:
        portfolio: Current portfolio state
        portfolio_returns: Historical portfolio returns (daily)
        benchmark_returns: Historical benchmark returns (daily)
        benchmark_holdings: Benchmark holdings (for active share)
        volume_data: Average daily volumes by symbol
        factor_returns: Factor return series for style analysis
        period: Reporting period
        benchmark_ticker: Benchmark ticker symbol

    Returns:
        CIODashboard with all institutional analytics

    Example:
        >>> dashboard = build_cio_dashboard(
        ...     portfolio, port_returns, bench_returns,
        ...     bench_holdings, volume_data, factor_returns
        ... )
        >>> dashboard.active_share
        0.42  # 42% active share
    """
    # Metadata
    as_of_date = datetime.now()
    total_value = sum(h.market_value or 0 for h in portfolio.holdings)

    # === Active Metrics ===
    # Active share
    portfolio_weights = analytics.compute_weights(portfolio.holdings, total_value)

    if benchmark_holdings:
        benchmark_total = sum(h.market_value or 0 for h in benchmark_holdings)
        benchmark_weights = analytics.compute_weights(benchmark_holdings, benchmark_total)
    else:
        # If no benchmark holdings, assume equal-weight or use proxy
        benchmark_weights = {}

    active_share_value = analytics.compute_active_share(portfolio_weights, benchmark_weights)

    # Tracking error and IR
    active_rets = tracking_error.compute_active_returns(portfolio_returns, benchmark_returns)
    te = tracking_error.compute_tracking_error(active_rets)
    ir = tracking_error.compute_information_ratio(active_rets)

    # === Performance ===
    # Period returns
    period_days = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "3Y": 756, "ITD": len(portfolio_returns)}
    lookback = period_days.get(period, 252)

    portfolio_returns_period = portfolio_returns.tail(lookback)
    benchmark_returns_period = benchmark_returns.tail(lookback)

    portfolio_return = (1 + portfolio_returns_period).prod() - 1
    benchmark_return = (1 + benchmark_returns_period).prod() - 1
    active_return = portfolio_return - benchmark_return

    # === Attribution ===
    # Position-level contributions
    # Get period returns for each position
    position_returns = _compute_position_returns(portfolio.holdings, portfolio_returns_period)

    contributions = attribution.compute_contribution_to_return(
        portfolio.holdings,
        position_returns,
        total_value
    )

    # Brinson attribution (if benchmark holdings available)
    if benchmark_holdings:
        brinson = attribution.brinson_attribution(
            portfolio.holdings,
            benchmark_holdings,
            position_returns,
            _compute_position_returns(benchmark_holdings, benchmark_returns_period),
            total_value,
            sum(h.market_value or 0 for h in benchmark_holdings)
        )

        attribution_detail = AttributionDetail(
            period=period,
            allocation_effect=brinson["allocation_effect"],
            selection_effect=brinson["selection_effect"],
            interaction_effect=brinson["interaction_effect"],
            total_active_return=brinson["total_active_return"],
            sector_attribution=brinson["sector_attribution"],
            top_contributors=attribution.get_top_contributors(contributions, n=10),
            top_detractors=attribution.get_top_detractors(contributions, n=10)
        )
    else:
        # No benchmark - just top contributors/detractors
        attribution_detail = AttributionDetail(
            period=period,
            allocation_effect=0.0,
            selection_effect=active_return,
            interaction_effect=0.0,
            total_active_return=active_return,
            sector_attribution={},
            top_contributors=attribution.get_top_contributors(contributions, n=10),
            top_detractors=attribution.get_top_detractors(contributions, n=10)
        )

    # === Risk ===
    # Active risk contributors (from tracking error decomposition)
    te_components = tracking_error.analyze_tracking_error_components(
        active_rets.tail(lookback),
        portfolio_returns_period,
        benchmark_returns_period
    )

    active_risk_contributors = [
        {"component": "Systematic", "contribution": te_components.get("systematic_component", 0.0)},
        {"component": "Idiosyncratic", "contribution": te_components.get("idiosyncratic_component", 0.0)}
    ]

    # Drawdown metrics
    dd_metrics = drawdown.compute_all_drawdown_metrics(
        portfolio_returns_period,
        benchmark_returns_period
    )

    drawdown_metrics_obj = DrawdownMetrics(
        max_drawdown=dd_metrics["max_drawdown"],
        current_drawdown=dd_metrics["current_drawdown"],
        recovery_date=dd_metrics.get("recovery_date"),
        max_drawdown_duration_days=dd_metrics.get("max_drawdown_duration_days"),
        relative_drawdown_max=dd_metrics.get("relative_drawdown_max"),
        calmar_ratio=drawdown.compute_calmar_ratio(portfolio_returns_period)
    )

    # Capture ratios
    capture_ratios_obj = CaptureRatios(
        upside_capture=dd_metrics.get("upside_capture"),
        downside_capture=dd_metrics.get("downside_capture"),
        up_months=dd_metrics.get("up_months", 0),
        down_months=dd_metrics.get("down_months", 0)
    )

    # === Positioning ===
    # Sector active weights
    sector_active_weights = _compute_sector_active_weights(
        portfolio.holdings,
        benchmark_holdings,
        total_value,
        sum(h.market_value or 0 for h in benchmark_holdings) if benchmark_holdings else total_value
    )

    # Factor exposures
    factor_exposures_result = factor_analysis.estimate_factor_exposures_regression(
        portfolio_returns_period,
        factor_returns
    )

    # Identify tilts
    tilts = factor_analysis.identify_factor_tilts(
        factor_exposures_result["exposures"],
        significance_threshold=0.10
    )

    factor_exposures_obj = FactorExposures(
        market_beta=factor_exposures_result["exposures"].get("Market", 1.0),
        size_factor=factor_exposures_result["exposures"].get("Size", 0.0),
        value_factor=factor_exposures_result["exposures"].get("Value", 0.0),
        growth_factor=factor_exposures_result["exposures"].get("Growth", 0.0),
        momentum_factor=factor_exposures_result["exposures"].get("Momentum", 0.0),
        quality_factor=factor_exposures_result["exposures"].get("Quality", 0.0),
        low_vol_factor=factor_exposures_result["exposures"].get("Low_Vol", 0.0),
        r_squared=factor_exposures_result["r_squared"],
        significant_tilts=tilts["significant_tilts"]
    )

    # === Concentration & Liquidity ===
    concentration_result = analytics.compute_concentration_metrics(portfolio.holdings, total_value)

    concentration_obj = ConcentrationMetrics(
        top5_weight=concentration_result["top5_weight"],
        top10_weight=concentration_result["top10_weight"],
        hhi_holdings=concentration_result["hhi_holdings"],
        hhi_sectors=0.0,  # Would need sector data
        effective_n_holdings=1.0 / concentration_result["hhi_holdings"] if concentration_result["hhi_holdings"] > 0 else 0
    )

    # Liquidity profile
    liq_profile = liquidity.compute_portfolio_liquidity_profile(
        portfolio.holdings,
        volume_data,
        max_volume_participation=0.20
    )

    liquidity_obj = LiquidityMetrics(
        weighted_avg_days_to_exit=liq_profile["weighted_avg_days_to_exit"],
        pct_highly_liquid=liq_profile["liquidity_distribution"].get("Highly Liquid", 0.0) * 100,
        pct_illiquid=liq_profile["liquidity_distribution"].get("Illiquid", 0.0) * 100
        + liq_profile["liquidity_distribution"].get("Highly Illiquid", 0.0) * 100,
        most_illiquid_positions=liq_profile["most_illiquid_positions"][:5]
    )

    # Build final dashboard
    dashboard = CIODashboard(
        period=period,
        as_of_date=as_of_date,
        portfolio_value=total_value,
        benchmark_ticker=benchmark_ticker,
        active_share=active_share_value,
        tracking_error=te,
        information_ratio=ir,
        portfolio_return=portfolio_return,
        benchmark_return=benchmark_return,
        active_return=active_return,
        attribution=attribution_detail,
        active_risk_contributors=active_risk_contributors,
        drawdown_metrics=drawdown_metrics_obj,
        capture_ratios=capture_ratios_obj,
        sector_active_weights=sector_active_weights,
        factor_exposures=factor_exposures_obj,
        concentration=concentration_obj,
        liquidity=liquidity_obj
    )

    return dashboard


def _compute_position_returns(
    holdings: List[Holding],
    returns_series: pd.Series
) -> Dict[str, float]:
    """
    Compute returns for each position over the period

    This is a simplified version - in production, would compute
    from actual price data for each symbol.

    Args:
        holdings: List of holdings
        returns_series: Portfolio-level returns

    Returns:
        Dict mapping symbol -> return
    """
    # Placeholder: assume all positions had same return as portfolio
    # In production, compute from actual price data per symbol
    total_return = (1 + returns_series).prod() - 1

    return {h.symbol: total_return for h in holdings}


def _compute_sector_active_weights(
    portfolio_holdings: List[Holding],
    benchmark_holdings: Optional[List[Holding]],
    portfolio_value: float,
    benchmark_value: float
) -> Dict[str, float]:
    """
    Compute active weights by sector

    Args:
        portfolio_holdings: Portfolio holdings
        benchmark_holdings: Benchmark holdings
        portfolio_value: Total portfolio value
        benchmark_value: Total benchmark value

    Returns:
        Dict mapping sector -> active weight
    """
    # Portfolio sector weights
    port_sectors = analytics.compute_weights_by_sector(portfolio_holdings, portfolio_value)

    # Benchmark sector weights
    if benchmark_holdings and benchmark_value > 0:
        bench_sectors = analytics.compute_weights_by_sector(benchmark_holdings, benchmark_value)
    else:
        bench_sectors = {}

    # Active weights
    all_sectors = set(port_sectors.keys()) | set(bench_sectors.keys())
    active_weights = {}

    for sector in all_sectors:
        port_weight = port_sectors.get(sector, 0.0)
        bench_weight = bench_sectors.get(sector, 0.0)
        active_weights[sector] = port_weight - bench_weight

    return active_weights
