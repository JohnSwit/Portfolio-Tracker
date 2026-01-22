"""
Liquidity analysis and implementation risk metrics

Implements:
- Days-to-exit calculations (time to liquidate positions)
- Volume-based liquidity scoring
- Liquidity risk aggregation
- Position sizing constraints
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

from models.portfolio import Holding

logger = logging.getLogger(__name__)


def compute_days_to_exit(
    position_shares: float,
    avg_daily_volume: float,
    max_volume_participation: float = 0.20
) -> float:
    """
    Compute days required to exit a position given volume constraints

    Formula:
        days_to_exit = position_shares / (avg_daily_volume * max_participation)

    Args:
        position_shares: Number of shares held
        avg_daily_volume: Average daily trading volume
        max_volume_participation: Max % of daily volume to trade (default 20%)

    Returns:
        Number of trading days required to exit position

    Example:
        >>> compute_days_to_exit(100000, 1000000, 0.20)
        0.5  # Can exit in half a day
    """
    if avg_daily_volume <= 0:
        logger.warning(f"Invalid avg_daily_volume: {avg_daily_volume}")
        return float('inf')

    if max_volume_participation <= 0 or max_volume_participation > 1:
        logger.warning(f"Invalid max_volume_participation: {max_volume_participation}")
        max_volume_participation = 0.20

    max_daily_shares = avg_daily_volume * max_volume_participation
    if max_daily_shares <= 0:
        return float('inf')

    days = abs(position_shares) / max_daily_shares
    return days


def compute_liquidity_score(
    days_to_exit: float,
    thresholds: Optional[Dict[str, float]] = None
) -> str:
    """
    Categorize liquidity based on days-to-exit

    Args:
        days_to_exit: Days required to exit position
        thresholds: Custom thresholds (default: highly_liquid=1, liquid=3, illiquid=10)

    Returns:
        One of: "Highly Liquid", "Liquid", "Moderately Liquid", "Illiquid", "Highly Illiquid"

    Default thresholds:
        - Highly Liquid: < 1 day
        - Liquid: 1-3 days
        - Moderately Liquid: 3-10 days
        - Illiquid: 10-30 days
        - Highly Illiquid: > 30 days
    """
    if thresholds is None:
        thresholds = {
            "highly_liquid": 1.0,
            "liquid": 3.0,
            "moderately_liquid": 10.0,
            "illiquid": 30.0
        }

    if days_to_exit < thresholds["highly_liquid"]:
        return "Highly Liquid"
    elif days_to_exit < thresholds["liquid"]:
        return "Liquid"
    elif days_to_exit < thresholds["moderately_liquid"]:
        return "Moderately Liquid"
    elif days_to_exit < thresholds["illiquid"]:
        return "Illiquid"
    else:
        return "Highly Illiquid"


def compute_portfolio_liquidity_profile(
    holdings: List[Holding],
    volume_data: Dict[str, float],
    max_volume_participation: float = 0.20
) -> Dict[str, any]:
    """
    Compute liquidity profile for entire portfolio

    Args:
        holdings: List of portfolio holdings
        volume_data: Dict mapping symbol -> avg_daily_volume
        max_volume_participation: Max % of daily volume to trade

    Returns:
        Dictionary with:
        - position_liquidity: List of dicts with symbol, shares, days_to_exit, score
        - weighted_avg_days_to_exit: Portfolio-weighted average DTE
        - liquidity_distribution: % of portfolio in each liquidity bucket
        - most_illiquid_positions: Top 5 most illiquid positions
        - total_market_value: Total portfolio value
    """
    if not holdings:
        return {
            "position_liquidity": [],
            "weighted_avg_days_to_exit": 0.0,
            "liquidity_distribution": {},
            "most_illiquid_positions": [],
            "total_market_value": 0.0
        }

    total_value = sum(h.market_value or 0 for h in holdings)
    position_liquidity = []
    liquidity_buckets = {
        "Highly Liquid": 0.0,
        "Liquid": 0.0,
        "Moderately Liquid": 0.0,
        "Illiquid": 0.0,
        "Highly Illiquid": 0.0
    }

    weighted_dte_sum = 0.0

    for holding in holdings:
        symbol = holding.symbol
        shares = abs(holding.quantity or 0)
        market_value = holding.market_value or 0
        weight = market_value / total_value if total_value > 0 else 0

        # Get volume data
        avg_volume = volume_data.get(symbol, 0)

        if avg_volume > 0:
            dte = compute_days_to_exit(shares, avg_volume, max_volume_participation)
        else:
            dte = float('inf')
            logger.warning(f"No volume data for {symbol}, marking as illiquid")

        score = compute_liquidity_score(dte)

        position_liquidity.append({
            "symbol": symbol,
            "shares": shares,
            "market_value": market_value,
            "weight": weight,
            "avg_daily_volume": avg_volume,
            "days_to_exit": dte if dte != float('inf') else None,
            "liquidity_score": score
        })

        # Accumulate for distribution
        liquidity_buckets[score] += weight

        # Weighted average (exclude inf)
        if dte != float('inf'):
            weighted_dte_sum += weight * dte

    # Sort by days to exit (most illiquid first)
    position_liquidity_sorted = sorted(
        position_liquidity,
        key=lambda x: (x["days_to_exit"] is None, x["days_to_exit"] or 0),
        reverse=True
    )

    most_illiquid = position_liquidity_sorted[:5]

    return {
        "position_liquidity": position_liquidity_sorted,
        "weighted_avg_days_to_exit": weighted_dte_sum,
        "liquidity_distribution": liquidity_buckets,
        "most_illiquid_positions": most_illiquid,
        "total_market_value": total_value
    }


def compute_bid_ask_impact(
    shares: float,
    bid_price: float,
    ask_price: float,
    trade_side: str = "sell"
) -> Dict[str, float]:
    """
    Compute market impact from bid-ask spread

    Args:
        shares: Number of shares to trade
        bid_price: Current bid price
        ask_price: Current ask price
        trade_side: "buy" or "sell"

    Returns:
        Dictionary with:
        - spread_bps: Bid-ask spread in basis points
        - impact_cost: Dollar cost of crossing spread
        - impact_bps: Impact as % of mid price

    Example:
        >>> compute_bid_ask_impact(1000, 99.90, 100.10, "sell")
        {'spread_bps': 20, 'impact_cost': 100, 'impact_bps': 10}
    """
    if bid_price <= 0 or ask_price <= 0 or ask_price < bid_price:
        logger.warning(f"Invalid bid/ask: {bid_price}/{ask_price}")
        return {
            "spread_bps": None,
            "impact_cost": None,
            "impact_bps": None
        }

    mid_price = (bid_price + ask_price) / 2
    spread = ask_price - bid_price
    spread_bps = (spread / mid_price) * 10000

    # Cost of crossing spread
    if trade_side.lower() == "sell":
        # Sell at bid, miss out on mid-bid difference
        slippage_per_share = mid_price - bid_price
    else:  # buy
        # Buy at ask, pay ask-mid difference
        slippage_per_share = ask_price - mid_price

    impact_cost = abs(shares) * slippage_per_share
    impact_bps = (slippage_per_share / mid_price) * 10000

    return {
        "spread_bps": spread_bps,
        "impact_cost": impact_cost,
        "impact_bps": impact_bps,
        "mid_price": mid_price
    }


def estimate_market_impact(
    shares: float,
    avg_daily_volume: float,
    price: float,
    volatility: float = 0.02
) -> Dict[str, float]:
    """
    Estimate market impact using simplified square-root model

    Formula (simplified Almgren-Chriss):
        impact = volatility * sqrt(shares / avg_daily_volume) * price

    Args:
        shares: Number of shares to trade
        avg_daily_volume: Average daily volume
        price: Current stock price
        volatility: Daily volatility (default 2%)

    Returns:
        Dictionary with:
        - temporary_impact: Temporary price impact ($)
        - temporary_impact_bps: Impact in basis points
        - volume_participation: % of daily volume

    Note:
        This is a simplified model. Real impact depends on:
        - Order type and urgency
        - Market conditions
        - Trading style
    """
    if avg_daily_volume <= 0 or price <= 0:
        return {
            "temporary_impact": None,
            "temporary_impact_bps": None,
            "volume_participation": None
        }

    volume_pct = abs(shares) / avg_daily_volume

    # Square-root impact model
    # impact_bps = volatility * sqrt(volume_participation) * 10000
    impact_bps = volatility * np.sqrt(volume_pct) * 10000

    # Dollar impact
    impact_dollars = (impact_bps / 10000) * price * abs(shares)

    return {
        "temporary_impact": impact_dollars,
        "temporary_impact_bps": impact_bps,
        "volume_participation": volume_pct * 100,
        "shares": abs(shares),
        "avg_daily_volume": avg_daily_volume
    }


def compute_turnover_capacity(
    portfolio_value: float,
    liquidity_profile: Dict[str, any],
    target_horizon_days: int = 5
) -> Dict[str, float]:
    """
    Compute maximum portfolio turnover rate given liquidity constraints

    Args:
        portfolio_value: Total portfolio value
        liquidity_profile: Result from compute_portfolio_liquidity_profile()
        target_horizon_days: Desired liquidation horizon

    Returns:
        Dictionary with:
        - max_daily_turnover_pct: Max % of portfolio that can be turned over daily
        - constrained_positions: Positions limiting turnover
        - unconstrained_value: Value that can be liquidated quickly
    """
    if not liquidity_profile or portfolio_value <= 0:
        return {
            "max_daily_turnover_pct": 0.0,
            "constrained_positions": [],
            "unconstrained_value": 0.0
        }

    weighted_dte = liquidity_profile["weighted_avg_days_to_exit"]

    # Max daily turnover = 100% / weighted_days_to_exit
    # Adjusted for target horizon
    if weighted_dte > 0:
        max_daily_turnover = min(100.0 / weighted_dte, 100.0 / target_horizon_days)
    else:
        max_daily_turnover = 100.0

    # Find constrained positions (DTE > target horizon)
    constrained = []
    unconstrained_value = 0.0

    for pos in liquidity_profile["position_liquidity"]:
        dte = pos["days_to_exit"]
        if dte is None or dte > target_horizon_days:
            constrained.append({
                "symbol": pos["symbol"],
                "days_to_exit": dte,
                "market_value": pos["market_value"]
            })
        else:
            unconstrained_value += pos["market_value"]

    return {
        "max_daily_turnover_pct": max_daily_turnover,
        "constrained_positions": constrained,
        "unconstrained_value": unconstrained_value,
        "unconstrained_pct": (unconstrained_value / portfolio_value * 100) if portfolio_value > 0 else 0
    }


def validate_position_size(
    shares: float,
    avg_daily_volume: float,
    max_volume_pct: float = 0.20,
    max_days_to_exit: float = 5.0
) -> Dict[str, any]:
    """
    Validate if position size is within liquidity constraints

    Args:
        shares: Proposed position size
        avg_daily_volume: Average daily volume
        max_volume_pct: Max % of daily volume
        max_days_to_exit: Max acceptable days to exit

    Returns:
        Dictionary with:
        - is_valid: Boolean indicating if position meets constraints
        - days_to_exit: Actual days to exit
        - max_shares_allowed: Maximum shares allowed under constraints
        - excess_shares: Shares above limit (if any)
    """
    dte = compute_days_to_exit(shares, avg_daily_volume, max_volume_pct)

    max_shares_allowed = avg_daily_volume * max_volume_pct * max_days_to_exit
    is_valid = dte <= max_days_to_exit

    excess = max(0, abs(shares) - max_shares_allowed)

    return {
        "is_valid": is_valid,
        "days_to_exit": dte if dte != float('inf') else None,
        "max_shares_allowed": max_shares_allowed,
        "excess_shares": excess if not is_valid else 0,
        "utilization_pct": (abs(shares) / max_shares_allowed * 100) if max_shares_allowed > 0 else None
    }


# Helper function to get volume data (placeholder for actual implementation)
def get_volume_data_from_market(
    symbols: List[str],
    lookback_days: int = 20
) -> Dict[str, float]:
    """
    Fetch average daily volume for symbols

    This is a placeholder. In production, integrate with:
    - market_data_service.get_volume_data()
    - Or external data provider

    Args:
        symbols: List of symbols
        lookback_days: Days to average over

    Returns:
        Dict mapping symbol -> avg_daily_volume
    """
    # Placeholder implementation
    logger.warning("Using placeholder volume data - integrate with market_data_service")

    # Return dummy data for testing
    return {symbol: 1_000_000 for symbol in symbols}
