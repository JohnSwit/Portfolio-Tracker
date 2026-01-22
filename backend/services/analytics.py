"""
Core analytics engine for institutional long-only portfolio analysis

Provides fundamental calculations for:
- Active share and active weights
- Concentration metrics
- Weight analysis
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging

from models.portfolio import Holding

logger = logging.getLogger(__name__)


def compute_weights(
    holdings: List[Holding],
    total_value: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute portfolio weights by symbol

    Args:
        holdings: List of portfolio holdings
        total_value: Total portfolio value (if None, computed from holdings)

    Returns:
        Dictionary mapping symbol -> weight (as fraction, sums to 1.0)

    Example:
        >>> holdings = [Holding(symbol="AAPL", market_value=10000),
        ...             Holding(symbol="MSFT", market_value=5000)]
        >>> weights = compute_weights(holdings)
        >>> weights["AAPL"]
        0.6667
    """
    if not holdings:
        return {}

    if total_value is None:
        total_value = sum(h.market_value or 0 for h in holdings)

    if total_value == 0:
        logger.warning("Total portfolio value is zero")
        return {h.symbol: 0.0 for h in holdings}

    weights = {}
    for holding in holdings:
        mv = holding.market_value or 0
        weights[holding.symbol] = mv / total_value

    return weights


def compute_weights_by_sector(
    holdings: List[Holding],
    total_value: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute portfolio weights by sector

    Args:
        holdings: List of portfolio holdings
        total_value: Total portfolio value

    Returns:
        Dictionary mapping sector -> weight

    Note:
        Holdings without sector are grouped under "Unknown"
    """
    if not holdings:
        return {}

    if total_value is None:
        total_value = sum(h.market_value or 0 for h in holdings)

    if total_value == 0:
        return {}

    sector_values = {}
    for holding in holdings:
        sector = holding.sector or "Unknown"
        mv = holding.market_value or 0
        sector_values[sector] = sector_values.get(sector, 0) + mv

    sector_weights = {
        sector: value / total_value
        for sector, value in sector_values.items()
    }

    return sector_weights


def compute_active_weights(
    portfolio_weights: Dict[str, float],
    benchmark_weights: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute active weights (portfolio - benchmark) for each symbol

    Args:
        portfolio_weights: Portfolio weights by symbol
        benchmark_weights: Benchmark weights by symbol

    Returns:
        Dictionary mapping symbol -> active weight (can be negative)

    Example:
        >>> port = {"AAPL": 0.15, "MSFT": 0.10}
        >>> bench = {"AAPL": 0.10, "MSFT": 0.12, "GOOGL": 0.08}
        >>> active = compute_active_weights(port, bench)
        >>> active["AAPL"]
        0.05  # overweight
        >>> active["MSFT"]
        -0.02  # underweight
        >>> active["GOOGL"]
        -0.08  # not held
    """
    # Get all symbols from both portfolio and benchmark
    all_symbols = set(portfolio_weights.keys()) | set(benchmark_weights.keys())

    active_weights = {}
    for symbol in all_symbols:
        port_wt = portfolio_weights.get(symbol, 0.0)
        bench_wt = benchmark_weights.get(symbol, 0.0)
        active_weights[symbol] = port_wt - bench_wt

    return active_weights


def compute_active_share(
    portfolio_weights: Dict[str, float],
    benchmark_weights: Dict[str, float]
) -> float:
    """
    Compute active share metric

    Active Share = 0.5 * sum(|portfolio_weight_i - benchmark_weight_i|)

    Interpretation:
        - 0%: Perfect index replication
        - 100%: No overlap with benchmark
        - Typical long-only active: 40-80%

    Args:
        portfolio_weights: Portfolio weights by symbol
        benchmark_weights: Benchmark weights by symbol

    Returns:
        Active share as a fraction (0.0 to 1.0)

    Example:
        >>> port = {"AAPL": 0.5, "MSFT": 0.5}
        >>> bench = {"AAPL": 0.5, "GOOGL": 0.5}
        >>> active_share = compute_active_share(port, bench)
        >>> active_share
        0.5  # 50% active share
    """
    active_weights = compute_active_weights(portfolio_weights, benchmark_weights)

    # Sum of absolute active weights
    total_abs_active = sum(abs(wt) for wt in active_weights.values())

    # Active share is half of this sum
    active_share = 0.5 * total_abs_active

    return active_share


def compute_concentration_metrics(
    holdings: List[Holding],
    total_value: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute concentration metrics for portfolio

    Metrics:
        - top5_weight: Sum of top 5 holdings weights
        - top10_weight: Sum of top 10 holdings weights
        - hhi_holdings: Herfindahl-Hirschman Index (sum of squared weights)
        - num_holdings: Number of holdings

    Args:
        holdings: List of portfolio holdings
        total_value: Total portfolio value

    Returns:
        Dictionary of concentration metrics

    Note:
        HHI interpretation:
        - 1.0: Single holding (max concentration)
        - 1/N: Equally weighted N holdings
        - Lower = more diversified

    Example:
        >>> holdings = [Holding(symbol="A", market_value=100),
        ...             Holding(symbol="B", market_value=50)]
        >>> metrics = compute_concentration_metrics(holdings)
        >>> round(metrics["hhi_holdings"], 3)
        0.556  # (100/150)^2 + (50/150)^2
    """
    if not holdings:
        return {
            "top5_weight": 0.0,
            "top10_weight": 0.0,
            "hhi_holdings": 0.0,
            "num_holdings": 0
        }

    # Compute weights
    weights = compute_weights(holdings, total_value)

    # Sort by weight descending
    sorted_weights = sorted(weights.values(), reverse=True)

    # Top N concentrations
    top5_weight = sum(sorted_weights[:5])
    top10_weight = sum(sorted_weights[:10])

    # HHI = sum of squared weights
    hhi = sum(w ** 2 for w in sorted_weights)

    return {
        "top5_weight": top5_weight,
        "top10_weight": top10_weight,
        "hhi_holdings": hhi,
        "num_holdings": len(holdings)
    }


def compute_sector_concentration_metrics(
    holdings: List[Holding],
    total_value: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute concentration metrics by sector

    Args:
        holdings: List of portfolio holdings
        total_value: Total portfolio value

    Returns:
        Dictionary with:
        - hhi_sectors: HHI for sector weights
        - max_sector_weight: Largest sector weight
        - num_sectors: Number of distinct sectors
    """
    if not holdings:
        return {
            "hhi_sectors": 0.0,
            "max_sector_weight": 0.0,
            "num_sectors": 0
        }

    sector_weights = compute_weights_by_sector(holdings, total_value)

    # HHI for sectors
    hhi = sum(w ** 2 for w in sector_weights.values())

    # Max sector weight
    max_weight = max(sector_weights.values()) if sector_weights else 0.0

    return {
        "hhi_sectors": hhi,
        "max_sector_weight": max_weight,
        "num_sectors": len(sector_weights)
    }


def compute_position_sizes(
    holdings: List[Holding],
    total_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Create DataFrame of position sizes for analysis

    Args:
        holdings: List of portfolio holdings
        total_value: Total portfolio value

    Returns:
        DataFrame with columns:
        - symbol: Ticker
        - market_value: Position market value
        - weight: Position weight
        - sector: Sector (or "Unknown")
        - rank: Rank by size (1 = largest)

    Sorted by weight descending
    """
    if not holdings:
        return pd.DataFrame(columns=["symbol", "market_value", "weight", "sector", "rank"])

    weights = compute_weights(holdings, total_value)

    data = []
    for holding in holdings:
        data.append({
            "symbol": holding.symbol,
            "market_value": holding.market_value or 0,
            "weight": weights.get(holding.symbol, 0.0),
            "sector": holding.sector or "Unknown"
        })

    df = pd.DataFrame(data)
    df = df.sort_values("weight", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    return df


def compute_turnover(
    current_weights: Dict[str, float],
    previous_weights: Dict[str, float]
) -> float:
    """
    Compute portfolio turnover between two periods

    Turnover = 0.5 * sum(|current_weight_i - previous_weight_i|)

    Args:
        current_weights: Current period weights by symbol
        previous_weights: Previous period weights by symbol

    Returns:
        Turnover as a fraction (0.0 to 1.0)

    Example:
        >>> prev = {"AAPL": 0.5, "MSFT": 0.5}
        >>> curr = {"AAPL": 0.3, "GOOGL": 0.7}
        >>> turnover = compute_turnover(curr, prev)
        >>> turnover
        0.6  # 60% turnover (sold MSFT, bought GOOGL, reduced AAPL)
    """
    all_symbols = set(current_weights.keys()) | set(previous_weights.keys())

    total_abs_diff = 0.0
    for symbol in all_symbols:
        curr_wt = current_weights.get(symbol, 0.0)
        prev_wt = previous_weights.get(symbol, 0.0)
        total_abs_diff += abs(curr_wt - prev_wt)

    turnover = 0.5 * total_abs_diff

    return turnover


# Utility functions

def validate_weights(weights: Dict[str, float], tolerance: float = 0.01) -> bool:
    """
    Validate that weights sum to approximately 1.0

    Args:
        weights: Dictionary of weights
        tolerance: Acceptable deviation from 1.0

    Returns:
        True if valid, False otherwise
    """
    total = sum(weights.values())
    return abs(total - 1.0) < tolerance


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize weights to sum to 1.0

    Args:
        weights: Dictionary of weights

    Returns:
        Normalized weights (sum = 1.0)
    """
    total = sum(weights.values())

    if total == 0:
        logger.warning("Cannot normalize weights: total is zero")
        return weights

    return {symbol: wt / total for symbol, wt in weights.items()}
