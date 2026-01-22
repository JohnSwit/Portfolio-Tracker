"""
Performance attribution for institutional portfolio analysis

Implements:
- Brinson attribution (allocation, selection, interaction)
- Position-level contribution to return
- Sector-level attribution analysis
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging

from models.portfolio import Holding

logger = logging.getLogger(__name__)


def compute_contribution_to_return(
    holdings: List[Holding],
    returns: Dict[str, float],
    total_value: Optional[float] = None
) -> List[Dict[str, float]]:
    """
    Compute position-level contribution to portfolio return

    Contribution = weight * return

    Args:
        holdings: List of portfolio holdings
        returns: Dictionary mapping symbol -> return (as fraction)
        total_value: Total portfolio value (computed if None)

    Returns:
        List of dicts with keys: symbol, weight, return, contribution
        Sorted by contribution (descending)

    Example:
        >>> holdings = [Holding(symbol="AAPL", market_value=10000),
        ...             Holding(symbol="MSFT", market_value=5000)]
        >>> returns = {"AAPL": 0.10, "MSFT": 0.05}
        >>> contrib = compute_contribution_to_return(holdings, returns)
        >>> contrib[0]["contribution"]
        0.0667  # (10000/15000) * 0.10
    """
    if not holdings:
        return []

    if total_value is None:
        total_value = sum(h.market_value or 0 for h in holdings)

    if total_value == 0:
        logger.warning("Total portfolio value is zero")
        return []

    contributions = []
    for holding in holdings:
        symbol = holding.symbol
        weight = (holding.market_value or 0) / total_value
        ret = returns.get(symbol, 0.0)
        contribution = weight * ret

        contributions.append({
            "symbol": symbol,
            "weight": weight,
            "return": ret,
            "contribution": contribution
        })

    # Sort by contribution (descending)
    contributions.sort(key=lambda x: x["contribution"], reverse=True)

    return contributions


def brinson_attribution(
    portfolio_holdings: List[Holding],
    benchmark_holdings: List[Holding],
    portfolio_returns: Dict[str, float],
    benchmark_returns: Dict[str, float],
    portfolio_total_value: Optional[float] = None,
    benchmark_total_value: Optional[float] = None
) -> Dict[str, any]:
    """
    Brinson-Fachler attribution analysis

    Decomposes active return into:
    - Allocation effect: return from sector over/underweighting
    - Selection effect: return from security selection within sectors
    - Interaction effect: combined allocation + selection

    Formula (by sector):
    - Allocation: (w_p - w_b) * (r_b - r_bench_total)
    - Selection: w_b * (r_p - r_b)
    - Interaction: (w_p - w_b) * (r_p - r_b)

    Where:
    - w_p, w_b = portfolio and benchmark sector weights
    - r_p, r_b = portfolio and benchmark sector returns
    - r_bench_total = total benchmark return

    Args:
        portfolio_holdings: Portfolio holdings
        benchmark_holdings: Benchmark holdings
        portfolio_returns: Symbol-level returns for portfolio
        benchmark_returns: Symbol-level returns for benchmark
        portfolio_total_value: Total portfolio value
        benchmark_total_value: Total benchmark value

    Returns:
        Dictionary with:
        - allocation_effect: Total allocation effect
        - selection_effect: Total selection effect
        - interaction_effect: Total interaction effect
        - total_active_return: Sum of effects
        - sector_attribution: Breakdown by sector

    Note:
        Holdings without sector are grouped under "Unknown"
    """
    if not portfolio_holdings or not benchmark_holdings:
        return {
            "allocation_effect": 0.0,
            "selection_effect": 0.0,
            "interaction_effect": 0.0,
            "total_active_return": 0.0,
            "sector_attribution": {}
        }

    # Compute portfolio and benchmark total values
    if portfolio_total_value is None:
        portfolio_total_value = sum(h.market_value or 0 for h in portfolio_holdings)

    if benchmark_total_value is None:
        benchmark_total_value = sum(h.market_value or 0 for h in benchmark_holdings)

    if portfolio_total_value == 0 or benchmark_total_value == 0:
        logger.warning("Portfolio or benchmark value is zero")
        return {
            "allocation_effect": 0.0,
            "selection_effect": 0.0,
            "interaction_effect": 0.0,
            "total_active_return": 0.0,
            "sector_attribution": {}
        }

    # Aggregate holdings by sector
    portfolio_sectors = _aggregate_by_sector(portfolio_holdings, portfolio_returns, portfolio_total_value)
    benchmark_sectors = _aggregate_by_sector(benchmark_holdings, benchmark_returns, benchmark_total_value)

    # Compute total benchmark return
    benchmark_total_return = sum(
        sector_data["weight"] * sector_data["return"]
        for sector_data in benchmark_sectors.values()
    )

    # Get all sectors (union)
    all_sectors = set(portfolio_sectors.keys()) | set(benchmark_sectors.keys())

    # Compute attribution effects by sector
    sector_attribution = {}
    total_allocation = 0.0
    total_selection = 0.0
    total_interaction = 0.0

    for sector in all_sectors:
        port_data = portfolio_sectors.get(sector, {"weight": 0.0, "return": 0.0})
        bench_data = benchmark_sectors.get(sector, {"weight": 0.0, "return": 0.0})

        w_p = port_data["weight"]
        w_b = bench_data["weight"]
        r_p = port_data["return"]
        r_b = bench_data["return"]

        # Brinson effects
        allocation = (w_p - w_b) * (r_b - benchmark_total_return)
        selection = w_b * (r_p - r_b)
        interaction = (w_p - w_b) * (r_p - r_b)

        sector_attribution[sector] = {
            "portfolio_weight": w_p,
            "benchmark_weight": w_b,
            "portfolio_return": r_p,
            "benchmark_return": r_b,
            "allocation_effect": allocation,
            "selection_effect": selection,
            "interaction_effect": interaction,
            "total_effect": allocation + selection + interaction
        }

        total_allocation += allocation
        total_selection += selection
        total_interaction += interaction

    return {
        "allocation_effect": total_allocation,
        "selection_effect": total_selection,
        "interaction_effect": total_interaction,
        "total_active_return": total_allocation + total_selection + total_interaction,
        "benchmark_return": benchmark_total_return,
        "sector_attribution": sector_attribution
    }


def _aggregate_by_sector(
    holdings: List[Holding],
    returns: Dict[str, float],
    total_value: float
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate holdings by sector, computing sector weights and returns

    Args:
        holdings: List of holdings
        returns: Symbol-level returns
        total_value: Total portfolio value

    Returns:
        Dictionary mapping sector -> {weight, return}
        where return is value-weighted average of constituent returns
    """
    sector_data = {}

    for holding in holdings:
        sector = holding.sector or "Unknown"
        symbol = holding.symbol
        market_value = holding.market_value or 0
        ret = returns.get(symbol, 0.0)

        if sector not in sector_data:
            sector_data[sector] = {"value": 0.0, "weighted_return": 0.0}

        sector_data[sector]["value"] += market_value
        sector_data[sector]["weighted_return"] += market_value * ret

    # Compute weights and average returns
    result = {}
    for sector, data in sector_data.items():
        weight = data["value"] / total_value if total_value > 0 else 0.0
        avg_return = data["weighted_return"] / data["value"] if data["value"] > 0 else 0.0

        result[sector] = {
            "weight": weight,
            "return": avg_return
        }

    return result


def get_top_contributors(
    contributions: List[Dict[str, float]],
    n: int = 10
) -> List[Dict[str, float]]:
    """
    Get top N contributors to return

    Args:
        contributions: List from compute_contribution_to_return()
        n: Number of top contributors to return

    Returns:
        Top N contributors (already sorted by contribution)
    """
    return contributions[:n]


def get_top_detractors(
    contributions: List[Dict[str, float]],
    n: int = 10
) -> List[Dict[str, float]]:
    """
    Get top N detractors to return (most negative contributors)

    Args:
        contributions: List from compute_contribution_to_return()
        n: Number of top detractors to return

    Returns:
        Top N detractors sorted by contribution (ascending)
    """
    # Sort by contribution ascending (most negative first)
    sorted_contribs = sorted(contributions, key=lambda x: x["contribution"])
    return sorted_contribs[:n]


def compute_active_return_from_contributions(
    contributions: List[Dict[str, float]]
) -> float:
    """
    Compute total portfolio active return from position contributions

    Args:
        contributions: List from compute_contribution_to_return()

    Returns:
        Sum of all contributions
    """
    return sum(c["contribution"] for c in contributions)


# Validation utilities

def validate_attribution(
    allocation: float,
    selection: float,
    interaction: float,
    portfolio_return: float,
    benchmark_return: float,
    tolerance: float = 0.001
) -> bool:
    """
    Validate that Brinson attribution sums to active return

    Args:
        allocation: Allocation effect
        selection: Selection effect
        interaction: Interaction effect
        portfolio_return: Total portfolio return
        benchmark_return: Total benchmark return
        tolerance: Acceptable error tolerance

    Returns:
        True if validation passes
    """
    active_return = portfolio_return - benchmark_return
    attribution_sum = allocation + selection + interaction

    error = abs(active_return - attribution_sum)

    if error > tolerance:
        logger.warning(
            f"Attribution validation failed: "
            f"active return = {active_return:.6f}, "
            f"attribution sum = {attribution_sum:.6f}, "
            f"error = {error:.6f}"
        )
        return False

    return True
