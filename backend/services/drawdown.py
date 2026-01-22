"""
Advanced drawdown analysis and capture ratio calculations

Complements existing risk.py with detailed drawdown metrics
and relative performance analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Compute drawdown series from returns

    Args:
        returns: Series of returns (daily, monthly, etc.)

    Returns:
        Series of drawdowns (negative values)

    Formula:
        drawdown_t = (cum_return_t - running_max_t) / running_max_t
    """
    if len(returns) == 0:
        return pd.Series(dtype=float)

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max

    return drawdown


def compute_max_drawdown(returns: pd.Series) -> Dict[str, any]:
    """
    Compute maximum drawdown with detailed metadata

    Args:
        returns: Series of returns with datetime index

    Returns:
        Dictionary with:
        - max_drawdown: Maximum drawdown (negative)
        - max_drawdown_start: Start date of max drawdown
        - max_drawdown_end: End date (trough)
        - max_drawdown_duration_days: Duration in days
        - recovery_date: Date of recovery (or None if not recovered)

    Example:
        >>> returns = pd.Series([0.01, 0.02, -0.05, -0.03, 0.04, 0.03])
        >>> result = compute_max_drawdown(returns)
        >>> result['max_drawdown']
        -0.0794  # Approximate
    """
    if len(returns) == 0:
        return {
            "max_drawdown": 0.0,
            "max_drawdown_start": None,
            "max_drawdown_end": None,
            "max_drawdown_duration_days": None,
            "recovery_date": None
        }

    drawdown = compute_drawdown_series(returns)
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()

    # Find max drawdown
    max_dd = drawdown.min()
    max_dd_end = drawdown.idxmin()

    # Find start (last peak before trough)
    max_dd_start = None
    if max_dd_end is not None:
        cumulative_to_end = cumulative.loc[:max_dd_end]
        running_max_to_end = running_max.loc[:max_dd_end]
        peak_mask = cumulative_to_end == running_max_to_end
        peaks = cumulative_to_end[peak_mask]
        if len(peaks) > 0:
            max_dd_start = peaks.index[-1]

    # Calculate duration
    duration = None
    if max_dd_start is not None and max_dd_end is not None:
        if isinstance(max_dd_start, pd.Timestamp) and isinstance(max_dd_end, pd.Timestamp):
            duration = (max_dd_end - max_dd_start).days
        elif isinstance(returns.index, pd.DatetimeIndex):
            duration = (max_dd_end - max_dd_start).days
        else:
            # Index positions
            duration = returns.index.get_loc(max_dd_end) - returns.index.get_loc(max_dd_start)

    # Find recovery date (when drawdown returns to 0)
    recovery_date = None
    if max_dd_end is not None:
        after_trough = drawdown.loc[max_dd_end:]
        recovered = after_trough >= -0.001  # Allow small tolerance
        if recovered.any():
            recovery_date = after_trough[recovered].index[0]

    return {
        "max_drawdown": max_dd,
        "max_drawdown_start": max_dd_start,
        "max_drawdown_end": max_dd_end,
        "max_drawdown_duration_days": duration,
        "recovery_date": recovery_date
    }


def compute_current_drawdown(returns: pd.Series) -> Dict[str, any]:
    """
    Compute current drawdown from peak

    Args:
        returns: Series of returns with datetime index

    Returns:
        Dictionary with:
        - current_drawdown: Current drawdown (negative)
        - current_drawdown_start: Start date of current drawdown
        - current_drawdown_duration_days: Duration in days
    """
    if len(returns) == 0:
        return {
            "current_drawdown": 0.0,
            "current_drawdown_start": None,
            "current_drawdown_duration_days": None
        }

    drawdown = compute_drawdown_series(returns)
    current_dd = drawdown.iloc[-1]

    # Find start of current drawdown (last zero drawdown)
    current_dd_start = None
    duration = None

    if current_dd < -0.001:  # In drawdown
        # Find last time we were at peak (drawdown == 0)
        at_peak = drawdown >= -0.001
        if at_peak.any():
            peaks = drawdown[at_peak]
            current_dd_start = peaks.index[-1]

            # Calculate duration
            if isinstance(current_dd_start, pd.Timestamp):
                duration = (returns.index[-1] - current_dd_start).days
            else:
                duration = len(returns) - returns.index.get_loc(current_dd_start) - 1

    return {
        "current_drawdown": current_dd,
        "current_drawdown_start": current_dd_start,
        "current_drawdown_duration_days": duration
    }


def compute_relative_drawdown(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> Dict[str, any]:
    """
    Compute relative drawdown (portfolio vs benchmark)

    Relative drawdown = cumulative portfolio return - cumulative benchmark return

    Args:
        portfolio_returns: Portfolio return series
        benchmark_returns: Benchmark return series

    Returns:
        Dictionary with:
        - relative_drawdown_max: Maximum relative drawdown
        - relative_drawdown_current: Current relative drawdown
        - relative_dd_max_date: Date of max relative drawdown

    Note:
        Negative relative drawdown means underperformance vs benchmark
    """
    # Align series
    aligned = pd.DataFrame({
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }).dropna()

    if len(aligned) == 0:
        return {
            "relative_drawdown_max": 0.0,
            "relative_drawdown_current": 0.0,
            "relative_dd_max_date": None
        }

    # Compute cumulative returns
    port_cumulative = (1 + aligned['portfolio']).cumprod()
    bench_cumulative = (1 + aligned['benchmark']).cumprod()

    # Relative performance
    relative_perf = port_cumulative / bench_cumulative - 1

    # Relative drawdown
    running_max = relative_perf.expanding().max()
    relative_dd = relative_perf - running_max

    return {
        "relative_drawdown_max": relative_dd.min(),
        "relative_drawdown_current": relative_dd.iloc[-1],
        "relative_dd_max_date": relative_dd.idxmin()
    }


def compute_capture_ratios(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    frequency: str = 'M'
) -> Dict[str, any]:
    """
    Compute upside and downside capture ratios

    Capture Ratio = (Portfolio Avg Return in Period) / (Benchmark Avg Return in Period) * 100

    Interpretation:
    - Upside capture > 100%: Portfolio captures more than market gains
    - Downside capture < 100%: Portfolio loses less than market declines
    - Ideal: High upside capture, low downside capture

    Args:
        portfolio_returns: Portfolio return series
        benchmark_returns: Benchmark return series
        frequency: Resampling frequency ('M' = monthly, 'D' = daily)

    Returns:
        Dictionary with:
        - upside_capture: Upside capture ratio (%)
        - downside_capture: Downside capture ratio (%)
        - up_months: Number of up months (benchmark positive)
        - down_months: Number of down months (benchmark negative)

    Example:
        >>> # Portfolio gains 12% when market gains 10% -> upside capture = 120%
        >>> # Portfolio loses 8% when market loses 10% -> downside capture = 80%
    """
    # Align series
    aligned = pd.DataFrame({
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }).dropna()

    if len(aligned) == 0:
        return {
            "upside_capture": None,
            "downside_capture": None,
            "up_months": 0,
            "down_months": 0
        }

    # Resample to specified frequency if needed
    if frequency != 'D':
        aligned = aligned.resample(frequency).apply(lambda x: (1 + x).prod() - 1)

    # Split into up and down periods
    up_periods = aligned[aligned['benchmark'] > 0]
    down_periods = aligned[aligned['benchmark'] < 0]

    # Calculate capture ratios
    upside_capture = None
    if len(up_periods) > 0:
        port_up_avg = up_periods['portfolio'].mean()
        bench_up_avg = up_periods['benchmark'].mean()
        if bench_up_avg != 0:
            upside_capture = (port_up_avg / bench_up_avg) * 100

    downside_capture = None
    if len(down_periods) > 0:
        port_down_avg = down_periods['portfolio'].mean()
        bench_down_avg = down_periods['benchmark'].mean()
        if bench_down_avg != 0:
            downside_capture = (port_down_avg / bench_down_avg) * 100

    return {
        "upside_capture": upside_capture,
        "downside_capture": downside_capture,
        "up_months": len(up_periods),
        "down_months": len(down_periods)
    }


def compute_all_drawdown_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None
) -> Dict[str, any]:
    """
    Compute all drawdown metrics in one call

    Args:
        portfolio_returns: Portfolio return series
        benchmark_returns: Optional benchmark return series

    Returns:
        Dictionary combining results from:
        - compute_max_drawdown()
        - compute_current_drawdown()
        - compute_relative_drawdown() (if benchmark provided)
        - compute_capture_ratios() (if benchmark provided)
    """
    result = {}

    # Max drawdown
    max_dd = compute_max_drawdown(portfolio_returns)
    result.update({
        "max_drawdown": max_dd["max_drawdown"],
        "max_drawdown_start": max_dd["max_drawdown_start"],
        "max_drawdown_end": max_dd["max_drawdown_end"],
        "max_drawdown_duration_days": max_dd["max_drawdown_duration_days"],
        "recovery_date": max_dd["recovery_date"]
    })

    # Current drawdown
    curr_dd = compute_current_drawdown(portfolio_returns)
    result.update({
        "current_drawdown": curr_dd["current_drawdown"],
        "current_drawdown_start": curr_dd["current_drawdown_start"],
        "current_drawdown_duration_days": curr_dd["current_drawdown_duration_days"]
    })

    # Relative metrics (if benchmark provided)
    if benchmark_returns is not None:
        rel_dd = compute_relative_drawdown(portfolio_returns, benchmark_returns)
        result.update({
            "relative_drawdown_max": rel_dd["relative_drawdown_max"],
            "relative_drawdown_current": rel_dd["relative_drawdown_current"]
        })

        capture = compute_capture_ratios(portfolio_returns, benchmark_returns)
        result.update({
            "upside_capture": capture["upside_capture"],
            "downside_capture": capture["downside_capture"],
            "up_months": capture["up_months"],
            "down_months": capture["down_months"]
        })

    return result


# Utility functions

def compute_calmar_ratio(
    returns: pd.Series,
    annualization_factor: int = 252
) -> Optional[float]:
    """
    Compute Calmar ratio (annualized return / max drawdown)

    Args:
        returns: Return series
        annualization_factor: 252 for daily, 12 for monthly

    Returns:
        Calmar ratio or None if max drawdown is zero
    """
    if len(returns) == 0:
        return None

    annualized_return = returns.mean() * annualization_factor
    max_dd_result = compute_max_drawdown(returns)
    max_dd = max_dd_result["max_drawdown"]

    if max_dd >= -0.001:  # No drawdown
        return None

    return annualized_return / abs(max_dd)


def compute_sterling_ratio(
    returns: pd.Series,
    annualization_factor: int = 252,
    top_n_drawdowns: int = 5
) -> Optional[float]:
    """
    Compute Sterling ratio (annualized return / avg of top N drawdowns)

    Args:
        returns: Return series
        annualization_factor: 252 for daily, 12 for monthly
        top_n_drawdowns: Number of top drawdowns to average

    Returns:
        Sterling ratio or None if insufficient data
    """
    if len(returns) < top_n_drawdowns:
        return None

    annualized_return = returns.mean() * annualization_factor

    # Find all local drawdown troughs
    drawdown = compute_drawdown_series(returns)

    # Simple approach: take top N worst drawdowns
    worst_drawdowns = drawdown.nsmallest(top_n_drawdowns)
    avg_dd = worst_drawdowns.mean()

    if avg_dd >= -0.001:
        return None

    return annualized_return / abs(avg_dd)
