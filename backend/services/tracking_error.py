"""
Tracking error and active risk calculations

Implements:
- Realized tracking error (ex-post)
- Active return series
- Contribution to active risk (CTAR)
- Information ratio
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def compute_active_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> pd.Series:
    """
    Compute active return series (portfolio - benchmark)

    Args:
        portfolio_returns: Daily portfolio returns (time series)
        benchmark_returns: Daily benchmark returns (time series)

    Returns:
        Series of active returns (aligned dates)

    Note:
        Uses inner join on dates to handle different trading calendars
    """
    # Align on common dates
    aligned = pd.DataFrame({
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }).dropna()

    active_returns = aligned['portfolio'] - aligned['benchmark']

    return active_returns


def compute_tracking_error(
    active_returns: pd.Series,
    annualization_factor: int = 252
) -> float:
    """
    Compute realized tracking error (annualized)

    TE = std(active_returns_daily) * sqrt(annualization_factor)

    Args:
        active_returns: Daily active return series
        annualization_factor: Days per year (default 252)

    Returns:
        Annualized tracking error (as decimal, e.g., 0.05 = 5%)

    Note:
        Requires at least 30 days of data for meaningful calculation
    """
    if len(active_returns) < 30:
        logger.warning(f"Only {len(active_returns)} days of data for TE calculation (min 30 recommended)")

    if len(active_returns) == 0:
        return np.nan

    # Standard deviation of daily active returns
    te_daily = active_returns.std()

    # Annualize
    te_annual = te_daily * np.sqrt(annualization_factor)

    return float(te_annual)


def compute_information_ratio(
    active_returns: pd.Series,
    annualization_factor: int = 252
) -> Optional[float]:
    """
    Compute Information Ratio

    IR = mean(active_returns) * annualization_factor / TE

    Args:
        active_returns: Daily active return series
        annualization_factor: Days per year

    Returns:
        Information ratio (scalar)

    Note:
        Returns None if TE is zero or near-zero
    """
    if len(active_returns) == 0:
        return None

    # Annualized active return
    mean_daily_active = active_returns.mean()
    annual_active_return = mean_daily_active * annualization_factor

    # Tracking error
    te = compute_tracking_error(active_returns, annualization_factor)

    if te < 0.0001:  # Near-zero TE (index replication)
        logger.info("TE near zero, IR not meaningful")
        return None

    ir = annual_active_return / te

    return float(ir)


def compute_rolling_tracking_error(
    active_returns: pd.Series,
    window_days: int = 252,
    annualization_factor: int = 252
) -> pd.Series:
    """
    Compute rolling tracking error

    Args:
        active_returns: Daily active return series
        window_days: Rolling window size (default 252 = 1 year)
        annualization_factor: Days per year

    Returns:
        Series of rolling TE values
    """
    rolling_std = active_returns.rolling(window=window_days, min_periods=60).std()
    rolling_te = rolling_std * np.sqrt(annualization_factor)

    return rolling_te


def compute_contribution_to_active_risk_simple(
    portfolio_returns_by_position: pd.DataFrame,
    portfolio_weights: Dict[str, float],
    benchmark_weights: Dict[str, float],
    total_tracking_error: float
) -> Dict[str, float]:
    """
    Compute contribution to active risk (simplified approach)

    Uses marginal contribution approximation:
    CTAR_i ≈ active_weight_i * (std_dev_i / TE) * correlation_with_active_returns

    Args:
        portfolio_returns_by_position: DataFrame of returns by position (columns = symbols)
        portfolio_weights: Current portfolio weights
        benchmark_weights: Current benchmark weights
        total_tracking_error: Portfolio tracking error

    Returns:
        Dictionary mapping symbol -> contribution to TE

    Note:
        This is an approximation. Full CTAR requires covariance matrix of active positions.
        Contributions may not sum exactly to TE due to correlation effects.
    """
    if total_tracking_error < 0.0001:
        logger.warning("TE near zero, CTAR not meaningful")
        return {symbol: 0.0 for symbol in portfolio_weights.keys()}

    contributions = {}

    for symbol in portfolio_weights.keys():
        active_weight = portfolio_weights.get(symbol, 0.0) - benchmark_weights.get(symbol, 0.0)

        if symbol in portfolio_returns_by_position.columns:
            returns = portfolio_returns_by_position[symbol].dropna()

            if len(returns) > 30:
                # Volatility of this position
                vol = returns.std() * np.sqrt(252)

                # Approximate contribution
                # This is simplified; full calculation would use covariance matrix
                marginal_contrib = active_weight * vol

                contributions[symbol] = marginal_contrib
            else:
                contributions[symbol] = 0.0
        else:
            contributions[symbol] = 0.0

    return contributions


def analyze_tracking_error_components(
    active_returns: pd.Series,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> Dict[str, float]:
    """
    Analyze tracking error components

    Decomposes TE into:
    - Systematic: Part explained by market movements
    - Idiosyncratic: Residual/specific risk

    Args:
        active_returns: Active return series
        portfolio_returns: Portfolio return series
        benchmark_returns: Benchmark return series

    Returns:
        Dictionary with TE components
    """
    # Align data
    data = pd.DataFrame({
        'active': active_returns,
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }).dropna()

    if len(data) < 30:
        logger.warning("Insufficient data for TE decomposition")
        return {
            'total_te': np.nan,
            'systematic_te': np.nan,
            'idiosyncratic_te': np.nan
        }

    # Total TE
    total_te = data['active'].std() * np.sqrt(252)

    # Simple regression of portfolio returns on benchmark returns
    # Residuals represent idiosyncratic risk
    from scipy import stats

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        data['benchmark'], data['portfolio']
    )

    # Predicted portfolio returns (systematic component)
    predicted = slope * data['benchmark'] + intercept

    # Residuals (idiosyncratic component)
    residuals = data['portfolio'] - predicted

    # Idiosyncratic TE
    idiosyncratic_te = residuals.std() * np.sqrt(252)

    # Systematic TE (approximation)
    systematic_te = np.sqrt(max(0, total_te**2 - idiosyncratic_te**2))

    return {
        'total_te': float(total_te),
        'systematic_te': float(systematic_te),
        'idiosyncratic_te': float(idiosyncratic_te),
        'r_squared': float(r_value ** 2)
    }


def compute_te_ex_ante(
    portfolio_weights: Dict[str, float],
    benchmark_weights: Dict[str, float],
    covariance_matrix: pd.DataFrame
) -> float:
    """
    Compute ex-ante (predicted) tracking error using covariance matrix

    TE^2 = (w_p - w_b)^T * Cov * (w_p - w_b)

    Args:
        portfolio_weights: Portfolio weights by symbol
        benchmark_weights: Benchmark weights by symbol
        covariance_matrix: Covariance matrix of returns (annualized)

    Returns:
        Ex-ante tracking error (annualized)

    Note:
        Requires covariance matrix estimation from historical returns
    """
    # Get active weights
    symbols = list(set(portfolio_weights.keys()) | set(benchmark_weights.keys()))

    active_weights = np.array([
        portfolio_weights.get(s, 0.0) - benchmark_weights.get(s, 0.0)
        for s in symbols
    ])

    # Ensure covariance matrix matches symbols
    try:
        cov_subset = covariance_matrix.loc[symbols, symbols]
    except KeyError:
        logger.error("Covariance matrix missing some symbols")
        return np.nan

    # TE^2 = w^T * Cov * w
    te_squared = active_weights.T @ cov_subset.values @ active_weights

    te = np.sqrt(max(0, te_squared))

    return float(te)


# Helper functions

def align_return_series(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Align portfolio and benchmark return series on common dates

    Args:
        portfolio_returns: Portfolio return series
        benchmark_returns: Benchmark return series

    Returns:
        Tuple of (aligned_portfolio, aligned_benchmark)
    """
    df = pd.DataFrame({
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }).dropna()

    return df['portfolio'], df['benchmark']


def validate_return_series(
    returns: pd.Series,
    min_length: int = 30,
    max_return: float = 1.0
) -> bool:
    """
    Validate return series for quality

    Args:
        returns: Return series to validate
        min_length: Minimum required length
        max_return: Maximum single-day return (to detect errors)

    Returns:
        True if valid, False otherwise
    """
    if len(returns) < min_length:
        logger.warning(f"Return series too short: {len(returns)} < {min_length}")
        return False

    if returns.max() > max_return:
        logger.warning(f"Unusually large return detected: {returns.max():.2%}")
        return False

    if returns.isna().sum() > len(returns) * 0.1:
        logger.warning(f"Too many missing values: {returns.isna().sum()}/{len(returns)}")
        return False

    return True
