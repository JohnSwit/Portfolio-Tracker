"""
Factor analysis and style attribution using ETF regression

Implements:
- Factor exposure estimation via ETF regression
- Style analysis (value/growth, size, momentum, quality)
- Factor contribution to returns
- Factor tilt identification
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from scipy import stats
from scipy.optimize import nnls
import logging

logger = logging.getLogger(__name__)


# Standard factor ETFs for regression
FACTOR_ETFS = {
    "Market": "SPY",      # S&P 500 (Market/Beta)
    "Size": "IWM",        # Russell 2000 (Small Cap)
    "Value": "IWD",       # Russell 1000 Value
    "Growth": "IWF",      # Russell 1000 Growth
    "Momentum": "MTUM",   # MSCI USA Momentum
    "Quality": "QUAL",    # MSCI USA Quality
    "Low_Vol": "USMV"     # MSCI USA Minimum Volatility
}


def estimate_factor_exposures_regression(
    portfolio_returns: pd.Series,
    factor_returns: Dict[str, pd.Series],
    include_constant: bool = True
) -> Dict[str, any]:
    """
    Estimate factor exposures using multiple regression

    Model:
        R_portfolio = alpha + beta_1 * F_1 + beta_2 * F_2 + ... + epsilon

    Args:
        portfolio_returns: Portfolio return series
        factor_returns: Dict mapping factor name -> return series
        include_constant: Whether to include intercept (alpha)

    Returns:
        Dictionary with:
        - exposures: Dict of factor -> beta coefficient
        - alpha: Intercept (if include_constant=True)
        - r_squared: R-squared of regression
        - t_stats: T-statistics for each factor
        - p_values: P-values for significance testing
        - residual_vol: Volatility of residuals (idiosyncratic risk)

    Example:
        >>> factor_returns = {"Market": spy_returns, "Size": iwm_returns}
        >>> result = estimate_factor_exposures_regression(port_returns, factor_returns)
        >>> result["exposures"]["Market"]
        0.95  # 95% market beta
    """
    if len(portfolio_returns) == 0:
        return {
            "exposures": {},
            "alpha": None,
            "r_squared": None,
            "t_stats": {},
            "p_values": {},
            "residual_vol": None
        }

    # Align all series
    df = pd.DataFrame({"portfolio": portfolio_returns})
    for factor_name, factor_ret in factor_returns.items():
        df[factor_name] = factor_ret

    df = df.dropna()

    if len(df) < len(factor_returns) + 2:  # Need enough observations
        logger.warning(f"Insufficient data for regression: {len(df)} observations")
        return {
            "exposures": {},
            "alpha": None,
            "r_squared": None,
            "t_stats": {},
            "p_values": {},
            "residual_vol": None
        }

    # Prepare regression
    y = df["portfolio"].values
    X = df[list(factor_returns.keys())].values

    # Fit regression using numpy
    if include_constant:
        # Add constant for alpha
        n = len(y)
        X_with_const = np.column_stack([np.ones(n), X])

        # Ordinary least squares: beta = (X'X)^-1 X'y
        try:
            coefficients = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix in regression")
            return {
                "exposures": {},
                "alpha": None,
                "r_squared": None,
                "t_stats": {},
                "p_values": {},
                "residual_vol": None
            }

        alpha = coefficients[0]
        betas = coefficients[1:]

        # Calculate R-squared
        y_pred = X_with_const @ coefficients
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Calculate residuals
        residuals = y - y_pred
        residual_vol = np.std(residuals)

        # Calculate t-stats and p-values
        n = len(y)
        k = len(factor_returns)
        mse = np.sum(residuals ** 2) / (n - k - 1)

        # Standard errors
        X_with_const = np.column_stack([np.ones(n), X])
        try:
            var_covar = mse * np.linalg.inv(X_with_const.T @ X_with_const)
            std_errors = np.sqrt(np.diag(var_covar))

            # T-stats
            t_stat_alpha = alpha / std_errors[0] if std_errors[0] > 0 else 0
            t_stats_betas = betas / std_errors[1:] if len(std_errors) > 1 else []

            # P-values (two-tailed)
            p_value_alpha = 2 * (1 - stats.t.cdf(abs(t_stat_alpha), n - k - 1))
            p_values_betas = [2 * (1 - stats.t.cdf(abs(t), n - k - 1)) for t in t_stats_betas]

        except np.linalg.LinAlgError:
            logger.warning("Could not compute standard errors (singular matrix)")
            t_stats_betas = [0] * len(betas)
            p_values_betas = [1] * len(betas)
            p_value_alpha = 1

    else:
        # No constant
        try:
            betas = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix in regression")
            return {
                "exposures": {},
                "alpha": None,
                "r_squared": None,
                "t_stats": {},
                "p_values": {},
                "residual_vol": None
            }

        alpha = 0

        # Calculate R-squared
        y_pred = X @ betas
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        residuals = y - y_pred
        residual_vol = np.std(residuals)

        t_stats_betas = [0] * len(betas)
        p_values_betas = [1] * len(betas)
        p_value_alpha = 1

    # Package results
    exposures = {}
    t_stats = {}
    p_values = {}

    for i, factor_name in enumerate(factor_returns.keys()):
        exposures[factor_name] = betas[i]
        t_stats[factor_name] = t_stats_betas[i] if i < len(t_stats_betas) else 0
        p_values[factor_name] = p_values_betas[i] if i < len(p_values_betas) else 1

    return {
        "exposures": exposures,
        "alpha": alpha,
        "alpha_annualized": alpha * 252 if alpha is not None else None,
        "r_squared": r_squared,
        "t_stats": t_stats,
        "p_values": p_values,
        "residual_vol": residual_vol,
        "residual_vol_annualized": residual_vol * np.sqrt(252) if residual_vol else None,
        "num_observations": len(df)
    }


def compute_sharpe_style_weights(
    portfolio_returns: pd.Series,
    factor_returns: Dict[str, pd.Series]
) -> Dict[str, float]:
    """
    Compute Sharpe style weights (constrained regression)

    Uses non-negative least squares to ensure weights sum to 1
    and are all non-negative (true style analysis).

    Args:
        portfolio_returns: Portfolio return series
        factor_returns: Dict of factor name -> return series

    Returns:
        Dictionary of factor -> weight (sums to ~1.0)

    Note:
        This is Sharpe's returns-based style analysis.
        Weights represent portfolio's effective style exposure.
    """
    # Align data
    df = pd.DataFrame({"portfolio": portfolio_returns})
    for factor_name, factor_ret in factor_returns.items():
        df[factor_name] = factor_ret

    df = df.dropna()

    if len(df) < len(factor_returns) + 2:
        logger.warning("Insufficient data for style analysis")
        return {name: 0.0 for name in factor_returns.keys()}

    # Prepare data
    y = df["portfolio"].values
    X = df[list(factor_returns.keys())].values

    # Non-negative least squares with sum-to-one constraint
    from scipy.optimize import nnls

    # NNLS doesn't enforce sum-to-one, so we'll use a simple approach
    # Unconstrained NNLS
    weights, residual = nnls(X, y)

    # Normalize to sum to 1
    total_weight = np.sum(weights)
    if total_weight > 0:
        weights = weights / total_weight
    else:
        # Fallback: equal weight
        weights = np.ones(len(factor_returns)) / len(factor_returns)

    # Package results
    style_weights = {}
    for i, factor_name in enumerate(factor_returns.keys()):
        style_weights[factor_name] = weights[i]

    return style_weights


def identify_factor_tilts(
    exposures: Dict[str, float],
    benchmark_exposures: Optional[Dict[str, float]] = None,
    significance_threshold: float = 0.10
) -> Dict[str, any]:
    """
    Identify factor tilts relative to benchmark

    A tilt is significant if |exposure - benchmark| > threshold

    Args:
        exposures: Portfolio factor exposures
        benchmark_exposures: Benchmark exposures (default: market-neutral)
        significance_threshold: Minimum deviation to be considered a tilt

    Returns:
        Dictionary with:
        - tilts: Dict of factor -> tilt amount
        - significant_tilts: List of factors with significant tilts
        - tilt_direction: Dict of factor -> "overweight" or "underweight"

    Example:
        >>> exposures = {"Value": 0.30, "Growth": 0.10}
        >>> benchmark = {"Value": 0.15, "Growth": 0.15}
        >>> tilts = identify_factor_tilts(exposures, benchmark)
        >>> tilts["significant_tilts"]
        ["Value"]  # 0.30 - 0.15 = 0.15 tilt
    """
    if benchmark_exposures is None:
        # Default benchmark: market beta = 1, all others = 0
        benchmark_exposures = {factor: 0.0 for factor in exposures.keys()}
        if "Market" in benchmark_exposures:
            benchmark_exposures["Market"] = 1.0

    tilts = {}
    significant_tilts = []
    tilt_direction = {}

    for factor, exposure in exposures.items():
        benchmark_exp = benchmark_exposures.get(factor, 0.0)
        tilt = exposure - benchmark_exp

        tilts[factor] = tilt

        if abs(tilt) > significance_threshold:
            significant_tilts.append(factor)
            tilt_direction[factor] = "overweight" if tilt > 0 else "underweight"

    return {
        "tilts": tilts,
        "significant_tilts": significant_tilts,
        "tilt_direction": tilt_direction
    }


def compute_factor_contributions(
    exposures: Dict[str, float],
    factor_returns: Dict[str, pd.Series],
    period: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute factor contributions to portfolio return

    Contribution = exposure * factor_return

    Args:
        exposures: Factor exposures (betas)
        factor_returns: Factor return series
        period: Optional period to compute over ('1M', '3M', '1Y')
                If None, uses full period

    Returns:
        Dictionary mapping factor -> contribution to return

    Example:
        >>> exposures = {"Market": 0.95, "Value": 0.20}
        >>> # If market returned 10% and value returned 5%
        >>> contributions = compute_factor_contributions(exposures, factor_rets)
        >>> contributions["Market"]
        0.095  # 0.95 * 0.10 = 9.5%
    """
    contributions = {}

    for factor, beta in exposures.items():
        if factor not in factor_returns:
            contributions[factor] = 0.0
            continue

        ret_series = factor_returns[factor]

        if period:
            # Filter to period
            # This is simplified - in production, properly handle period slicing
            ret_series = ret_series.tail(
                {"1M": 21, "3M": 63, "1Y": 252}.get(period, len(ret_series))
            )

        # Total return over period
        total_return = (1 + ret_series).prod() - 1

        # Contribution = beta * factor_return
        contribution = beta * total_return

        contributions[factor] = contribution

    return contributions


def analyze_style_drift(
    historical_exposures: pd.DataFrame,
    lookback_window: int = 252
) -> Dict[str, any]:
    """
    Analyze style drift over time

    Args:
        historical_exposures: DataFrame with columns = factors, rows = dates
        lookback_window: Window for computing drift

    Returns:
        Dictionary with:
        - current_exposures: Most recent exposures
        - avg_exposures: Average exposures over window
        - exposure_volatility: Std dev of exposures
        - drift_magnitude: Sum of absolute changes
        - drifting_factors: Factors with high volatility

    Note:
        High exposure volatility indicates style drift
    """
    if len(historical_exposures) < lookback_window:
        lookback_window = len(historical_exposures)

    recent_data = historical_exposures.tail(lookback_window)

    current_exposures = historical_exposures.iloc[-1].to_dict()
    avg_exposures = recent_data.mean().to_dict()
    exposure_volatility = recent_data.std().to_dict()

    # Drift magnitude: sum of absolute changes
    changes = recent_data.diff().abs()
    drift_magnitude = changes.sum().to_dict()

    # Identify drifting factors (high volatility)
    drifting_factors = []
    drift_threshold = 0.10  # 10% std dev

    for factor, vol in exposure_volatility.items():
        if vol > drift_threshold:
            drifting_factors.append(factor)

    return {
        "current_exposures": current_exposures,
        "avg_exposures": avg_exposures,
        "exposure_volatility": exposure_volatility,
        "drift_magnitude": drift_magnitude,
        "drifting_factors": drifting_factors
    }


def decompose_return_by_factors(
    portfolio_return: float,
    exposures: Dict[str, float],
    factor_returns: Dict[str, float],
    alpha: float = 0.0
) -> Dict[str, any]:
    """
    Decompose portfolio return into factor contributions + selection

    Total Return = Alpha + Sum(Beta_i * Factor_Return_i) + Residual

    Args:
        portfolio_return: Realized portfolio return
        exposures: Factor exposures (betas)
        factor_returns: Realized factor returns
        alpha: Alpha from regression

    Returns:
        Dictionary with:
        - alpha_contribution: Alpha
        - factor_contributions: Dict of factor -> contribution
        - total_factor_contribution: Sum of all factor contributions
        - residual: Unexplained return (selection/timing)

    Note:
        Residual = Total Return - Alpha - Sum(Factor Contributions)
    """
    factor_contributions = {}
    total_factor = 0.0

    for factor, beta in exposures.items():
        factor_ret = factor_returns.get(factor, 0.0)
        contribution = beta * factor_ret
        factor_contributions[factor] = contribution
        total_factor += contribution

    residual = portfolio_return - alpha - total_factor

    return {
        "total_return": portfolio_return,
        "alpha_contribution": alpha,
        "factor_contributions": factor_contributions,
        "total_factor_contribution": total_factor,
        "residual": residual,
        "explained_return": alpha + total_factor,
        "pct_explained": abs((alpha + total_factor) / portfolio_return * 100) if portfolio_return != 0 else None
    }


# Utility functions

def get_factor_etf_tickers() -> Dict[str, str]:
    """
    Get standard factor ETF tickers

    Returns:
        Dictionary mapping factor name -> ticker
    """
    return FACTOR_ETFS.copy()


def validate_factor_model(
    r_squared: float,
    residual_vol: float,
    min_r_squared: float = 0.70,
    max_residual_vol: float = 0.10
) -> Dict[str, any]:
    """
    Validate quality of factor model fit

    Args:
        r_squared: R-squared from regression
        residual_vol: Annualized residual volatility
        min_r_squared: Minimum acceptable R-squared
        max_residual_vol: Maximum acceptable residual vol

    Returns:
        Dictionary with:
        - is_valid: Boolean indicating if model meets criteria
        - r_squared_ok: Boolean for R-squared check
        - residual_vol_ok: Boolean for residual vol check
        - quality_score: Overall quality (0-100)
    """
    r_squared_ok = r_squared >= min_r_squared if r_squared is not None else False
    residual_vol_ok = residual_vol <= max_residual_vol if residual_vol is not None else False

    is_valid = r_squared_ok and residual_vol_ok

    # Quality score: weighted average
    quality = 0
    if r_squared is not None:
        quality += (r_squared / min_r_squared) * 50
    if residual_vol is not None:
        quality += (1 - min(residual_vol / max_residual_vol, 1)) * 50

    quality = min(quality, 100)

    return {
        "is_valid": is_valid,
        "r_squared_ok": r_squared_ok,
        "residual_vol_ok": residual_vol_ok,
        "quality_score": quality,
        "r_squared": r_squared,
        "residual_vol": residual_vol
    }
