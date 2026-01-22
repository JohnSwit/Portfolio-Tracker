"""
Unit tests for tracking error calculations
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Import tracking_error module directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "tracking_error",
    os.path.join(os.path.dirname(__file__), '..', 'backend', 'services', 'tracking_error.py')
)
tracking_error = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracking_error)


class TestActiveReturns:
    """Test active return calculations"""

    def test_compute_active_returns(self):
        """Test basic active return calculation"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')

        port_returns = pd.Series([0.01] * 100, index=dates)
        bench_returns = pd.Series([0.008] * 100, index=dates)

        active = tracking_error.compute_active_returns(port_returns, bench_returns)

        assert len(active) == 100
        assert abs(active.mean() - 0.002) < 0.0001  # 0.01 - 0.008

    def test_compute_active_returns_misaligned(self):
        """Test active returns with misaligned dates"""
        dates1 = pd.date_range('2024-01-01', periods=100, freq='D')
        dates2 = pd.date_range('2024-01-05', periods=100, freq='D')  # Offset start

        port_returns = pd.Series([0.01] * 100, index=dates1)
        bench_returns = pd.Series([0.008] * 100, index=dates2)

        active = tracking_error.compute_active_returns(port_returns, bench_returns)

        # Should only include overlapping dates
        assert len(active) < 100


class TestTrackingError:
    """Test tracking error calculations"""

    def test_compute_tracking_error_zero(self):
        """Test TE when active returns are zero (perfect tracking)"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        active_returns = pd.Series([0.0] * 100, index=dates)

        te = tracking_error.compute_tracking_error(active_returns)

        assert abs(te) < 0.0001  # Should be near zero

    def test_compute_tracking_error_known_value(self):
        """Test TE with known standard deviation"""
        # Daily active returns with std = 0.01 (1%)
        # Annualized TE = 0.01 * sqrt(252) ≈ 0.1587 (15.87%)

        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        active_returns = pd.Series(np.random.normal(0, 0.01, 252), index=dates)

        te = tracking_error.compute_tracking_error(active_returns)

        # Should be approximately 15-16% (with some random variation)
        assert 0.12 < te < 0.20  # Reasonable range given randomness

    def test_compute_tracking_error_insufficient_data(self):
        """Test TE with insufficient data (logs warning)"""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        active_returns = pd.Series([0.001] * 10, index=dates)

        # Should still compute but log warning
        te = tracking_error.compute_tracking_error(active_returns)

        assert not np.isnan(te)


class TestInformationRatio:
    """Test information ratio calculations"""

    def test_compute_information_ratio(self):
        """Test basic IR calculation"""
        # Mean daily active return = 0.001 (0.1%)
        # Annualized = 0.001 * 252 = 0.252 (25.2%)
        # Daily std = 0.01
        # Annualized TE = 0.01 * sqrt(252) ≈ 0.1587
        # IR = 0.252 / 0.1587 ≈ 1.59

        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        active_returns = pd.Series(np.random.normal(0.001, 0.01, 252), index=dates)

        ir = tracking_error.compute_information_ratio(active_returns)

        # IR should be positive and reasonable
        assert ir is not None
        assert 0.5 < ir < 3.0  # Typical range for good active managers

    def test_compute_information_ratio_zero_te(self):
        """Test IR when TE is zero (should return None)"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        active_returns = pd.Series([0.0] * 100, index=dates)

        ir = tracking_error.compute_information_ratio(active_returns)

        assert ir is None  # IR not meaningful when TE ≈ 0


class TestRollingTE:
    """Test rolling tracking error"""

    def test_compute_rolling_tracking_error(self):
        """Test rolling TE calculation"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=500, freq='D')

        # First half: low volatility, second half: high volatility
        active_returns_1 = np.random.normal(0, 0.005, 250)
        active_returns_2 = np.random.normal(0, 0.015, 250)
        active_returns = pd.Series(
            np.concatenate([active_returns_1, active_returns_2]),
            index=dates
        )

        rolling_te = tracking_error.compute_rolling_tracking_error(
            active_returns, window_days=252
        )

        # Rolling TE should increase in second half
        first_half_mean = rolling_te.iloc[252:350].mean()
        second_half_mean = rolling_te.iloc[350:].mean()

        assert second_half_mean > first_half_mean


class TestAlignmentAndValidation:
    """Test utility functions"""

    def test_align_return_series(self):
        """Test return series alignment"""
        dates1 = pd.date_range('2024-01-01', periods=100, freq='D')
        dates2 = pd.date_range('2024-01-05', periods=100, freq='D')

        port_returns = pd.Series(range(100), index=dates1)
        bench_returns = pd.Series(range(100), index=dates2)

        aligned_port, aligned_bench = tracking_error.align_return_series(
            port_returns, bench_returns
        )

        # Should have same length
        assert len(aligned_port) == len(aligned_bench)
        # Should be shorter than originals
        assert len(aligned_port) < 100

    def test_validate_return_series_valid(self):
        """Test validation with valid return series"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        returns = pd.Series(np.random.normal(0, 0.01, 100), index=dates)

        assert tracking_error.validate_return_series(returns) is True

    def test_validate_return_series_too_short(self):
        """Test validation with series too short"""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        returns = pd.Series(np.random.normal(0, 0.01, 10), index=dates)

        assert tracking_error.validate_return_series(returns) is False

    def test_validate_return_series_outlier(self):
        """Test validation with unrealistic return"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        returns = pd.Series([0.01] * 100, index=dates)
        returns.iloc[50] = 2.0  # 200% single-day return (unrealistic)

        assert tracking_error.validate_return_series(returns) is False


class TestTEComponents:
    """Test TE decomposition"""

    def test_analyze_tracking_error_components(self):
        """Test TE decomposition into systematic and idiosyncratic"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=252, freq='D')

        # Portfolio partially tracks benchmark + some idiosyncratic risk
        bench_returns = pd.Series(np.random.normal(0.0005, 0.01, 252), index=dates)
        idio_returns = pd.Series(np.random.normal(0, 0.005, 252), index=dates)

        # Portfolio = 0.9 * benchmark + idiosyncratic
        port_returns = 0.9 * bench_returns + idio_returns

        active_returns = port_returns - bench_returns

        components = tracking_error.analyze_tracking_error_components(
            active_returns, port_returns, bench_returns
        )

        assert 'total_te' in components
        assert 'systematic_te' in components
        assert 'idiosyncratic_te' in components
        assert 'r_squared' in components

        # R-squared should be high (portfolio tracks benchmark closely)
        assert components['r_squared'] > 0.5

        # Total TE should be approximately sum of components
        # (Actually: total^2 ≈ systematic^2 + idiosyncratic^2)
        total_sq = components['total_te'] ** 2
        sys_sq = components['systematic_te'] ** 2
        idio_sq = components['idiosyncratic_te'] ** 2

        assert abs(total_sq - (sys_sq + idio_sq)) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
