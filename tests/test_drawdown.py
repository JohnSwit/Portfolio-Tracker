"""
Unit tests for drawdown analysis and capture ratios
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Import drawdown module directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "drawdown",
    os.path.join(os.path.dirname(__file__), '..', 'backend', 'services', 'drawdown.py')
)
drawdown = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drawdown)


class TestDrawdownSeries:
    """Test drawdown series calculation"""

    def test_compute_drawdown_series_simple(self):
        """Test basic drawdown series"""
        # Returns: +10%, +5%, -20%, +10%
        returns = pd.Series([0.10, 0.05, -0.20, 0.10])

        dd = drawdown.compute_drawdown_series(returns)

        # Cumulative: 1.10, 1.155, 0.924, 1.0164
        # Running max: 1.10, 1.155, 1.155, 1.155
        # Drawdown: 0, 0, -0.20, -0.12

        assert len(dd) == 4
        assert abs(dd.iloc[0]) < 0.001  # No drawdown at start
        assert abs(dd.iloc[1]) < 0.001  # Still at peak
        assert dd.iloc[2] < -0.19  # In drawdown after -20%
        assert dd.iloc[3] > dd.iloc[2]  # Recovering (less negative drawdown)

    def test_compute_drawdown_series_no_drawdown(self):
        """Test with all positive returns (no drawdown)"""
        returns = pd.Series([0.01, 0.02, 0.03, 0.01])

        dd = drawdown.compute_drawdown_series(returns)

        # Should be all zeros (or very close)
        assert all(dd >= -0.001)

    def test_compute_drawdown_series_empty(self):
        """Test with empty series"""
        returns = pd.Series(dtype=float)

        dd = drawdown.compute_drawdown_series(returns)

        assert len(dd) == 0


class TestMaxDrawdown:
    """Test maximum drawdown calculations"""

    def test_compute_max_drawdown_simple(self):
        """Test max drawdown with known scenario"""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        # Pattern: up, up, crash, down, recover
        returns = pd.Series([0.05, 0.05, -0.20, -0.10, 0.05, 0.05, 0.10, 0.05, 0.05, 0.05], index=dates)

        result = drawdown.compute_max_drawdown(returns)

        assert result["max_drawdown"] is not None
        assert result["max_drawdown"] < -0.25  # Significant drawdown
        assert result["max_drawdown_start"] is not None
        assert result["max_drawdown_end"] is not None
        assert result["max_drawdown_duration_days"] is not None

    def test_compute_max_drawdown_recovery(self):
        """Test max drawdown with recovery"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        # Early drawdown that fully recovers
        returns_list = [0.01] * 20 + [-0.03] * 10 + [0.02] * 70
        returns = pd.Series(returns_list, index=dates)

        result = drawdown.compute_max_drawdown(returns)

        # Should detect recovery
        assert result["recovery_date"] is not None
        assert result["recovery_date"] > result["max_drawdown_end"]

    def test_compute_max_drawdown_no_recovery(self):
        """Test max drawdown without recovery"""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        # Drawdown with no recovery
        returns_list = [0.02] * 10 + [-0.02] * 40
        returns = pd.Series(returns_list, index=dates)

        result = drawdown.compute_max_drawdown(returns)

        # Should have no recovery date
        assert result["recovery_date"] is None or result["recovery_date"] >= dates[-1]


class TestCurrentDrawdown:
    """Test current drawdown calculations"""

    def test_compute_current_drawdown_in_drawdown(self):
        """Test current drawdown when in drawdown"""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        # Peak at day 20, then decline
        returns_list = [0.02] * 20 + [-0.01] * 30
        returns = pd.Series(returns_list, index=dates)

        result = drawdown.compute_current_drawdown(returns)

        assert result["current_drawdown"] < -0.01  # In drawdown
        assert result["current_drawdown_start"] is not None
        assert result["current_drawdown_duration_days"] is not None
        assert result["current_drawdown_duration_days"] > 0

    def test_compute_current_drawdown_at_peak(self):
        """Test current drawdown when at peak"""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        returns = pd.Series([0.01] * 50, index=dates)

        result = drawdown.compute_current_drawdown(returns)

        # Should be at or near zero
        assert result["current_drawdown"] >= -0.01
        assert result["current_drawdown_duration_days"] is None or result["current_drawdown_duration_days"] == 0


class TestRelativeDrawdown:
    """Test relative drawdown vs benchmark"""

    def test_compute_relative_drawdown_underperformance(self):
        """Test relative drawdown when underperforming benchmark"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')

        # Portfolio lags benchmark
        port_returns = pd.Series([0.005] * 100, index=dates)
        bench_returns = pd.Series([0.01] * 100, index=dates)

        result = drawdown.compute_relative_drawdown(port_returns, bench_returns)

        # Should show underperformance (negative relative drawdown)
        assert result["relative_drawdown_max"] < -0.01
        assert result["relative_drawdown_current"] < 0

    def test_compute_relative_drawdown_outperformance(self):
        """Test relative drawdown when outperforming benchmark"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')

        # Portfolio beats benchmark
        port_returns = pd.Series([0.015] * 100, index=dates)
        bench_returns = pd.Series([0.01] * 100, index=dates)

        result = drawdown.compute_relative_drawdown(port_returns, bench_returns)

        # No relative drawdown (continuous outperformance)
        # Max relative drawdown should be close to zero
        assert result["relative_drawdown_max"] > -0.01


class TestCaptureRatios:
    """Test capture ratio calculations"""

    def test_compute_capture_ratios_symmetric(self):
        """Test capture ratios when portfolio tracks benchmark"""
        dates = pd.date_range('2024-01-01', periods=252, freq='D')

        # Portfolio = benchmark
        np.random.seed(42)
        bench_returns = pd.Series(np.random.normal(0.001, 0.01, 252), index=dates)
        port_returns = bench_returns.copy()

        result = drawdown.compute_capture_ratios(port_returns, bench_returns)

        # Both captures should be close to 100%
        assert result["upside_capture"] is not None
        assert result["downside_capture"] is not None
        assert 95 < result["upside_capture"] < 105
        assert 95 < result["downside_capture"] < 105

    def test_compute_capture_ratios_high_upside(self):
        """Test capture ratios with higher upside capture"""
        dates = pd.date_range('2024-01-01', periods=252, freq='D')

        np.random.seed(42)
        bench_returns = pd.Series(np.random.normal(0.001, 0.01, 252), index=dates)

        # Portfolio amplifies returns (1.2x)
        port_returns = bench_returns * 1.2

        result = drawdown.compute_capture_ratios(port_returns, bench_returns)

        # Both should be around 120% (amplified equally)
        assert result["upside_capture"] > 100
        assert result["downside_capture"] > 100

    def test_compute_capture_ratios_asymmetric(self):
        """Test capture ratios with asymmetric performance"""
        dates = pd.date_range('2024-01-01', periods=24, freq='MS')

        # Benchmark: alternating up/down months
        bench_returns = pd.Series([0.05 if i % 2 == 0 else -0.03 for i in range(24)], index=dates)

        # Portfolio: captures more upside, less downside
        port_returns = pd.Series([0.06 if i % 2 == 0 else -0.02 for i in range(24)], index=dates)

        result = drawdown.compute_capture_ratios(port_returns, bench_returns, frequency='M')

        # Upside capture should be > 100%, downside < 100%
        assert result["upside_capture"] > 100
        assert result["downside_capture"] < 100
        assert result["up_months"] == 12
        assert result["down_months"] == 12


class TestDrawdownMetricsCombined:
    """Test combined drawdown metrics"""

    def test_compute_all_drawdown_metrics_full(self):
        """Test computing all metrics at once with benchmark"""
        dates = pd.date_range('2024-01-01', periods=252, freq='D')

        np.random.seed(42)
        port_returns = pd.Series(np.random.normal(0.001, 0.015, 252), index=dates)
        bench_returns = pd.Series(np.random.normal(0.001, 0.01, 252), index=dates)

        result = drawdown.compute_all_drawdown_metrics(port_returns, bench_returns)

        # Should have all keys
        assert "max_drawdown" in result
        assert "current_drawdown" in result
        assert "relative_drawdown_max" in result
        assert "upside_capture" in result
        assert "downside_capture" in result

    def test_compute_all_drawdown_metrics_no_benchmark(self):
        """Test computing metrics without benchmark"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        returns = pd.Series(np.random.normal(0.001, 0.01, 100), index=dates)

        result = drawdown.compute_all_drawdown_metrics(returns)

        # Should have basic metrics but not relative ones
        assert "max_drawdown" in result
        assert "current_drawdown" in result
        assert "relative_drawdown_max" not in result
        assert "upside_capture" not in result


class TestRiskAdjustedRatios:
    """Test risk-adjusted ratios"""

    def test_compute_calmar_ratio(self):
        """Test Calmar ratio calculation"""
        dates = pd.date_range('2024-01-01', periods=252, freq='D')

        # Positive average return with some drawdown
        returns = pd.Series([0.005] * 200 + [-0.01] * 52, index=dates)

        calmar = drawdown.compute_calmar_ratio(returns, annualization_factor=252)

        assert calmar is not None
        assert calmar > 0  # Positive return divided by positive max DD

    def test_compute_calmar_ratio_no_drawdown(self):
        """Test Calmar ratio with no drawdown"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        returns = pd.Series([0.01] * 100, index=dates)

        calmar = drawdown.compute_calmar_ratio(returns)

        # Should return None (undefined when no drawdown)
        assert calmar is None

    def test_compute_sterling_ratio(self):
        """Test Sterling ratio calculation"""
        dates = pd.date_range('2024-01-01', periods=500, freq='D')

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 500), index=dates)

        sterling = drawdown.compute_sterling_ratio(returns, annualization_factor=252, top_n_drawdowns=5)

        assert sterling is not None
        # Should be a reasonable value
        assert -10 < sterling < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
