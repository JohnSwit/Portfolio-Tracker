"""
Unit tests for core analytics module
"""

import pytest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from models.portfolio import Holding
from services.analytics import (
    compute_weights,
    compute_weights_by_sector,
    compute_active_weights,
    compute_active_share,
    compute_concentration_metrics,
    compute_sector_concentration_metrics,
    compute_position_sizes,
    compute_turnover,
    validate_weights,
    normalize_weights
)


class TestWeightCalculations:
    """Test portfolio weight calculations"""

    def test_compute_weights_simple(self):
        """Test basic weight calculation"""
        holdings = [
            Holding(symbol="AAPL", quantity=100, cost_basis=10000, market_value=15000),
            Holding(symbol="MSFT", quantity=50, cost_basis=5000, market_value=10000),
        ]

        weights = compute_weights(holdings)

        assert len(weights) == 2
        assert abs(weights["AAPL"] - 0.6) < 0.001  # 15000 / 25000
        assert abs(weights["MSFT"] - 0.4) < 0.001  # 10000 / 25000
        assert abs(sum(weights.values()) - 1.0) < 0.001

    def test_compute_weights_empty(self):
        """Test weights with empty holdings"""
        weights = compute_weights([])
        assert weights == {}

    def test_compute_weights_zero_value(self):
        """Test weights when total value is zero"""
        holdings = [
            Holding(symbol="AAPL", quantity=0, cost_basis=0, market_value=0),
        ]

        weights = compute_weights(holdings)
        assert weights["AAPL"] == 0.0

    def test_compute_weights_with_total_value(self):
        """Test providing explicit total value"""
        holdings = [
            Holding(symbol="AAPL", quantity=100, cost_basis=10000, market_value=15000),
        ]

        weights = compute_weights(holdings, total_value=30000)
        assert abs(weights["AAPL"] - 0.5) < 0.001  # 15000 / 30000


class TestSectorWeights:
    """Test sector-level weight calculations"""

    def test_compute_weights_by_sector(self):
        """Test sector weight grouping"""
        holdings = [
            Holding(symbol="AAPL", quantity=100, cost_basis=10000,
                   market_value=15000, sector="Technology"),
            Holding(symbol="MSFT", quantity=50, cost_basis=5000,
                   market_value=10000, sector="Technology"),
            Holding(symbol="JPM", quantity=30, cost_basis=3000,
                   market_value=5000, sector="Financials"),
        ]

        sector_weights = compute_weights_by_sector(holdings)

        assert len(sector_weights) == 2
        assert abs(sector_weights["Technology"] - 0.833) < 0.01  # (15000+10000)/30000
        assert abs(sector_weights["Financials"] - 0.167) < 0.01  # 5000/30000

    def test_compute_weights_by_sector_unknown(self):
        """Test handling of holdings without sector"""
        holdings = [
            Holding(symbol="AAPL", quantity=100, cost_basis=10000,
                   market_value=15000, sector="Technology"),
            Holding(symbol="XYZ", quantity=50, cost_basis=5000,
                   market_value=10000, sector=None),  # No sector
        ]

        sector_weights = compute_weights_by_sector(holdings)

        assert "Technology" in sector_weights
        assert "Unknown" in sector_weights
        assert abs(sector_weights["Technology"] - 0.6) < 0.001
        assert abs(sector_weights["Unknown"] - 0.4) < 0.001


class TestActiveWeights:
    """Test active weight and active share calculations"""

    def test_compute_active_weights(self):
        """Test active weight calculation"""
        port_weights = {"AAPL": 0.15, "MSFT": 0.10, "GOOGL": 0.05}
        bench_weights = {"AAPL": 0.10, "MSFT": 0.12, "GOOGL": 0.08, "AMZN": 0.05}

        active = compute_active_weights(port_weights, bench_weights)

        assert abs(active["AAPL"] - 0.05) < 0.001  # Overweight
        assert abs(active["MSFT"] - (-0.02)) < 0.001  # Underweight
        assert abs(active["GOOGL"] - (-0.03)) < 0.001  # Underweight
        assert abs(active["AMZN"] - (-0.05)) < 0.001  # Not held

    def test_compute_active_share_identical(self):
        """Test active share when portfolio matches benchmark"""
        weights = {"AAPL": 0.5, "MSFT": 0.5}

        active_share = compute_active_share(weights, weights)

        assert abs(active_share) < 0.001  # Should be 0%

    def test_compute_active_share_no_overlap(self):
        """Test active share with no overlap"""
        port_weights = {"AAPL": 0.5, "MSFT": 0.5}
        bench_weights = {"GOOGL": 0.5, "AMZN": 0.5}

        active_share = compute_active_share(port_weights, bench_weights)

        assert abs(active_share - 1.0) < 0.001  # Should be 100%

    def test_compute_active_share_example(self):
        """Test active share with known example"""
        # Portfolio: 60% AAPL, 40% MSFT
        # Benchmark: 50% AAPL, 30% MSFT, 20% GOOGL
        # Active weights: +10% AAPL, +10% MSFT, -20% GOOGL
        # Sum of abs = 40%, Active Share = 20%

        port_weights = {"AAPL": 0.6, "MSFT": 0.4}
        bench_weights = {"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2}

        active_share = compute_active_share(port_weights, bench_weights)

        assert abs(active_share - 0.2) < 0.001  # 20%


class TestConcentration:
    """Test concentration metrics"""

    def test_compute_concentration_metrics(self):
        """Test concentration calculations"""
        # Create 10 holdings with decreasing weights
        holdings = []
        values = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        for i, val in enumerate(values):
            holdings.append(
                Holding(symbol=f"SYM{i}", quantity=val,
                       cost_basis=val, market_value=val)
            )

        metrics = compute_concentration_metrics(holdings)

        total = sum(values)

        # Top 5 weight
        expected_top5 = sum(values[:5]) / total
        assert abs(metrics["top5_weight"] - expected_top5) < 0.001

        # Top 10 weight (all holdings)
        expected_top10 = 1.0
        assert abs(metrics["top10_weight"] - expected_top10) < 0.001

        # HHI
        expected_hhi = sum((v/total)**2 for v in values)
        assert abs(metrics["hhi_holdings"] - expected_hhi) < 0.001

        assert metrics["num_holdings"] == 10

    def test_compute_concentration_single_holding(self):
        """Test concentration with single holding (max concentration)"""
        holdings = [
            Holding(symbol="AAPL", quantity=100, cost_basis=10000, market_value=10000)
        ]

        metrics = compute_concentration_metrics(holdings)

        assert abs(metrics["top5_weight"] - 1.0) < 0.001
        assert abs(metrics["hhi_holdings"] - 1.0) < 0.001  # Max concentration

    def test_compute_sector_concentration(self):
        """Test sector concentration metrics"""
        holdings = [
            Holding(symbol="AAPL", quantity=100, cost_basis=5000,
                   market_value=5000, sector="Technology"),
            Holding(symbol="MSFT", quantity=50, cost_basis=3000,
                   market_value=3000, sector="Technology"),
            Holding(symbol="JPM", quantity=30, cost_basis=2000,
                   market_value=2000, sector="Financials"),
        ]

        metrics = compute_sector_concentration_metrics(holdings)

        # Tech = 80%, Financials = 20%
        # HHI = 0.8^2 + 0.2^2 = 0.68
        expected_hhi = 0.8**2 + 0.2**2

        assert abs(metrics["hhi_sectors"] - expected_hhi) < 0.001
        assert abs(metrics["max_sector_weight"] - 0.8) < 0.001
        assert metrics["num_sectors"] == 2


class TestTurnover:
    """Test turnover calculations"""

    def test_compute_turnover_no_change(self):
        """Test turnover when weights don't change"""
        weights = {"AAPL": 0.5, "MSFT": 0.5}

        turnover = compute_turnover(weights, weights)

        assert abs(turnover) < 0.001  # 0% turnover

    def test_compute_turnover_complete_change(self):
        """Test turnover with complete portfolio change"""
        prev_weights = {"AAPL": 0.5, "MSFT": 0.5}
        curr_weights = {"GOOGL": 0.5, "AMZN": 0.5}

        turnover = compute_turnover(curr_weights, prev_weights)

        assert abs(turnover - 1.0) < 0.001  # 100% turnover

    def test_compute_turnover_partial(self):
        """Test partial turnover"""
        # Sell 20% of AAPL, buy 20% GOOGL
        prev_weights = {"AAPL": 0.5, "MSFT": 0.5}
        curr_weights = {"AAPL": 0.3, "MSFT": 0.5, "GOOGL": 0.2}

        turnover = compute_turnover(curr_weights, prev_weights)

        # Changes: AAPL -0.2, GOOGL +0.2
        # Sum of abs = 0.4, turnover = 0.2 (20%)
        assert abs(turnover - 0.2) < 0.001


class TestUtilities:
    """Test utility functions"""

    def test_validate_weights_valid(self):
        """Test weight validation with valid weights"""
        weights = {"AAPL": 0.5, "MSFT": 0.5}

        assert validate_weights(weights) is True

    def test_validate_weights_invalid(self):
        """Test weight validation with invalid weights"""
        weights = {"AAPL": 0.6, "MSFT": 0.5}  # Sum = 1.1

        assert validate_weights(weights) is False

    def test_validate_weights_within_tolerance(self):
        """Test weight validation within tolerance"""
        weights = {"AAPL": 0.501, "MSFT": 0.500}  # Sum = 1.001

        assert validate_weights(weights, tolerance=0.01) is True
        assert validate_weights(weights, tolerance=0.0001) is False

    def test_normalize_weights(self):
        """Test weight normalization"""
        weights = {"AAPL": 0.6, "MSFT": 0.5}  # Sum = 1.1

        normalized = normalize_weights(weights)

        assert abs(sum(normalized.values()) - 1.0) < 0.001
        assert abs(normalized["AAPL"] - 0.545) < 0.01  # 0.6/1.1
        assert abs(normalized["MSFT"] - 0.455) < 0.01  # 0.5/1.1

    def test_normalize_weights_zero(self):
        """Test normalization with zero total"""
        weights = {"AAPL": 0.0, "MSFT": 0.0}

        normalized = normalize_weights(weights)

        # Should return unchanged (with warning logged)
        assert normalized == weights


class TestPositionSizes:
    """Test position size DataFrame generation"""

    def test_compute_position_sizes(self):
        """Test position sizes DataFrame"""
        holdings = [
            Holding(symbol="AAPL", quantity=100, cost_basis=10000,
                   market_value=15000, sector="Technology"),
            Holding(symbol="MSFT", quantity=50, cost_basis=5000,
                   market_value=10000, sector="Technology"),
            Holding(symbol="JPM", quantity=30, cost_basis=3000,
                   market_value=5000, sector="Financials"),
        ]

        df = compute_position_sizes(holdings)

        assert len(df) == 3
        assert list(df.columns) == ["symbol", "market_value", "weight", "sector", "rank"]

        # Should be sorted by weight descending
        assert df.iloc[0]["symbol"] == "AAPL"
        assert df.iloc[0]["rank"] == 1
        assert df.iloc[2]["symbol"] == "JPM"
        assert df.iloc[2]["rank"] == 3

        # Check weights
        assert abs(df.iloc[0]["weight"] - 0.5) < 0.001  # 15000/30000


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
