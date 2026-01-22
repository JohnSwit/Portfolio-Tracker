"""
Unit tests for liquidity analysis and implementation risk metrics
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Import liquidity module directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "liquidity",
    os.path.join(os.path.dirname(__file__), '..', 'backend', 'services', 'liquidity.py')
)
liquidity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(liquidity)

# Import Holding model
from models.portfolio import Holding


class TestDaysToExit:
    """Test days-to-exit calculations"""

    def test_compute_days_to_exit_basic(self):
        """Test basic days to exit calculation"""
        # 100k shares, 1M daily volume, 20% participation
        # = 100k / (1M * 0.20) = 0.5 days
        dte = liquidity.compute_days_to_exit(100_000, 1_000_000, 0.20)
        assert abs(dte - 0.5) < 0.01

    def test_compute_days_to_exit_large_position(self):
        """Test with large position relative to volume"""
        # 5M shares, 1M daily volume, 20% participation
        # = 5M / 200k = 25 days
        dte = liquidity.compute_days_to_exit(5_000_000, 1_000_000, 0.20)
        assert abs(dte - 25.0) < 0.01

    def test_compute_days_to_exit_small_position(self):
        """Test with small position (fractional day)"""
        # 10k shares, 1M daily volume, 20% participation
        dte = liquidity.compute_days_to_exit(10_000, 1_000_000, 0.20)
        assert dte < 0.1  # Very liquid

    def test_compute_days_to_exit_zero_volume(self):
        """Test with zero volume (illiquid)"""
        dte = liquidity.compute_days_to_exit(100_000, 0, 0.20)
        assert dte == float('inf')

    def test_compute_days_to_exit_negative_shares(self):
        """Test with negative shares (short position)"""
        # Should use absolute value
        dte = liquidity.compute_days_to_exit(-100_000, 1_000_000, 0.20)
        assert abs(dte - 0.5) < 0.01


class TestLiquidityScore:
    """Test liquidity scoring"""

    def test_compute_liquidity_score_highly_liquid(self):
        """Test highly liquid classification"""
        score = liquidity.compute_liquidity_score(0.5)
        assert score == "Highly Liquid"

    def test_compute_liquidity_score_liquid(self):
        """Test liquid classification"""
        score = liquidity.compute_liquidity_score(2.0)
        assert score == "Liquid"

    def test_compute_liquidity_score_moderately_liquid(self):
        """Test moderately liquid classification"""
        score = liquidity.compute_liquidity_score(5.0)
        assert score == "Moderately Liquid"

    def test_compute_liquidity_score_illiquid(self):
        """Test illiquid classification"""
        score = liquidity.compute_liquidity_score(20.0)
        assert score == "Illiquid"

    def test_compute_liquidity_score_highly_illiquid(self):
        """Test highly illiquid classification"""
        score = liquidity.compute_liquidity_score(50.0)
        assert score == "Highly Illiquid"

    def test_compute_liquidity_score_custom_thresholds(self):
        """Test with custom thresholds"""
        custom = {
            "highly_liquid": 0.5,
            "liquid": 2.0,
            "moderately_liquid": 5.0,
            "illiquid": 15.0
        }
        score = liquidity.compute_liquidity_score(3.0, thresholds=custom)
        assert score == "Moderately Liquid"


class TestPortfolioLiquidityProfile:
    """Test portfolio-level liquidity analysis"""

    def test_compute_portfolio_liquidity_profile_basic(self):
        """Test basic portfolio liquidity profile"""
        holdings = [
            Holding(
                symbol="AAPL",
                quantity=10000,
                market_value=1_500_000,
                cost_basis=1_200_000
            ),
            Holding(
                symbol="ILLIQUID",
                quantity=50000,
                market_value=500_000,
                cost_basis=480_000
            )
        ]

        volume_data = {
            "AAPL": 50_000_000,  # Very liquid
            "ILLIQUID": 100_000   # Very illiquid
        }

        profile = liquidity.compute_portfolio_liquidity_profile(
            holdings, volume_data, max_volume_participation=0.20
        )

        assert "position_liquidity" in profile
        assert "weighted_avg_days_to_exit" in profile
        assert "liquidity_distribution" in profile
        assert "most_illiquid_positions" in profile

        # Should have 2 positions
        assert len(profile["position_liquidity"]) == 2

        # ILLIQUID should be first (most illiquid)
        assert profile["position_liquidity"][0]["symbol"] == "ILLIQUID"

    def test_compute_portfolio_liquidity_profile_empty(self):
        """Test with empty portfolio"""
        profile = liquidity.compute_portfolio_liquidity_profile([], {})

        assert profile["position_liquidity"] == []
        assert profile["weighted_avg_days_to_exit"] == 0.0
        assert profile["total_market_value"] == 0.0

    def test_compute_portfolio_liquidity_profile_distribution(self):
        """Test liquidity distribution calculation"""
        holdings = [
            Holding(symbol="LIQ1", quantity=1000, market_value=500_000, cost_basis=400_000),
            Holding(symbol="LIQ2", quantity=1000, market_value=500_000, cost_basis=400_000)
        ]

        volume_data = {
            "LIQ1": 10_000_000,  # Highly liquid
            "LIQ2": 50_000       # Illiquid
        }

        profile = liquidity.compute_portfolio_liquidity_profile(holdings, volume_data)

        distribution = profile["liquidity_distribution"]

        # Should have both liquid and illiquid buckets with weight
        assert distribution["Highly Liquid"] > 0 or distribution["Liquid"] > 0
        # Total distribution should sum to ~1.0 (100%)
        total = sum(distribution.values())
        assert abs(total - 1.0) < 0.01


class TestBidAskImpact:
    """Test bid-ask spread impact calculations"""

    def test_compute_bid_ask_impact_sell(self):
        """Test bid-ask impact for sell order"""
        result = liquidity.compute_bid_ask_impact(
            shares=1000,
            bid_price=99.90,
            ask_price=100.10,
            trade_side="sell"
        )

        # Spread = 0.20, mid = 100.00
        # Spread bps = 0.20 / 100 * 10000 = 20 bps
        assert abs(result["spread_bps"] - 20) < 0.1

        # Selling at bid (99.90) vs mid (100.00) = 0.10 slippage per share
        # Impact = 1000 * 0.10 = $100
        assert abs(result["impact_cost"] - 100) < 1

        # Impact bps = 0.10 / 100 * 10000 = 10 bps
        assert abs(result["impact_bps"] - 10) < 0.1

    def test_compute_bid_ask_impact_buy(self):
        """Test bid-ask impact for buy order"""
        result = liquidity.compute_bid_ask_impact(
            shares=1000,
            bid_price=99.90,
            ask_price=100.10,
            trade_side="buy"
        )

        # Buying at ask (100.10) vs mid (100.00) = 0.10 premium per share
        # Impact = 1000 * 0.10 = $100
        assert abs(result["impact_cost"] - 100) < 1

    def test_compute_bid_ask_impact_invalid_prices(self):
        """Test with invalid bid/ask prices"""
        result = liquidity.compute_bid_ask_impact(
            shares=1000,
            bid_price=100.10,  # Bid > ask (invalid)
            ask_price=99.90,
            trade_side="sell"
        )

        assert result["spread_bps"] is None
        assert result["impact_cost"] is None


class TestMarketImpact:
    """Test market impact estimation"""

    def test_estimate_market_impact_small_order(self):
        """Test impact of small order"""
        result = liquidity.estimate_market_impact(
            shares=10_000,
            avg_daily_volume=10_000_000,
            price=100.0,
            volatility=0.02
        )

        # Volume participation = 0.1%
        assert abs(result["volume_participation"] - 0.1) < 0.01

        # Impact should be small
        assert result["temporary_impact_bps"] < 10

    def test_estimate_market_impact_large_order(self):
        """Test impact of large order"""
        result = liquidity.estimate_market_impact(
            shares=1_000_000,  # 10% of volume
            avg_daily_volume=10_000_000,
            price=100.0,
            volatility=0.02
        )

        # Volume participation = 10%
        assert abs(result["volume_participation"] - 10.0) < 0.1

        # Impact should be meaningful (sqrt of 10% ≈ 31.6%)
        # impact_bps = 0.02 * sqrt(0.10) * 10000 ≈ 63 bps
        assert result["temporary_impact_bps"] > 50

    def test_estimate_market_impact_invalid_volume(self):
        """Test with invalid volume"""
        result = liquidity.estimate_market_impact(
            shares=1000,
            avg_daily_volume=0,
            price=100.0
        )

        assert result["temporary_impact"] is None


class TestTurnoverCapacity:
    """Test turnover capacity calculations"""

    def test_compute_turnover_capacity_liquid(self):
        """Test turnover capacity with liquid portfolio"""
        holdings = [
            Holding(symbol="LIQ1", quantity=10000, market_value=1_000_000, cost_basis=900_000)
        ]

        volume_data = {"LIQ1": 50_000_000}  # Very liquid

        profile = liquidity.compute_portfolio_liquidity_profile(holdings, volume_data)

        capacity = liquidity.compute_turnover_capacity(
            portfolio_value=1_000_000,
            liquidity_profile=profile,
            target_horizon_days=5
        )

        # Should have high turnover capacity
        assert capacity["max_daily_turnover_pct"] > 10
        assert len(capacity["constrained_positions"]) == 0

    def test_compute_turnover_capacity_illiquid(self):
        """Test turnover capacity with illiquid position"""
        holdings = [
            Holding(symbol="ILLIQ", quantity=100000, market_value=1_000_000, cost_basis=950_000)
        ]

        volume_data = {"ILLIQ": 50_000}  # Very illiquid

        profile = liquidity.compute_portfolio_liquidity_profile(holdings, volume_data)

        capacity = liquidity.compute_turnover_capacity(
            portfolio_value=1_000_000,
            liquidity_profile=profile,
            target_horizon_days=5
        )

        # Should have constrained positions
        assert len(capacity["constrained_positions"]) > 0
        assert capacity["constrained_positions"][0]["symbol"] == "ILLIQ"


class TestPositionSizeValidation:
    """Test position size validation"""

    def test_validate_position_size_valid(self):
        """Test valid position size"""
        result = liquidity.validate_position_size(
            shares=50_000,
            avg_daily_volume=10_000_000,
            max_volume_pct=0.20,
            max_days_to_exit=5.0
        )

        # 50k / (10M * 0.20) = 0.025 days < 5 days
        assert result["is_valid"] is True
        assert result["days_to_exit"] < 1
        assert result["excess_shares"] == 0

    def test_validate_position_size_invalid(self):
        """Test oversized position"""
        result = liquidity.validate_position_size(
            shares=20_000_000,  # Very large
            avg_daily_volume=1_000_000,
            max_volume_pct=0.20,
            max_days_to_exit=5.0
        )

        # 20M / (1M * 0.20) = 100 days > 5 days
        assert result["is_valid"] is False
        assert result["days_to_exit"] > 5
        assert result["excess_shares"] > 0

    def test_validate_position_size_utilization(self):
        """Test utilization calculation"""
        result = liquidity.validate_position_size(
            shares=500_000,
            avg_daily_volume=10_000_000,
            max_volume_pct=0.20,
            max_days_to_exit=5.0
        )

        # Max allowed = 10M * 0.20 * 5 = 10M shares
        # Utilization = 500k / 10M = 5%
        assert result["utilization_pct"] is not None
        assert result["utilization_pct"] < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
