"""
Unit tests for performance attribution calculations
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Import attribution module directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "attribution",
    os.path.join(os.path.dirname(__file__), '..', 'backend', 'services', 'attribution.py')
)
attribution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attribution)

# Import Holding model
from models.portfolio import Holding


class TestContributionToReturn:
    """Test position-level contribution calculations"""

    def test_compute_contribution_basic(self):
        """Test basic contribution calculation"""
        holdings = [
            Holding(
                symbol="AAPL",
                quantity=10,
                market_value=10000,
                cost_basis=9000,
                sector="Technology"
            ),
            Holding(
                symbol="MSFT",
                quantity=20,
                market_value=5000,
                cost_basis=4500,
                sector="Technology"
            )
        ]

        returns = {"AAPL": 0.10, "MSFT": 0.05}

        contributions = attribution.compute_contribution_to_return(holdings, returns)

        # Total value = 15000
        # AAPL: weight = 10000/15000 = 0.6667, contribution = 0.6667 * 0.10 = 0.0667
        # MSFT: weight = 5000/15000 = 0.3333, contribution = 0.3333 * 0.05 = 0.0167

        assert len(contributions) == 2

        # Should be sorted by contribution (AAPL first)
        assert contributions[0]["symbol"] == "AAPL"
        assert abs(contributions[0]["contribution"] - 0.0667) < 0.001

        assert contributions[1]["symbol"] == "MSFT"
        assert abs(contributions[1]["contribution"] - 0.0167) < 0.001

    def test_compute_contribution_negative_returns(self):
        """Test contribution with negative returns"""
        holdings = [
            Holding(symbol="STOCK1", quantity=100, market_value=10000, cost_basis=12000),
            Holding(symbol="STOCK2", quantity=50, market_value=5000, cost_basis=4000)
        ]

        returns = {"STOCK1": -0.10, "STOCK2": 0.20}

        contributions = attribution.compute_contribution_to_return(holdings, returns)

        # STOCK1: -0.0667, STOCK2: +0.0667
        # Should be sorted by contribution (STOCK2 first)
        assert contributions[0]["symbol"] == "STOCK2"
        assert contributions[0]["contribution"] > 0

        assert contributions[1]["symbol"] == "STOCK1"
        assert contributions[1]["contribution"] < 0

    def test_compute_contribution_missing_returns(self):
        """Test contribution when some returns are missing"""
        holdings = [
            Holding(symbol="AAPL", quantity=10, market_value=10000, cost_basis=9000),
            Holding(symbol="MSFT", quantity=20, market_value=5000, cost_basis=4500)
        ]

        # Only AAPL has a return
        returns = {"AAPL": 0.10}

        contributions = attribution.compute_contribution_to_return(holdings, returns)

        assert len(contributions) == 2

        # MSFT should have 0 contribution (missing return defaults to 0)
        msft_contrib = next(c for c in contributions if c["symbol"] == "MSFT")
        assert msft_contrib["contribution"] == 0.0


class TestBrinsonAttribution:
    """Test Brinson attribution calculations"""

    def test_brinson_simple_case(self):
        """Test Brinson with simple two-sector example"""
        # Portfolio: 70% Tech (return 10%), 30% Finance (return 5%)
        # Benchmark: 50% Tech (return 8%), 50% Finance (return 4%)

        portfolio_holdings = [
            Holding(symbol="TECH1", quantity=100, market_value=7000,
                   cost_basis=6000, sector="Technology"),
            Holding(symbol="FIN1", quantity=50, market_value=3000,
                   cost_basis=2800, sector="Finance")
        ]

        benchmark_holdings = [
            Holding(symbol="TECH_B", quantity=100, market_value=5000,
                   cost_basis=4500, sector="Technology"),
            Holding(symbol="FIN_B", quantity=100, market_value=5000,
                   cost_basis=4700, sector="Finance")
        ]

        portfolio_returns = {"TECH1": 0.10, "FIN1": 0.05}
        benchmark_returns = {"TECH_B": 0.08, "FIN_B": 0.04}

        result = attribution.brinson_attribution(
            portfolio_holdings,
            benchmark_holdings,
            portfolio_returns,
            benchmark_returns
        )

        # Verify keys exist
        assert "allocation_effect" in result
        assert "selection_effect" in result
        assert "interaction_effect" in result
        assert "total_active_return" in result
        assert "sector_attribution" in result

        # Benchmark return = 0.5 * 0.08 + 0.5 * 0.04 = 0.06 (6%)
        assert abs(result["benchmark_return"] - 0.06) < 0.001

        # Portfolio return = 0.7 * 0.10 + 0.3 * 0.05 = 0.085 (8.5%)
        # Active return = 0.085 - 0.06 = 0.025 (2.5%)

        # Total active return should equal sum of effects
        total_effect = (
            result["allocation_effect"] +
            result["selection_effect"] +
            result["interaction_effect"]
        )
        assert abs(result["total_active_return"] - total_effect) < 0.0001

    def test_brinson_perfect_tracking(self):
        """Test Brinson when portfolio perfectly tracks benchmark"""
        holdings = [
            Holding(symbol="AAPL", quantity=100, market_value=5000,
                   cost_basis=4500, sector="Technology"),
            Holding(symbol="JPM", quantity=50, market_value=5000,
                   cost_basis=4800, sector="Finance")
        ]

        returns = {"AAPL": 0.10, "JPM": 0.05}

        # Same holdings and returns for both
        result = attribution.brinson_attribution(
            holdings, holdings, returns, returns
        )

        # All effects should be zero (perfect tracking)
        assert abs(result["allocation_effect"]) < 0.001
        assert abs(result["selection_effect"]) < 0.001
        assert abs(result["interaction_effect"]) < 0.001
        assert abs(result["total_active_return"]) < 0.001

    def test_brinson_sector_breakdown(self):
        """Test that sector attribution contains expected fields"""
        portfolio_holdings = [
            Holding(symbol="TECH1", quantity=100, market_value=6000,
                   cost_basis=5500, sector="Technology"),
            Holding(symbol="FIN1", quantity=50, market_value=4000,
                   cost_basis=3800, sector="Finance")
        ]

        benchmark_holdings = [
            Holding(symbol="TECH_B", quantity=100, market_value=5000,
                   cost_basis=4500, sector="Technology"),
            Holding(symbol="FIN_B", quantity=100, market_value=5000,
                   cost_basis=4700, sector="Finance")
        ]

        portfolio_returns = {"TECH1": 0.12, "FIN1": 0.06}
        benchmark_returns = {"TECH_B": 0.10, "FIN_B": 0.05}

        result = attribution.brinson_attribution(
            portfolio_holdings,
            benchmark_holdings,
            portfolio_returns,
            benchmark_returns
        )

        sector_attr = result["sector_attribution"]

        # Should have both sectors
        assert "Technology" in sector_attr
        assert "Finance" in sector_attr

        # Each sector should have all required fields
        for sector, data in sector_attr.items():
            assert "portfolio_weight" in data
            assert "benchmark_weight" in data
            assert "portfolio_return" in data
            assert "benchmark_return" in data
            assert "allocation_effect" in data
            assert "selection_effect" in data
            assert "interaction_effect" in data
            assert "total_effect" in data

    def test_brinson_unknown_sector(self):
        """Test Brinson with holdings missing sector info"""
        portfolio_holdings = [
            Holding(symbol="STOCK1", quantity=100, market_value=10000,
                   cost_basis=9000, sector=None)  # No sector
        ]

        benchmark_holdings = [
            Holding(symbol="STOCK_B", quantity=100, market_value=10000,
                   cost_basis=9500, sector=None)
        ]

        portfolio_returns = {"STOCK1": 0.10}
        benchmark_returns = {"STOCK_B": 0.08}

        result = attribution.brinson_attribution(
            portfolio_holdings,
            benchmark_holdings,
            portfolio_returns,
            benchmark_returns
        )

        # Should group under "Unknown"
        assert "Unknown" in result["sector_attribution"]


class TestTopContributors:
    """Test top contributors/detractors functions"""

    def test_get_top_contributors(self):
        """Test getting top positive contributors"""
        contributions = [
            {"symbol": "AAPL", "contribution": 0.05},
            {"symbol": "MSFT", "contribution": 0.03},
            {"symbol": "GOOGL", "contribution": 0.02},
            {"symbol": "AMZN", "contribution": -0.01},
            {"symbol": "META", "contribution": -0.02}
        ]

        top = attribution.get_top_contributors(contributions, n=3)

        assert len(top) == 3
        assert top[0]["symbol"] == "AAPL"
        assert top[1]["symbol"] == "MSFT"
        assert top[2]["symbol"] == "GOOGL"

    def test_get_top_detractors(self):
        """Test getting top negative contributors"""
        contributions = [
            {"symbol": "AAPL", "contribution": 0.05},
            {"symbol": "MSFT", "contribution": 0.03},
            {"symbol": "GOOGL", "contribution": 0.02},
            {"symbol": "AMZN", "contribution": -0.01},
            {"symbol": "META", "contribution": -0.02}
        ]

        detractors = attribution.get_top_detractors(contributions, n=2)

        assert len(detractors) == 2
        # Most negative first
        assert detractors[0]["symbol"] == "META"
        assert detractors[1]["symbol"] == "AMZN"


class TestAttributionValidation:
    """Test attribution validation"""

    def test_validate_attribution_valid(self):
        """Test validation with correct attribution"""
        portfolio_return = 0.10
        benchmark_return = 0.08
        allocation = 0.01
        selection = 0.008
        interaction = 0.002

        # Should sum to 0.02 (active return)
        is_valid = attribution.validate_attribution(
            allocation, selection, interaction,
            portfolio_return, benchmark_return
        )

        assert is_valid is True

    def test_validate_attribution_invalid(self):
        """Test validation with incorrect attribution (doesn't sum)"""
        portfolio_return = 0.10
        benchmark_return = 0.08
        allocation = 0.01
        selection = 0.005
        interaction = 0.001

        # Sums to 0.016, but active return is 0.02 (error = 0.004 > tolerance)
        is_valid = attribution.validate_attribution(
            allocation, selection, interaction,
            portfolio_return, benchmark_return,
            tolerance=0.001
        )

        assert is_valid is False


class TestActiveReturnFromContributions:
    """Test computing active return from contributions"""

    def test_compute_active_return(self):
        """Test summing contributions"""
        contributions = [
            {"symbol": "AAPL", "contribution": 0.05},
            {"symbol": "MSFT", "contribution": 0.03},
            {"symbol": "GOOGL", "contribution": -0.01}
        ]

        active_return = attribution.compute_active_return_from_contributions(contributions)

        assert abs(active_return - 0.07) < 0.001

    def test_compute_active_return_empty(self):
        """Test with empty contributions"""
        active_return = attribution.compute_active_return_from_contributions([])
        assert active_return == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
