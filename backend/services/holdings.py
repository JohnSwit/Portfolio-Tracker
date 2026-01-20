import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
import logging

from models.portfolio import Portfolio, Holding, HoldingsAnalysis
from services.market_data import market_data_service

logger = logging.getLogger(__name__)


class HoldingsService:
    """Service for analyzing portfolio holdings"""

    def analyze_holdings(
        self,
        portfolio: Portfolio,
        benchmark_weights: Optional[Dict[str, float]] = None
    ) -> HoldingsAnalysis:
        """
        Perform comprehensive holdings analysis

        Args:
            portfolio: Portfolio data
            benchmark_weights: Optional benchmark weights for active share

        Returns:
            HoldingsAnalysis object
        """
        try:
            if not portfolio.holdings:
                return self._empty_analysis()

            # Calculate total value and weights
            total_value = sum(h.market_value or 0 for h in portfolio.holdings)
            if total_value == 0:
                return self._empty_analysis()

            # Enrich holdings with market data if needed
            self._enrich_holdings(portfolio)

            # Concentration metrics
            sorted_holdings = sorted(
                portfolio.holdings,
                key=lambda h: h.market_value or 0,
                reverse=True
            )

            top_10_value = sum(h.market_value or 0 for h in sorted_holdings[:10])
            top_20_value = sum(h.market_value or 0 for h in sorted_holdings[:20])

            top_10_concentration = top_10_value / total_value
            top_20_concentration = top_20_value / total_value

            # Herfindahl index (sum of squared weights)
            weights = [(h.market_value or 0) / total_value for h in portfolio.holdings]
            herfindahl_index = sum(w ** 2 for w in weights)

            # Active share
            active_share = None
            if benchmark_weights:
                active_share = self._calculate_active_share(
                    portfolio, benchmark_weights, total_value
                )

            # Exposure breakdowns
            sector_exposure = self._calculate_sector_exposure(portfolio, total_value)
            country_exposure = self._calculate_country_exposure(portfolio, total_value)
            industry_exposure = self._calculate_industry_exposure(portfolio, total_value)
            asset_class_exposure = self._calculate_asset_class_exposure(portfolio, total_value)

            # Liquidity metrics
            liquidity_metrics = self._calculate_liquidity_metrics(portfolio, total_value)

            # Factor exposures
            factor_metrics = self._calculate_factor_metrics(portfolio, total_value)

            return HoldingsAnalysis(
                top_10_concentration=top_10_concentration,
                top_20_concentration=top_20_concentration,
                herfindahl_index=herfindahl_index,
                active_share=active_share,
                sector_exposure=sector_exposure,
                country_exposure=country_exposure,
                industry_exposure=industry_exposure,
                asset_class_exposure=asset_class_exposure,
                avg_daily_volume=liquidity_metrics['avg_volume'],
                weighted_liquidity_score=liquidity_metrics['liquidity_score'],
                illiquid_holdings_pct=liquidity_metrics['illiquid_pct'],
                avg_market_cap=factor_metrics['avg_market_cap'],
                median_market_cap=factor_metrics['median_market_cap'],
                avg_pe_ratio=factor_metrics['avg_pe'],
                avg_pb_ratio=factor_metrics['avg_pb'],
                weighted_beta=factor_metrics['weighted_beta'],
                value_score=factor_metrics['value_score'],
                growth_score=factor_metrics['growth_score'],
                momentum_score=factor_metrics['momentum_score'],
                quality_score=factor_metrics['quality_score']
            )

        except Exception as e:
            logger.error(f"Error analyzing holdings: {e}")
            raise

    def _enrich_holdings(self, portfolio: Portfolio):
        """Enrich holdings with market data if missing"""
        for holding in portfolio.holdings:
            # Only fetch if critical data is missing
            if holding.sector is None or holding.market_cap is None:
                try:
                    info = market_data_service.get_stock_info(holding.symbol)
                    holding.sector = holding.sector or info.get('sector')
                    holding.industry = holding.industry or info.get('industry')
                    holding.country = holding.country or info.get('country')
                    holding.market_cap = holding.market_cap or info.get('market_cap')
                    holding.pe_ratio = holding.pe_ratio or info.get('pe_ratio')
                    holding.pb_ratio = holding.pb_ratio or info.get('pb_ratio')
                    holding.dividend_yield = holding.dividend_yield or info.get('dividend_yield')
                    holding.beta = holding.beta or info.get('beta')
                    holding.avg_daily_volume = holding.avg_daily_volume or info.get('avg_volume')
                    holding.current_price = holding.current_price or info.get('current_price')

                    if holding.current_price and holding.quantity:
                        holding.market_value = holding.current_price * holding.quantity
                except Exception as e:
                    logger.warning(f"Could not enrich data for {holding.symbol}: {e}")

    def _calculate_active_share(
        self,
        portfolio: Portfolio,
        benchmark_weights: Dict[str, float],
        total_value: float
    ) -> float:
        """Calculate active share vs benchmark"""
        portfolio_weights = {
            h.symbol: (h.market_value or 0) / total_value
            for h in portfolio.holdings
        }

        # Get all symbols (portfolio + benchmark)
        all_symbols = set(portfolio_weights.keys()) | set(benchmark_weights.keys())

        # Active share = 0.5 * sum(|portfolio_weight - benchmark_weight|)
        active_share = 0.5 * sum(
            abs(portfolio_weights.get(symbol, 0) - benchmark_weights.get(symbol, 0))
            for symbol in all_symbols
        )

        return active_share

    def _calculate_sector_exposure(
        self,
        portfolio: Portfolio,
        total_value: float
    ) -> Dict[str, float]:
        """Calculate exposure by sector"""
        sector_values = defaultdict(float)

        for holding in portfolio.holdings:
            sector = holding.sector or 'Unknown'
            sector_values[sector] += holding.market_value or 0

        return {
            sector: value / total_value
            for sector, value in sector_values.items()
        }

    def _calculate_country_exposure(
        self,
        portfolio: Portfolio,
        total_value: float
    ) -> Dict[str, float]:
        """Calculate exposure by country"""
        country_values = defaultdict(float)

        for holding in portfolio.holdings:
            country = holding.country or 'Unknown'
            country_values[country] += holding.market_value or 0

        return {
            country: value / total_value
            for country, value in country_values.items()
        }

    def _calculate_industry_exposure(
        self,
        portfolio: Portfolio,
        total_value: float
    ) -> Dict[str, float]:
        """Calculate exposure by industry"""
        industry_values = defaultdict(float)

        for holding in portfolio.holdings:
            industry = holding.industry or 'Unknown'
            industry_values[industry] += holding.market_value or 0

        return {
            industry: value / total_value
            for industry, value in industry_values.items()
        }

    def _calculate_asset_class_exposure(
        self,
        portfolio: Portfolio,
        total_value: float
    ) -> Dict[str, float]:
        """Calculate exposure by asset class"""
        asset_class_values = defaultdict(float)

        for holding in portfolio.holdings:
            asset_class = holding.asset_class or 'Unknown'
            asset_class_values[asset_class] += holding.market_value or 0

        return {
            asset_class: value / total_value
            for asset_class, value in asset_class_values.items()
        }

    def _calculate_liquidity_metrics(
        self,
        portfolio: Portfolio,
        total_value: float
    ) -> Dict[str, float]:
        """Calculate liquidity metrics"""
        volumes = []
        liquidity_scores = []
        illiquid_value = 0.0

        for holding in portfolio.holdings:
            volume = holding.avg_daily_volume or 0
            volumes.append(volume)

            # Position size as % of daily volume
            if volume > 0 and holding.market_value:
                position_value = holding.market_value
                days_to_liquidate = position_value / (volume * (holding.current_price or 1))

                # Liquidity score: inverse of days to liquidate
                # Higher is better (more liquid)
                liquidity_score = 1 / (1 + days_to_liquidate) if days_to_liquidate > 0 else 0
                liquidity_scores.append(liquidity_score)

                # Consider illiquid if takes more than 5 days to liquidate
                if days_to_liquidate > 5:
                    illiquid_value += holding.market_value or 0
            else:
                liquidity_scores.append(0.0)

        avg_volume = np.mean(volumes) if volumes else 0.0
        weighted_liquidity = np.mean(liquidity_scores) if liquidity_scores else 0.0
        illiquid_pct = illiquid_value / total_value if total_value > 0 else 0.0

        return {
            'avg_volume': avg_volume,
            'liquidity_score': weighted_liquidity,
            'illiquid_pct': illiquid_pct
        }

    def _calculate_factor_metrics(
        self,
        portfolio: Portfolio,
        total_value: float
    ) -> Dict[str, Optional[float]]:
        """Calculate factor exposure metrics"""
        market_caps = []
        pe_ratios = []
        pb_ratios = []
        weighted_beta = 0.0

        # For factor scores
        value_scores = []
        growth_scores = []
        momentum_scores = []
        quality_scores = []
        weights = []

        for holding in portfolio.holdings:
            weight = (holding.market_value or 0) / total_value

            # Market cap
            if holding.market_cap:
                market_caps.append(holding.market_cap)

            # Valuation metrics
            if holding.pe_ratio:
                pe_ratios.append(holding.pe_ratio)
            if holding.pb_ratio:
                pb_ratios.append(holding.pb_ratio)

            # Beta
            if holding.beta:
                weighted_beta += holding.beta * weight

            # Factor scores
            if holding.pe_ratio and holding.pb_ratio:
                # Value score: lower P/E and P/B is better (higher value)
                value_score = 1 / (1 + holding.pe_ratio / 20) + 1 / (1 + holding.pb_ratio / 2)
                value_scores.append(value_score * weight)

                # Growth score: based on forward P/E premium
                # Simplified - in production would use actual growth estimates
                growth_score = holding.pe_ratio / 20 if holding.pe_ratio > 15 else 0
                growth_scores.append(growth_score * weight)

                weights.append(weight)

            # Momentum score - would need price history
            # Quality score - would need ROE, ROA, debt ratios

        # Calculate averages
        avg_market_cap = np.mean(market_caps) if market_caps else 0.0
        median_market_cap = np.median(market_caps) if market_caps else 0.0
        avg_pe = np.mean(pe_ratios) if pe_ratios else None
        avg_pb = np.mean(pb_ratios) if pb_ratios else None

        # Aggregate factor scores
        value_score = sum(value_scores) if value_scores else None
        growth_score = sum(growth_scores) if growth_scores else None

        return {
            'avg_market_cap': avg_market_cap,
            'median_market_cap': median_market_cap,
            'avg_pe': avg_pe,
            'avg_pb': avg_pb,
            'weighted_beta': weighted_beta,
            'value_score': value_score,
            'growth_score': growth_score,
            'momentum_score': None,  # Would need price history
            'quality_score': None  # Would need fundamental data
        }

    def _empty_analysis(self) -> HoldingsAnalysis:
        """Return empty analysis for portfolios with no holdings"""
        return HoldingsAnalysis(
            top_10_concentration=0.0,
            top_20_concentration=0.0,
            herfindahl_index=0.0,
            active_share=None,
            sector_exposure={},
            country_exposure={},
            industry_exposure={},
            asset_class_exposure={},
            avg_daily_volume=0.0,
            weighted_liquidity_score=0.0,
            illiquid_holdings_pct=0.0,
            avg_market_cap=0.0,
            median_market_cap=0.0,
            avg_pe_ratio=None,
            avg_pb_ratio=None,
            weighted_beta=0.0,
            value_score=None,
            growth_score=None,
            momentum_score=None,
            quality_score=None
        )


# Global instance
holdings_service = HoldingsService()
