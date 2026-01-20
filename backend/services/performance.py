import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from scipy.optimize import newton
import logging

from models.portfolio import Portfolio, Transaction, PerformanceMetrics, AttributionResult
from services.market_data import market_data_service

logger = logging.getLogger(__name__)


class PerformanceService:
    """Service for calculating portfolio performance metrics"""

    def calculate_twr(
        self,
        portfolio_values: pd.Series,
        cash_flows: pd.Series
    ) -> float:
        """
        Calculate Time-Weighted Return (TWR)

        TWR eliminates the effect of cash flows by linking sub-period returns

        Args:
            portfolio_values: Series of portfolio values indexed by date
            cash_flows: Series of cash flows indexed by date (positive = deposit)

        Returns:
            Time-weighted return as decimal
        """
        try:
            # Align series
            df = pd.DataFrame({
                'value': portfolio_values,
                'flow': cash_flows
            }).fillna(0)

            # Calculate sub-period returns
            df['prev_value'] = df['value'].shift(1)
            df['return'] = (df['value'] - df['prev_value'] - df['flow']) / df['prev_value']
            df['return'] = df['return'].fillna(0)

            # Link returns: (1 + r1) * (1 + r2) * ... - 1
            twr = (1 + df['return']).prod() - 1

            return float(twr)

        except Exception as e:
            logger.error(f"Error calculating TWR: {e}")
            return 0.0

    def calculate_mwr(
        self,
        transactions: List[Transaction],
        beginning_value: float,
        ending_value: float,
        start_date: datetime,
        end_date: datetime
    ) -> float:
        """
        Calculate Money-Weighted Return (Internal Rate of Return)

        MWR accounts for the timing and size of cash flows

        Args:
            transactions: List of transactions
            beginning_value: Portfolio value at start
            ending_value: Portfolio value at end
            start_date: Start date for calculation
            end_date: End date for calculation

        Returns:
            Money-weighted return (IRR) as decimal
        """
        try:
            # Build cash flow timeline
            total_days = (end_date - start_date).days
            if total_days == 0:
                return 0.0

            # Create cash flow array
            cash_flows = [-beginning_value]  # Initial investment (outflow)
            days = [0]

            # Add transactions
            for txn in transactions:
                if start_date <= txn.date <= end_date:
                    days_from_start = (txn.date - start_date).days

                    if txn.transaction_type == 'buy':
                        cash_flows.append(-txn.amount)  # Outflow
                    elif txn.transaction_type == 'sell':
                        cash_flows.append(txn.amount)  # Inflow
                    elif txn.transaction_type == 'dividend':
                        cash_flows.append(txn.amount)  # Inflow

                    days.append(days_from_start)

            # Add ending value
            cash_flows.append(ending_value)
            days.append(total_days)

            # Calculate IRR using Newton-Raphson method
            def npv(rate):
                """Net present value at given rate"""
                return sum(cf / ((1 + rate) ** (day / 365.0))
                          for cf, day in zip(cash_flows, days))

            def npv_derivative(rate):
                """Derivative of NPV"""
                return sum(-cf * (day / 365.0) / ((1 + rate) ** (day / 365.0 + 1))
                          for cf, day in zip(cash_flows, days))

            # Solve for IRR
            try:
                irr = newton(npv, 0.1, fprime=npv_derivative, maxiter=100)
                return float(irr)
            except:
                # If Newton's method fails, try simple approximation
                if beginning_value > 0:
                    return (ending_value - beginning_value) / beginning_value
                else:
                    return 0.0

        except Exception as e:
            logger.error(f"Error calculating MWR: {e}")
            return 0.0

    def calculate_performance_metrics(
        self,
        portfolio: Portfolio,
        transactions: List[Transaction],
        start_date: datetime,
        end_date: datetime,
        benchmark: str = "^GSPC"
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics

        Args:
            portfolio: Current portfolio
            transactions: Transaction history
            start_date: Start date
            end_date: End date
            benchmark: Benchmark ticker

        Returns:
            PerformanceMetrics object
        """
        try:
            # Calculate returns
            beginning_value = self._calculate_portfolio_value_at_date(
                portfolio, transactions, start_date
            )
            ending_value = portfolio.total_value or 0.0

            # Time-weighted return
            portfolio_values, cash_flows = self._build_value_series(
                portfolio, transactions, start_date, end_date
            )
            twr = self.calculate_twr(portfolio_values, cash_flows)

            # Money-weighted return (IRR)
            mwr = self.calculate_mwr(
                transactions,
                beginning_value,
                ending_value,
                start_date,
                end_date
            )

            # Total return
            total_return = (ending_value - beginning_value) / beginning_value if beginning_value > 0 else 0.0

            # Annualized return
            days = (end_date - start_date).days
            years = days / 365.0
            annualized_return = ((1 + twr) ** (1 / years) - 1) if years > 0 else twr

            # Benchmark comparison
            benchmark_data = market_data_service.get_benchmark_data(benchmark, start_date, end_date)
            benchmark_return = (benchmark_data.iloc[-1] - benchmark_data.iloc[0]) / benchmark_data.iloc[0]

            # Active return
            active_return = twr - benchmark_return

            # Tracking error
            portfolio_returns = portfolio_values.pct_change().dropna()
            benchmark_returns = benchmark_data.pct_change().dropna()

            # Align returns
            aligned = pd.DataFrame({
                'portfolio': portfolio_returns,
                'benchmark': benchmark_returns
            }).dropna()

            tracking_error = (aligned['portfolio'] - aligned['benchmark']).std() * np.sqrt(252)

            # Information ratio
            information_ratio = active_return / tracking_error if tracking_error > 0 else 0.0

            period_label = f"{start_date.date()} to {end_date.date()}"

            return PerformanceMetrics(
                period=period_label,
                start_date=start_date,
                end_date=end_date,
                twr=twr,
                mwr=mwr,
                total_return=total_return,
                annualized_return=annualized_return,
                benchmark_return=benchmark_return,
                active_return=active_return,
                tracking_error=tracking_error,
                information_ratio=information_ratio
            )

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            raise

    def calculate_attribution(
        self,
        portfolio: Portfolio,
        transactions: List[Transaction],
        start_date: datetime,
        end_date: datetime,
        benchmark_weights: Optional[Dict[str, float]] = None
    ) -> AttributionResult:
        """
        Calculate return attribution by sector, country, and stock

        Uses Brinson attribution methodology

        Args:
            portfolio: Portfolio data
            transactions: Transaction history
            start_date: Start date
            end_date: End date
            benchmark_weights: Optional benchmark weights by sector

        Returns:
            AttributionResult object
        """
        try:
            # Get stock returns for the period
            symbols = [h.symbol for h in portfolio.holdings]
            price_data = market_data_service.get_price_data(symbols, start_date, end_date)

            # Calculate individual stock returns
            stock_returns = {}
            for symbol in symbols:
                if len(symbols) == 1:
                    # For single symbol, yfinance returns different structure
                    if 'Close' in price_data.columns:
                        prices = price_data['Close']
                    else:
                        prices = price_data
                    # Ensure 1D series
                    if hasattr(prices, 'squeeze'):
                        prices = prices.squeeze()
                else:
                    prices = price_data[symbol]['Close']

                stock_return = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
                stock_returns[symbol] = float(stock_return)

            # Calculate portfolio weights
            total_value = sum(h.market_value or 0 for h in portfolio.holdings)
            weights = {h.symbol: (h.market_value or 0) / total_value
                      for h in portfolio.holdings if total_value > 0}

            # Portfolio return
            portfolio_return = sum(weights.get(s, 0) * stock_returns.get(s, 0)
                                  for s in symbols)

            # Sector attribution
            sector_attribution = self._calculate_sector_attribution(
                portfolio, stock_returns, weights, benchmark_weights
            )

            # Country attribution
            country_attribution = self._calculate_country_attribution(
                portfolio, stock_returns, weights
            )

            # Stock-level attribution
            stock_attribution = {
                symbol: {
                    'return': stock_returns.get(symbol, 0.0),
                    'weight': weights.get(symbol, 0.0),
                    'contribution': weights.get(symbol, 0.0) * stock_returns.get(symbol, 0.0)
                }
                for symbol in symbols
            }

            period_label = f"{start_date.date()} to {end_date.date()}"

            return AttributionResult(
                period=period_label,
                total_return=portfolio_return,
                sector_attribution=sector_attribution,
                country_attribution=country_attribution,
                stock_attribution=stock_attribution,
                factor_attribution={}  # To be implemented with factor analysis
            )

        except Exception as e:
            logger.error(f"Error calculating attribution: {e}")
            raise

    def _build_value_series(
        self,
        portfolio: Portfolio,
        transactions: List[Transaction],
        start_date: datetime,
        end_date: datetime
    ) -> Tuple[pd.Series, pd.Series]:
        """Build portfolio value and cash flow series"""
        # Simplified implementation - build daily value series
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # For now, use linear interpolation
        # In production, you'd calculate actual daily values
        beginning_value = self._calculate_portfolio_value_at_date(
            portfolio, transactions, start_date
        )
        ending_value = portfolio.total_value or 0.0

        values = pd.Series(
            np.linspace(beginning_value, ending_value, len(dates)),
            index=dates
        )

        # Build cash flows
        cash_flows = pd.Series(0.0, index=dates)
        for txn in transactions:
            if start_date <= txn.date <= end_date:
                if txn.transaction_type in ['buy', 'sell']:
                    cash_flows[txn.date] = txn.amount if txn.transaction_type == 'sell' else -txn.amount

        return values, cash_flows

    def _calculate_portfolio_value_at_date(
        self,
        portfolio: Portfolio,
        transactions: List[Transaction],
        date: datetime
    ) -> float:
        """Calculate portfolio value at a specific date"""
        # Simplified - in production, reconstruct holdings at that date
        if portfolio.inception_date and date < portfolio.inception_date:
            return 0.0

        # Estimate based on transactions
        total = 0.0
        for txn in transactions:
            if txn.date <= date:
                if txn.transaction_type == 'buy':
                    total += txn.amount
                elif txn.transaction_type == 'sell':
                    total -= txn.amount

        return max(total, 0.0)

    def _calculate_sector_attribution(
        self,
        portfolio: Portfolio,
        stock_returns: Dict[str, float],
        weights: Dict[str, float],
        benchmark_weights: Optional[Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate sector-level attribution"""
        sector_data = {}

        for holding in portfolio.holdings:
            sector = holding.sector or 'Unknown'
            symbol = holding.symbol

            if sector not in sector_data:
                sector_data[sector] = {
                    'weight': 0.0,
                    'return': 0.0,
                    'contribution': 0.0
                }

            weight = weights.get(symbol, 0.0)
            stock_return = stock_returns.get(symbol, 0.0)

            sector_data[sector]['weight'] += weight
            sector_data[sector]['contribution'] += weight * stock_return

        # Calculate sector returns
        for sector in sector_data:
            if sector_data[sector]['weight'] > 0:
                sector_data[sector]['return'] = (
                    sector_data[sector]['contribution'] / sector_data[sector]['weight']
                )

        return sector_data

    def _calculate_country_attribution(
        self,
        portfolio: Portfolio,
        stock_returns: Dict[str, float],
        weights: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate country-level attribution"""
        country_data = {}

        for holding in portfolio.holdings:
            country = holding.country or 'Unknown'
            symbol = holding.symbol

            if country not in country_data:
                country_data[country] = {
                    'weight': 0.0,
                    'return': 0.0,
                    'contribution': 0.0
                }

            weight = weights.get(symbol, 0.0)
            stock_return = stock_returns.get(symbol, 0.0)

            country_data[country]['weight'] += weight
            country_data[country]['contribution'] += weight * stock_return

        # Calculate country returns
        for country in country_data:
            if country_data[country]['weight'] > 0:
                country_data[country]['return'] = (
                    country_data[country]['contribution'] / country_data[country]['weight']
                )

        return country_data


# Global instance
performance_service = PerformanceService()
