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

    @staticmethod
    def _normalize_to_utc(dt: datetime) -> datetime:
        """Normalize datetime to timezone-aware UTC"""
        from datetime import timezone as dt_timezone
        if dt.tzinfo is None:
            return dt.replace(tzinfo=dt_timezone.utc)
        else:
            return dt.astimezone(dt_timezone.utc)

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
            # Avoid division by zero
            df['return'] = np.where(
                df['prev_value'] > 0,
                (df['value'] - df['prev_value'] - df['flow']) / df['prev_value'],
                0
            )
            df['return'] = df['return'].fillna(0)
            # Replace inf and -inf with 0
            df['return'] = df['return'].replace([np.inf, -np.inf], 0)

            # Link returns: (1 + r1) * (1 + r2) * ... - 1
            twr = (1 + df['return']).prod() - 1

            return float(twr) if not (pd.isna(twr) or np.isinf(twr)) else 0.0

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
            # Normalize dates to UTC timezone
            start_date = self._normalize_to_utc(start_date)
            end_date = self._normalize_to_utc(end_date)

            # Build cash flow timeline
            total_days = (end_date - start_date).days
            if total_days == 0:
                return 0.0

            # Create cash flow array
            cash_flows = [-beginning_value]  # Initial investment (outflow)
            days = [0]

            # Add transactions
            for txn in transactions:
                txn_date = self._normalize_to_utc(txn.date)
                if start_date <= txn_date <= end_date:
                    days_from_start = (txn_date - start_date).days

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
            # Normalize all dates to UTC timezone at the start
            start_date = self._normalize_to_utc(start_date)
            end_date = self._normalize_to_utc(end_date)

            # Calculate returns
            beginning_value = self._calculate_portfolio_value_at_date(
                portfolio, transactions, start_date
            )
            ending_value = portfolio.total_value or 0.0

            # Debug logging - print to console
            print(f"\n{'-'*60}")
            print(f"PERFORMANCE SERVICE CALCULATION:")
            print(f"  Period: {start_date.date()} to {end_date.date()}")
            print(f"  Beginning value: ${beginning_value:,.2f}")
            print(f"  Ending value: ${ending_value:,.2f}")
            print(f"  Portfolio inception: {portfolio.inception_date}")
            print(f"  Transactions count: {len(transactions)}")
            if beginning_value == 0:
                print(f"  ⚠️  WARNING: Beginning value is ZERO!")
            if ending_value == 0:
                print(f"  ⚠️  WARNING: Ending value is ZERO!")
            print(f"{'-'*60}")

            # Time-weighted return
            portfolio_values, cash_flows = self._build_value_series(
                portfolio, transactions, start_date, end_date
            )

            logger.info(f"  Value series length: {len(portfolio_values)}")
            logger.info(f"  Value series range: ${portfolio_values.min():,.2f} to ${portfolio_values.max():,.2f}")
            logger.info(f"  Cash flows sum: ${cash_flows.sum():,.2f}")

            twr = self.calculate_twr(portfolio_values, cash_flows)

            # Money-weighted return (IRR)
            mwr = self.calculate_mwr(
                transactions,
                beginning_value,
                ending_value,
                start_date,
                end_date
            )

            # Total return - avoid division by zero
            if beginning_value > 0:
                total_return = (ending_value - beginning_value) / beginning_value
            else:
                total_return = 0.0

            # Annualized return - avoid division by zero and invalid exponentiation
            days = (end_date - start_date).days
            years = days / 365.0
            if years > 0 and (1 + twr) > 0:
                annualized_return = ((1 + twr) ** (1 / years) - 1)
            else:
                annualized_return = twr

            # Benchmark comparison
            benchmark_data = market_data_service.get_benchmark_data(benchmark, start_date, end_date)
            if len(benchmark_data) > 0 and benchmark_data.iloc[0] != 0:
                benchmark_return = (benchmark_data.iloc[-1] - benchmark_data.iloc[0]) / benchmark_data.iloc[0]
            else:
                benchmark_return = 0.0

            # Active return
            active_return = twr - benchmark_return

            # Tracking error
            portfolio_returns = portfolio_values.pct_change(fill_method=None).dropna()
            benchmark_returns = benchmark_data.pct_change(fill_method=None).dropna()

            # Normalize timezones - convert both to timezone-naive for alignment
            if hasattr(portfolio_returns.index, 'tz') and portfolio_returns.index.tz is not None:
                portfolio_returns.index = portfolio_returns.index.tz_localize(None)
            if hasattr(benchmark_returns.index, 'tz') and benchmark_returns.index.tz is not None:
                benchmark_returns.index = benchmark_returns.index.tz_localize(None)

            # Align returns
            aligned = pd.DataFrame({
                'portfolio': portfolio_returns,
                'benchmark': benchmark_returns
            }).dropna()

            tracking_error = (aligned['portfolio'] - aligned['benchmark']).std() * np.sqrt(252) if len(aligned) > 0 else 0.0

            # Information ratio
            information_ratio = active_return / tracking_error if tracking_error > 0 else 0.0

            period_label = f"{start_date.date()} to {end_date.date()}"

            # Handle NaN values to prevent JSON serialization errors
            def safe_float(value):
                """Convert NaN/inf to 0.0 for JSON serialization"""
                if pd.isna(value) or np.isinf(value):
                    return 0.0
                return float(value)

            return PerformanceMetrics(
                period=period_label,
                start_date=start_date,
                end_date=end_date,
                twr=safe_float(twr),
                mwr=safe_float(mwr),
                total_return=safe_float(total_return),
                annualized_return=safe_float(annualized_return),
                benchmark_return=safe_float(benchmark_return),
                active_return=safe_float(active_return),
                tracking_error=safe_float(tracking_error),
                information_ratio=safe_float(information_ratio)
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
        # Normalize dates to UTC timezone (should already be normalized by caller)
        start_date = self._normalize_to_utc(start_date)
        end_date = self._normalize_to_utc(end_date)

        # Create date range in UTC
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Calculate beginning and ending values
        beginning_value = self._calculate_portfolio_value_at_date(
            portfolio, transactions, start_date
        )
        ending_value = portfolio.total_value

        # Ensure ending value is valid
        if not ending_value or ending_value <= 0:
            logger.warning(f"Invalid ending portfolio value: {ending_value}, using cost basis calculation")
            ending_value = self._calculate_portfolio_value_at_date(
                portfolio, transactions, end_date
            )

        # Use linear interpolation for daily values
        # Note: This is a simplification. True daily values would require daily price data
        values = pd.Series(
            np.linspace(beginning_value, ending_value, len(dates)),
            index=dates
        )

        # Build cash flows - only include actual buy/sell transactions (not auto CASH transactions)
        cash_flows = pd.Series(0.0, index=dates)
        for txn in transactions:
            txn_date = self._normalize_to_utc(txn.date)
            # Skip auto-generated CASH transactions
            if txn.symbol == 'CASH':
                continue
            if start_date <= txn_date <= end_date:
                if txn.transaction_type in ['buy', 'sell']:
                    # Use txn.amount directly - it's already signed correctly
                    # BUY: amount is negative (cash outflow)
                    # SELL: amount is positive (cash inflow)
                    cash_flows[txn_date] = txn.amount

        return values, cash_flows

    def _calculate_portfolio_value_at_date(
        self,
        portfolio: Portfolio,
        transactions: List[Transaction],
        date: datetime
    ) -> float:
        """Calculate portfolio value at a specific date by reconstructing holdings and using historical/current prices"""
        from collections import defaultdict

        # Normalize date to UTC
        date = self._normalize_to_utc(date)

        # Check inception date
        if portfolio.inception_date:
            inception = self._normalize_to_utc(portfolio.inception_date)
            if date < inception:
                return 0.0

        # Reconstruct holdings at the given date (similar to _calculate_portfolio_from_transactions)
        holdings_dict = defaultdict(lambda: {'quantity': 0.0, 'cost_basis': 0.0})

        print(f"\n>>> Reconstructing holdings at {date.date()}...")
        print(f"    Total transactions: {len(transactions)}")

        transactions_processed = 0
        transactions_skipped_cash = 0
        transactions_skipped_future = 0

        for txn in transactions:
            txn_date = self._normalize_to_utc(txn.date)

            # Skip CASH placeholder transactions
            if txn.symbol == 'CASH':
                transactions_skipped_cash += 1
                continue

            # Check date
            if txn_date > date:
                transactions_skipped_future += 1
                continue

            transactions_processed += 1
            symbol = txn.symbol

            if txn.transaction_type == 'buy':
                holdings_dict[symbol]['quantity'] += txn.quantity
                holdings_dict[symbol]['cost_basis'] += abs(txn.amount)
            elif txn.transaction_type == 'sell':
                holdings_dict[symbol]['quantity'] -= txn.quantity
                # Reduce cost basis proportionally
                if holdings_dict[symbol]['quantity'] > 0:
                    cost_per_share = holdings_dict[symbol]['cost_basis'] / (holdings_dict[symbol]['quantity'] + txn.quantity)
                    holdings_dict[symbol]['cost_basis'] -= (txn.quantity * cost_per_share)
                else:
                    holdings_dict[symbol]['cost_basis'] = 0

        print(f"    Processed: {transactions_processed}, Skipped CASH: {transactions_skipped_cash}, Skipped future: {transactions_skipped_future}")

        # Get prices - use current prices as approximation (in production, would use historical prices)
        symbols = [sym for sym, data in holdings_dict.items() if data['quantity'] > 0]
        if not symbols:
            print(f"    ⚠️  NO HOLDINGS FOUND - returning $0")
            return 0.0

        print(f"    Holdings found: {len(symbols)} securities")
        print(f"    Symbols: {', '.join(symbols)}")

        from services.market_data import market_data_service
        current_prices = market_data_service.get_current_prices(symbols)

        print(f"    Prices fetched: {len([p for p in current_prices.values() if p and p > 0])} / {len(symbols)}")

        # Calculate market value, falling back to cost basis if price unavailable
        total_value = 0.0
        for symbol, data in holdings_dict.items():
            if data['quantity'] > 0:
                price = current_prices.get(symbol)
                if price and price > 0:
                    # Use market price
                    value = data['quantity'] * price
                    total_value += value
                else:
                    # Fallback to cost basis if market price unavailable
                    print(f"    ⚠️  {symbol}: No price, using cost basis ${data['cost_basis']:,.2f}")
                    total_value += data['cost_basis']

        print(f"    >>> Total value calculated: ${total_value:,.2f}\n")
        return max(total_value, 0.0)

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
