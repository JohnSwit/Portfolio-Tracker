"""
Performance calculation service - Complete rebuild
Implements proper Simple Return, TWR, and MWR calculations
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from scipy.optimize import brentq
import logging
from collections import defaultdict

from models.portfolio import Portfolio, Transaction
from services.market_data import market_data_service

logger = logging.getLogger(__name__)


class PerformanceCalculator:
    """Calculate performance metrics for portfolio and individual securities"""

    TIME_PERIODS = {
        '1M': 30,
        '3M': 90,
        '6M': 180,
        'YTD': None,  # Special case
        '1Y': 365,
        '3Y': 365 * 3,
        '5Y': 365 * 5,
        '10Y': 365 * 10,
    }

    @staticmethod
    def _normalize_to_utc(dt: datetime) -> datetime:
        """Normalize datetime to timezone-aware UTC"""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def calculate_performance_for_all_periods(
        self,
        portfolio: Portfolio,
        transactions: List[Transaction],
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Calculate performance for all time periods

        Returns dict with:
        {
            'portfolio': {period: {simple, twr, mwr}},
            'securities': {symbol: {period: {simple, twr, mwr}}},
            'time_series': {period: DataFrame with daily values}
        }
        """
        logger.info("=== PERFORMANCE_V2 CALCULATION START ===")
        print("=== PERFORMANCE_V2 CALCULATION START ===")

        if not end_date:
            end_date = datetime.now(timezone.utc)
        else:
            end_date = self._normalize_to_utc(end_date)

        logger.info(f"Portfolio value: ${portfolio.total_value:,.2f}")
        logger.info(f"Transactions count: {len(transactions)}")
        print(f"Portfolio value: ${portfolio.total_value:,.2f}, Transactions: {len(transactions)}")

        results = {
            'portfolio': {},
            'securities': {},
            'time_series': {}
        }

        # Get all unique symbols
        symbols = list(set([t.symbol for t in transactions if t.symbol != 'CASH']))
        logger.info(f"Unique symbols: {symbols}")
        print(f"Unique symbols: {symbols}")

        for period_label, days in self.TIME_PERIODS.items():
            logger.info(f"Processing period: {period_label}")
            print(f"Processing period: {period_label}")

            # Calculate start date
            if period_label == 'YTD':
                start_date = datetime(end_date.year, 1, 1, tzinfo=timezone.utc)
            else:
                start_date = end_date - timedelta(days=days)

            # Portfolio-level performance
            logger.info(f"  Calculating portfolio performance for {period_label}...")
            print(f"  [Step A] Portfolio-level for {period_label}...")
            results['portfolio'][period_label] = self._calculate_portfolio_performance(
                portfolio, transactions, start_date, end_date
            )
            print(f"  [Step A] Portfolio-level DONE")

            # Security-level performance
            print(f"  [Step B] Security-level for {period_label} ({len(symbols)} symbols)...")
            for i, symbol in enumerate(symbols):
                print(f"    [{i+1}/{len(symbols)}] Calculating {symbol}...")
                if symbol not in results['securities']:
                    results['securities'][symbol] = {}

                results['securities'][symbol][period_label] = self._calculate_security_performance(
                    symbol, transactions, start_date, end_date
                )
                print(f"    [{i+1}/{len(symbols)}] {symbol} DONE")
            print(f"  [Step B] Security-level DONE")

            # Time series for charting
            logger.info(f"  Building time series for {period_label}...")
            print(f"  [Step C] Time series for {period_label}...")
            results['time_series'][period_label] = self._build_time_series(
                portfolio, transactions, start_date, end_date
            )
            logger.info(f"  Completed {period_label}")
            print(f"  [Step C] Time series DONE")
            print(f"✓ Completed {period_label}\n")

        logger.info("=== PERFORMANCE_V2 CALCULATION COMPLETE ===")
        print("=== PERFORMANCE_V2 CALCULATION COMPLETE ===")
        return results

    def _calculate_portfolio_performance(
        self,
        portfolio: Portfolio,
        transactions: List[Transaction],
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Calculate Simple, TWR, and MWR for entire portfolio"""

        print(f"    [Portfolio Perf] Getting start value...")
        # Get portfolio value at start and end
        start_value = self._calculate_portfolio_value_at_date(transactions, start_date)
        end_value = portfolio.total_value
        print(f"    [Portfolio Perf] Start=${start_value:,.2f}, End=${end_value:,.2f}")

        print(f"    [Portfolio Perf] Getting cash flows...")
        # Get all cash flows in the period (excluding CASH placeholders)
        cash_flows = self._get_portfolio_cash_flows(transactions, start_date, end_date)
        print(f"    [Portfolio Perf] Found {len(cash_flows)} cash flows")

        print(f"    [Portfolio Perf] Calculating Simple Return...")
        # Calculate Simple Return (Modified Dietz)
        simple_return = self._calculate_simple_return(
            start_value, end_value, cash_flows, start_date, end_date
        )
        print(f"    [Portfolio Perf] Simple Return: {simple_return}")

        print(f"    [Portfolio Perf] Calculating TWR (this may take a moment)...")
        # Calculate TWR
        twr = self._calculate_twr_portfolio(
            transactions, start_date, end_date
        )
        print(f"    [Portfolio Perf] TWR: {twr}")

        print(f"    [Portfolio Perf] Calculating MWR...")
        # Calculate MWR (IRR)
        mwr = self._calculate_mwr(
            cash_flows, start_value, end_value, start_date, end_date
        )
        print(f"    [Portfolio Perf] MWR: {mwr}")

        return {
            'simple_return': simple_return,
            'twr': twr,
            'mwr': mwr,
            'start_value': start_value,
            'end_value': end_value,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat()
        }

    def _calculate_security_performance(
        self,
        symbol: str,
        transactions: List[Transaction],
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Calculate Simple, TWR, and MWR for a single security"""

        # Filter transactions for this symbol
        symbol_txns = [t for t in transactions if t.symbol == symbol]

        # Get security value at start and end
        start_value = self._calculate_security_value_at_date(
            symbol, transactions, start_date
        )
        end_value = self._calculate_security_value_at_date(
            symbol, transactions, end_date
        )

        # Get cash flows for this security
        cash_flows = self._get_security_cash_flows(symbol, transactions, start_date, end_date)

        # Calculate Simple Return
        simple_return = self._calculate_simple_return(
            start_value, end_value, cash_flows, start_date, end_date
        )

        # Calculate TWR
        twr = self._calculate_twr_security(
            symbol, transactions, start_date, end_date
        )

        # Calculate MWR
        mwr = self._calculate_mwr(
            cash_flows, start_value, end_value, start_date, end_date
        )

        return {
            'simple_return': simple_return,
            'twr': twr,
            'mwr': mwr,
            'start_value': start_value,
            'end_value': end_value
        }

    def _calculate_simple_return(
        self,
        start_value: float,
        end_value: float,
        cash_flows: List[Tuple[datetime, float]],
        start_date: datetime,
        end_date: datetime
    ) -> Optional[float]:
        """
        Calculate Simple Return using Modified Dietz method

        Return = (Ending Value - Beginning Value - Net Contributions) /
                 (Beginning Value + Weighted Contributions)
        """
        if start_value <= 0:
            return None

        total_days = (end_date - start_date).days
        if total_days == 0:
            return 0.0

        # Calculate weighted cash flows
        weighted_flows = 0.0
        net_flows = 0.0

        for cf_date, cf_amount in cash_flows:
            days_held = (end_date - cf_date).days
            weight = days_held / total_days
            weighted_flows += cf_amount * weight
            net_flows += cf_amount

        # Modified Dietz formula
        denominator = start_value + weighted_flows
        if denominator == 0:
            return None

        simple_return = (end_value - start_value - net_flows) / denominator

        return simple_return

    def _calculate_twr_portfolio(
        self,
        transactions: List[Transaction],
        start_date: datetime,
        end_date: datetime
    ) -> Optional[float]:
        """
        Calculate Time-Weighted Return for portfolio
        Chain-links sub-period returns between cash flow dates
        """
        # Get all cash flow dates in the period
        cf_dates = sorted(list(set([
            self._normalize_to_utc(t.date)
            for t in transactions
            if t.symbol != 'CASH' and start_date <= self._normalize_to_utc(t.date) <= end_date
        ])))

        # Add start and end dates
        all_dates = [start_date] + cf_dates + [end_date]
        all_dates = sorted(list(set(all_dates)))

        # Calculate sub-period returns
        returns = []
        for i in range(len(all_dates) - 1):
            period_start = all_dates[i]
            period_end = all_dates[i + 1]

            # Get portfolio value at start of sub-period
            value_start = self._calculate_portfolio_value_at_date(transactions, period_start)

            # Get portfolio value at end of sub-period (before flows)
            value_end_before_flows = self._calculate_portfolio_value_at_date(transactions, period_end)

            # Get net flows on end date
            flows_on_date = sum([
                t.amount for t in transactions
                if t.symbol != 'CASH' and self._normalize_to_utc(t.date) == period_end
            ])

            # Value at end before flows
            value_end = value_end_before_flows - flows_on_date

            # Calculate sub-period return
            if value_start > 0:
                period_return = (value_end - value_start) / value_start
                returns.append(period_return)

        # Chain-link returns
        if not returns:
            return 0.0

        twr = 1.0
        for r in returns:
            twr *= (1.0 + r)
        twr -= 1.0

        return twr

    def _calculate_twr_security(
        self,
        symbol: str,
        transactions: List[Transaction],
        start_date: datetime,
        end_date: datetime
    ) -> Optional[float]:
        """Calculate TWR for a single security"""
        # Get cash flow dates for this security
        cf_dates = sorted(list(set([
            self._normalize_to_utc(t.date)
            for t in transactions
            if t.symbol == symbol and start_date <= self._normalize_to_utc(t.date) <= end_date
        ])))

        if not cf_dates:
            # No transactions in period, just price change
            start_value = self._calculate_security_value_at_date(symbol, transactions, start_date)
            end_value = self._calculate_security_value_at_date(symbol, transactions, end_date)

            if start_value > 0:
                return (end_value - start_value) / start_value
            return None

        # Add start and end dates
        all_dates = [start_date] + cf_dates + [end_date]
        all_dates = sorted(list(set(all_dates)))

        # Calculate sub-period returns
        returns = []
        for i in range(len(all_dates) - 1):
            period_start = all_dates[i]
            period_end = all_dates[i + 1]

            value_start = self._calculate_security_value_at_date(symbol, transactions, period_start)
            value_end_before_flows = self._calculate_security_value_at_date(symbol, transactions, period_end)

            # Get flows on end date for this symbol
            flows_on_date = sum([
                t.amount for t in transactions
                if t.symbol == symbol and self._normalize_to_utc(t.date) == period_end
            ])

            value_end = value_end_before_flows - flows_on_date

            if value_start > 0:
                period_return = (value_end - value_start) / value_start
                returns.append(period_return)

        # Chain-link
        if not returns:
            return 0.0

        twr = 1.0
        for r in returns:
            twr *= (1.0 + r)
        twr -= 1.0

        return twr

    def _calculate_mwr(
        self,
        cash_flows: List[Tuple[datetime, float]],
        start_value: float,
        end_value: float,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[float]:
        """
        Calculate Money-Weighted Return (IRR/XIRR)
        Solves: sum(CF_i / (1+IRR)^(days_i/365)) + EndValue/(1+IRR)^(days_end/365) - StartValue = 0
        """
        if start_value <= 0:
            return None

        total_days = (end_date - start_date).days
        if total_days == 0:
            return 0.0

        # Build NPV function
        def npv(rate):
            pv = -start_value  # Initial investment (negative)

            # Add discounted cash flows
            for cf_date, cf_amount in cash_flows:
                days_from_start = (cf_date - start_date).days
                years = days_from_start / 365.0
                pv += cf_amount / ((1 + rate) ** years)

            # Add discounted ending value
            years_total = total_days / 365.0
            pv += end_value / ((1 + rate) ** years_total)

            return pv

        # Try to solve for IRR
        try:
            # Try Brent's method with reasonable bounds
            irr = brentq(npv, -0.99, 10.0, xtol=1e-6, maxiter=100)
            return irr
        except (ValueError, RuntimeError):
            # IRR doesn't exist or can't be found
            # Fall back to simple approximation
            net_flows = sum([cf[1] for cf in cash_flows])
            if start_value + net_flows > 0:
                return (end_value - start_value - net_flows) / (start_value + net_flows)
            return None

    def _calculate_portfolio_value_at_date(
        self,
        transactions: List[Transaction],
        date: datetime
    ) -> float:
        """Calculate total portfolio value at a specific date"""
        date = self._normalize_to_utc(date)

        # Reconstruct holdings at date
        holdings = defaultdict(lambda: {'quantity': 0.0, 'cost_basis': 0.0})

        for txn in transactions:
            txn_date = self._normalize_to_utc(txn.date)
            if txn.symbol == 'CASH':
                continue

            if txn_date <= date:
                symbol = txn.symbol
                if txn.transaction_type == 'buy':
                    holdings[symbol]['quantity'] += txn.quantity
                    holdings[symbol]['cost_basis'] += abs(txn.amount)
                elif txn.transaction_type == 'sell':
                    holdings[symbol]['quantity'] -= txn.quantity
                    if holdings[symbol]['quantity'] > 0:
                        cost_per_share = holdings[symbol]['cost_basis'] / (holdings[symbol]['quantity'] + txn.quantity)
                        holdings[symbol]['cost_basis'] -= (txn.quantity * cost_per_share)
                    else:
                        holdings[symbol]['cost_basis'] = 0

        # Get current prices
        symbols = [sym for sym, data in holdings.items() if data['quantity'] > 0]
        if not symbols:
            return 0.0

        prices = market_data_service.get_current_prices(symbols)

        # Calculate total value
        total = 0.0
        for symbol, data in holdings.items():
            if data['quantity'] > 0:
                price = prices.get(symbol, 0)
                if price > 0:
                    total += data['quantity'] * price
                else:
                    # Fallback to cost basis
                    total += data['cost_basis']

        return total

    def _calculate_security_value_at_date(
        self,
        symbol: str,
        transactions: List[Transaction],
        date: datetime
    ) -> float:
        """Calculate value of a specific security at a date"""
        date = self._normalize_to_utc(date)

        # Get quantity held at date
        quantity = 0.0
        for txn in transactions:
            txn_date = self._normalize_to_utc(txn.date)
            if txn.symbol == symbol and txn_date <= date:
                if txn.transaction_type == 'buy':
                    quantity += txn.quantity
                elif txn.transaction_type == 'sell':
                    quantity -= txn.quantity

        if quantity <= 0:
            return 0.0

        # Get price at date (using current price as approximation)
        prices = market_data_service.get_current_prices([symbol])
        price = prices.get(symbol, 0)

        return quantity * price

    def _get_portfolio_cash_flows(
        self,
        transactions: List[Transaction],
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, float]]:
        """Get all portfolio cash flows in period"""
        flows = []
        for txn in transactions:
            txn_date = self._normalize_to_utc(txn.date)
            if txn.symbol != 'CASH' and start_date < txn_date <= end_date:
                flows.append((txn_date, txn.amount))

        return flows

    def _get_security_cash_flows(
        self,
        symbol: str,
        transactions: List[Transaction],
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, float]]:
        """Get cash flows for a specific security"""
        flows = []
        for txn in transactions:
            txn_date = self._normalize_to_utc(txn.date)
            if txn.symbol == symbol and start_date < txn_date <= end_date:
                flows.append((txn_date, txn.amount))

        return flows

    def _build_time_series(
        self,
        portfolio: Portfolio,
        transactions: List[Transaction],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Build daily time series for charting"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        simple_returns = []
        twr_returns = []
        mwr_returns = []

        # Calculate cumulative returns from start to each date
        for current_date in dates:
            current_date_dt = current_date.to_pydatetime().replace(tzinfo=timezone.utc)

            # Calculate returns from start_date to current_date
            perf = self._calculate_portfolio_performance(
                portfolio, transactions, start_date, current_date_dt
            )

            simple_returns.append(perf['simple_return'] if perf['simple_return'] is not None else 0)
            twr_returns.append(perf['twr'] if perf['twr'] is not None else 0)
            mwr_returns.append(perf['mwr'] if perf['mwr'] is not None else 0)

        df = pd.DataFrame({
            'date': dates,
            'simple_return': simple_returns,
            'twr': twr_returns,
            'mwr': mwr_returns
        })

        return df


# Global instance
performance_calculator = PerformanceCalculator()
