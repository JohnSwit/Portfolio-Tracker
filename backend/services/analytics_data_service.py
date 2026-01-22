"""
Analytics data service - Handles data ingestion and retrieval for analytics

This service:
1. Computes and stores historical portfolio returns
2. Fetches and caches benchmark/factor returns
3. Computes volume data
4. Provides data to analytics modules
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
import sqlite3

from services.market_data import market_data_service
from database import get_db

logger = logging.getLogger(__name__)


class AnalyticsDataService:
    """Service for managing analytics data"""

    def compute_portfolio_returns(
        self,
        account_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.Series:
        """
        Compute historical portfolio returns from transactions and holdings

        Args:
            account_id: Account ID
            start_date: Start date
            end_date: End date

        Returns:
            pd.Series of daily returns indexed by date
        """
        # Check cache first
        cached = self._get_cached_portfolio_returns(account_id, start_date, end_date)
        if cached is not None:
            return cached

        # Compute from transactions and price data
        # This is a simplified version - in production, you'd use the
        # actual portfolio valuation over time

        logger.info(f"Computing portfolio returns for {account_id} from {start_date} to {end_date}")

        # Get transactions for the account
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT date, symbol, quantity, price, amount, transaction_type
                FROM transactions
                WHERE account_id = ? AND date BETWEEN ? AND ?
                ORDER BY date
            """, (account_id, start_date.isoformat(), end_date.isoformat()))

            transactions = cursor.fetchall()

        if not transactions:
            logger.warning(f"No transactions found for {account_id}")
            return pd.Series(dtype=float)

        # Build daily portfolio values
        # This is simplified - you'd want to use your existing performance calculation
        # For now, return empty series - hook this up to your performance_v2.py logic

        returns = pd.Series(dtype=float)

        # Store in cache
        self._cache_portfolio_returns(account_id, returns)

        return returns

    def get_benchmark_returns(
        self,
        benchmark_ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.Series:
        """
        Get benchmark returns, using cache when available

        Args:
            benchmark_ticker: Benchmark ticker (e.g., 'SPY')
            start_date: Start date
            end_date: End date

        Returns:
            pd.Series of daily returns indexed by date
        """
        # Check cache
        cached = self._get_cached_benchmark_returns(benchmark_ticker, start_date, end_date)
        if cached is not None and len(cached) > 0:
            return cached

        # Fetch from market data service
        logger.info(f"Fetching benchmark returns for {benchmark_ticker}")

        try:
            price_data = market_data_service.get_benchmark_data(
                benchmark_ticker,
                start_date,
                end_date
            )

            returns = price_data.pct_change(fill_method=None).dropna()

            # Cache the results
            self._cache_benchmark_returns(benchmark_ticker, returns, price_data)

            return returns

        except Exception as e:
            logger.error(f"Error fetching benchmark returns: {e}")
            return pd.Series(dtype=float)

    def get_factor_returns(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.Series]:
        """
        Get factor ETF returns for all standard factors

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Dict mapping factor name -> return series
        """
        from services.factor_analysis import FACTOR_ETFS

        factor_returns = {}

        for factor_name, etf_ticker in FACTOR_ETFS.items():
            # Check cache
            cached = self._get_cached_factor_returns(factor_name, start_date, end_date)
            if cached is not None and len(cached) > 0:
                factor_returns[factor_name] = cached
                continue

            # Fetch from market
            try:
                logger.info(f"Fetching factor returns for {factor_name} ({etf_ticker})")
                price_data = market_data_service.get_benchmark_data(
                    etf_ticker,
                    start_date,
                    end_date
                )

                returns = price_data.pct_change(fill_method=None).dropna()
                factor_returns[factor_name] = returns

                # Cache
                self._cache_factor_returns(factor_name, etf_ticker, returns, price_data)

            except Exception as e:
                logger.error(f"Error fetching factor {factor_name}: {e}")
                factor_returns[factor_name] = pd.Series(dtype=float)

        return factor_returns

    def get_volume_data(
        self,
        symbols: List[str],
        lookback_days: int = 20
    ) -> Dict[str, float]:
        """
        Get average daily volume for symbols

        Args:
            symbols: List of symbols
            lookback_days: Days to average over

        Returns:
            Dict mapping symbol -> avg_daily_volume
        """
        volume_data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 10)

        for symbol in symbols:
            # Check cache
            cached = self._get_cached_volume(symbol, lookback_days)
            if cached is not None:
                volume_data[symbol] = cached
                continue

            # Fetch from market
            try:
                # Use market_data_service to get volume
                price_data = market_data_service.get_price_data(
                    [symbol],
                    start_date,
                    end_date
                )

                if symbol in price_data and hasattr(price_data[symbol], 'volume'):
                    volumes = price_data[symbol]['volume']
                    avg_volume = float(volumes.tail(lookback_days).mean())
                    volume_data[symbol] = avg_volume

                    # Cache
                    self._cache_volume(symbol, avg_volume)
                else:
                    # Fallback to reasonable default
                    volume_data[symbol] = 1_000_000
                    logger.warning(f"No volume data for {symbol}, using default")

            except Exception as e:
                logger.error(f"Error fetching volume for {symbol}: {e}")
                volume_data[symbol] = 1_000_000  # Default

        return volume_data

    # Cache helper methods

    def _get_cached_portfolio_returns(
        self,
        account_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.Series]:
        """Get cached portfolio returns"""
        try:
            with get_db() as conn:
                df = pd.read_sql_query("""
                    SELECT date, daily_return
                    FROM portfolio_returns
                    WHERE account_id = ? AND date BETWEEN ? AND ?
                    ORDER BY date
                """, conn, params=(account_id, start_date.isoformat(), end_date.isoformat()))

                if len(df) == 0:
                    return None

                df['date'] = pd.to_datetime(df['date'])
                return pd.Series(df['daily_return'].values, index=df['date'])

        except Exception as e:
            logger.error(f"Error reading portfolio returns cache: {e}")
            return None

    def _cache_portfolio_returns(self, account_id: str, returns: pd.Series):
        """Cache portfolio returns"""
        if len(returns) == 0:
            return

        try:
            with get_db() as conn:
                cursor = conn.cursor()
                for date, ret in returns.items():
                    cursor.execute("""
                        INSERT OR REPLACE INTO portfolio_returns
                        (account_id, date, daily_return, portfolio_value)
                        VALUES (?, ?, ?, ?)
                    """, (account_id, date.isoformat(), float(ret), 0.0))
                conn.commit()
        except Exception as e:
            logger.error(f"Error caching portfolio returns: {e}")

    def _get_cached_benchmark_returns(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.Series]:
        """Get cached benchmark returns"""
        try:
            with get_db() as conn:
                df = pd.read_sql_query("""
                    SELECT date, daily_return
                    FROM benchmark_returns
                    WHERE benchmark_ticker = ? AND date BETWEEN ? AND ?
                    ORDER BY date
                """, conn, params=(ticker, start_date.isoformat(), end_date.isoformat()))

                if len(df) == 0:
                    return None

                df['date'] = pd.to_datetime(df['date'])
                return pd.Series(df['daily_return'].values, index=df['date'])

        except Exception as e:
            logger.error(f"Error reading benchmark returns cache: {e}")
            return None

    def _cache_benchmark_returns(self, ticker: str, returns: pd.Series, prices: pd.Series):
        """Cache benchmark returns"""
        if len(returns) == 0:
            return

        try:
            with get_db() as conn:
                cursor = conn.cursor()
                for date in returns.index:
                    if date in prices.index:
                        cursor.execute("""
                            INSERT OR REPLACE INTO benchmark_returns
                            (benchmark_ticker, date, daily_return, close_price)
                            VALUES (?, ?, ?, ?)
                        """, (ticker, date.isoformat(), float(returns[date]), float(prices[date])))
                conn.commit()
        except Exception as e:
            logger.error(f"Error caching benchmark returns: {e}")

    def _get_cached_factor_returns(
        self,
        factor_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.Series]:
        """Get cached factor returns"""
        try:
            with get_db() as conn:
                df = pd.read_sql_query("""
                    SELECT date, daily_return
                    FROM factor_returns
                    WHERE factor_name = ? AND date BETWEEN ? AND ?
                    ORDER BY date
                """, conn, params=(factor_name, start_date.isoformat(), end_date.isoformat()))

                if len(df) == 0:
                    return None

                df['date'] = pd.to_datetime(df['date'])
                return pd.Series(df['daily_return'].values, index=df['date'])

        except Exception as e:
            logger.error(f"Error reading factor returns cache: {e}")
            return None

    def _cache_factor_returns(
        self,
        factor_name: str,
        etf_ticker: str,
        returns: pd.Series,
        prices: pd.Series
    ):
        """Cache factor returns"""
        if len(returns) == 0:
            return

        try:
            with get_db() as conn:
                cursor = conn.cursor()
                for date in returns.index:
                    if date in prices.index:
                        cursor.execute("""
                            INSERT OR REPLACE INTO factor_returns
                            (factor_name, etf_ticker, date, daily_return, close_price)
                            VALUES (?, ?, ?, ?, ?)
                        """, (factor_name, etf_ticker, date.isoformat(),
                              float(returns[date]), float(prices[date])))
                conn.commit()
        except Exception as e:
            logger.error(f"Error caching factor returns: {e}")

    def _get_cached_volume(self, symbol: str, lookback_days: int) -> Optional[float]:
        """Get cached volume data"""
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT avg_volume_20d
                    FROM volume_data
                    WHERE symbol = ?
                    ORDER BY date DESC
                    LIMIT 1
                """, (symbol,))

                row = cursor.fetchone()
                if row and row[0]:
                    return float(row[0])
                return None

        except Exception as e:
            logger.error(f"Error reading volume cache: {e}")
            return None

    def _cache_volume(self, symbol: str, avg_volume: float):
        """Cache volume data"""
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO volume_data
                    (symbol, date, volume, avg_volume_20d)
                    VALUES (?, ?, ?, ?)
                """, (symbol, datetime.now().isoformat(), int(avg_volume), avg_volume))
                conn.commit()
        except Exception as e:
            logger.error(f"Error caching volume: {e}")


# Global instance
analytics_data_service = AnalyticsDataService()
