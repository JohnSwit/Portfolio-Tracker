import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class MarketDataService:
    """Service for retrieving market data from various sources"""

    def __init__(self):
        self.cache_duration = timedelta(minutes=5)
        self._cache: Dict[str, Tuple[datetime, pd.DataFrame]] = {}

    def get_price_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical price data for multiple symbols

        Args:
            symbols: List of ticker symbols
            start_date: Start date for data
            end_date: End date (defaults to today)
            interval: Data interval (1d, 1h, etc.)

        Returns:
            DataFrame with price data
        """
        if end_date is None:
            end_date = datetime.now()

        try:
            data = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                interval=interval,
                group_by='ticker',
                auto_adjust=True,
                progress=False
            )
            return data
        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            raise

    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for symbols"""
        try:
            tickers = yf.Tickers(' '.join(symbols))
            prices = {}
            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.info
                    prices[symbol] = info.get('currentPrice') or info.get('regularMarketPrice', 0.0)
                except Exception as e:
                    logger.warning(f"Error fetching price for {symbol}: {e}")
                    prices[symbol] = 0.0
            return prices
        except Exception as e:
            logger.error(f"Error fetching current prices: {e}")
            return {symbol: 0.0 for symbol in symbols}

    def get_stock_info(self, symbol: str) -> Dict:
        """Get detailed information about a stock"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'symbol': symbol,
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'country': info.get('country'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'avg_volume': info.get('averageVolume'),
                'current_price': info.get('currentPrice') or info.get('regularMarketPrice'),
                '52w_high': info.get('fiftyTwoWeekHigh'),
                '52w_low': info.get('fiftyTwoWeekLow'),
            }
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {'symbol': symbol}

    def get_benchmark_data(
        self,
        benchmark: str,
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> pd.Series:
        """
        Get benchmark return data

        Common benchmarks:
        - ^GSPC (S&P 500)
        - ^DJI (Dow Jones)
        - ^IXIC (NASDAQ)
        - ^RUT (Russell 2000)
        """
        if end_date is None:
            end_date = datetime.now()

        try:
            data = yf.download(
                benchmark,
                start=start_date,
                end=end_date,
                progress=False
            )
            # For single symbol, ensure 1D series
            close_data = data['Close']
            if hasattr(close_data, 'squeeze'):
                close_data = close_data.squeeze()
            return close_data
        except Exception as e:
            logger.error(f"Error fetching benchmark data for {benchmark}: {e}")
            raise

    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate simple returns from price series"""
        return prices.pct_change().fillna(0)

    def get_risk_free_rate(self) -> float:
        """Get current risk-free rate (using 10-year Treasury)"""
        try:
            ticker = yf.Ticker("^TNX")
            info = ticker.info
            rate = info.get('regularMarketPrice', 4.0) / 100  # Convert to decimal
            return rate
        except Exception as e:
            logger.warning(f"Error fetching risk-free rate, using default: {e}")
            return 0.04  # Default 4%

    def get_factor_data(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get factor return data (Fama-French style factors)

        Using ETF proxies:
        - Market: SPY
        - Size (SMB): IWM - SPY
        - Value (HML): VTV - VUG
        - Momentum: MTUM
        - Quality: QUAL
        """
        if end_date is None:
            end_date = datetime.now()

        factor_etfs = {
            'Market': 'SPY',
            'Small': 'IWM',
            'Large': 'SPY',
            'Value': 'VTV',
            'Growth': 'VUG',
            'Momentum': 'MTUM',
            'Quality': 'QUAL',
        }

        try:
            data = yf.download(
                list(factor_etfs.values()),
                start=start_date,
                end=end_date,
                progress=False
            )['Close']

            # Calculate factor returns
            factors = pd.DataFrame()
            factors['Market'] = data['SPY'].pct_change()
            factors['Size'] = data['IWM'].pct_change() - data['SPY'].pct_change()  # SMB
            factors['Value'] = data['VTV'].pct_change() - data['VUG'].pct_change()  # HML
            factors['Momentum'] = data['MTUM'].pct_change()
            factors['Quality'] = data['QUAL'].pct_change()

            return factors.fillna(0)
        except Exception as e:
            logger.error(f"Error fetching factor data: {e}")
            raise

    def get_correlation_matrix(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Calculate correlation matrix for given symbols"""
        price_data = self.get_price_data(symbols, start_date, end_date)

        if len(symbols) == 1:
            closes = price_data['Close']
        else:
            closes = pd.DataFrame({symbol: price_data[symbol]['Close'] for symbol in symbols})

        returns = closes.pct_change().dropna()
        return returns.corr()


# Global instance
market_data_service = MarketDataService()
