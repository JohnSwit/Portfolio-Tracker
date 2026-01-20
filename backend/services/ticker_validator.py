import yfinance as yf
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


class TickerValidator:
    """Validate stock ticker symbols"""

    @staticmethod
    @lru_cache(maxsize=1000)
    def validate_ticker(symbol: str) -> dict:
        """
        Validate that a ticker symbol is a valid publicly traded stock

        Args:
            symbol: Stock ticker symbol

        Returns:
            dict with 'valid' boolean and optional 'name' and 'error'
        """
        try:
            # Clean up symbol
            symbol = symbol.strip().upper()

            # Get ticker info
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Check if we got any info at all
            if not info:
                return {
                    'valid': False,
                    'symbol': symbol,
                    'error': f"Ticker '{symbol}' not found"
                }

            # Try to get name from multiple possible fields
            name = (info.get('longName') or
                   info.get('shortName') or
                   info.get('name') or
                   info.get('symbol', symbol))

            # Check if it's actually a stock (not index, currency, etc.)
            # Some tickers may not have quoteType, so we'll be lenient
            quote_type = info.get('quoteType', '').upper()
            valid_types = ['EQUITY', 'ETF', 'MUTUALFUND', 'STOCK']

            # Reject known invalid types, but allow empty quoteType
            invalid_types = ['INDEX', 'CURRENCY', 'CRYPTOCURRENCY', 'FUTURE', 'OPTION']
            if quote_type and quote_type in invalid_types:
                return {
                    'valid': False,
                    'symbol': symbol,
                    'error': f"'{symbol}' is a {quote_type}, not a stock/ETF"
                }

            # If quoteType exists and is not in valid types, but also not in invalid types,
            # do an additional check: see if we can get historical data
            if quote_type and quote_type not in valid_types:
                # Try to get recent price data as a fallback validation
                try:
                    hist = ticker.history(period="5d")
                    if hist.empty:
                        return {
                            'valid': False,
                            'symbol': symbol,
                            'error': f"No price data available for '{symbol}'"
                        }
                except:
                    return {
                        'valid': False,
                        'symbol': symbol,
                        'error': f"'{symbol}' appears to be invalid (type: {quote_type})"
                    }

            logger.info(f"Ticker '{symbol}' validated successfully: {name}")

            return {
                'valid': True,
                'symbol': symbol,
                'name': name,
                'type': quote_type or 'EQUITY'
            }

        except Exception as e:
            logger.error(f"Error validating ticker '{symbol}': {e}")
            return {
                'valid': False,
                'symbol': symbol,
                'error': f"Error validating ticker: {str(e)}"
            }

    @staticmethod
    def validate_multiple_tickers(symbols: list) -> dict:
        """
        Validate multiple ticker symbols

        Args:
            symbols: List of ticker symbols

        Returns:
            dict with 'valid' list, 'invalid' list with errors
        """
        valid = []
        invalid = []

        for symbol in symbols:
            result = TickerValidator.validate_ticker(symbol)
            if result['valid']:
                valid.append({
                    'symbol': result['symbol'],
                    'name': result['name']
                })
            else:
                invalid.append({
                    'symbol': result['symbol'],
                    'error': result['error']
                })

        return {
            'valid': valid,
            'invalid': invalid,
            'total': len(symbols),
            'valid_count': len(valid),
            'invalid_count': len(invalid)
        }


ticker_validator = TickerValidator()
