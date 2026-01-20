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

            # Check if it's a valid stock
            # Valid stocks should have basic info like longName or shortName
            if not info or len(info) < 5:
                return {
                    'valid': False,
                    'symbol': symbol,
                    'error': f"Ticker '{symbol}' not found or invalid"
                }

            # Additional validation - check for key fields
            name = info.get('longName') or info.get('shortName')
            if not name:
                return {
                    'valid': False,
                    'symbol': symbol,
                    'error': f"Ticker '{symbol}' appears to be invalid"
                }

            # Check if it's actually a stock (not index, currency, etc.)
            quote_type = info.get('quoteType', '').upper()
            valid_types = ['EQUITY', 'ETF', 'MUTUALFUND']

            if quote_type and quote_type not in valid_types:
                return {
                    'valid': False,
                    'symbol': symbol,
                    'error': f"'{symbol}' is a {quote_type}, not a stock/ETF"
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
