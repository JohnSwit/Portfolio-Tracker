import os
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

try:
    from schwab import auth, client
    SCHWAB_AVAILABLE = True
except ImportError:
    SCHWAB_AVAILABLE = False
    logging.warning("Schwab API not available. Install schwab-py to use Schwab integration.")

from models.portfolio import Portfolio, Holding, Transaction

logger = logging.getLogger(__name__)


class SchwabClient:
    """Client for interacting with Schwab API"""

    def __init__(self):
        self.api_key = os.getenv("SCHWAB_API_KEY")
        self.secret = os.getenv("SCHWAB_SECRET")
        self.callback_url = os.getenv("SCHWAB_CALLBACK_URL", "https://localhost:8080/callback")
        self.token_path = "./schwab_token.json"
        self._client = None

    def initialize(self):
        """Initialize Schwab API client with authentication"""
        if not SCHWAB_AVAILABLE:
            raise RuntimeError("Schwab API not available. Install schwab-py package.")

        if not self.api_key or not self.secret:
            raise ValueError("Schwab API credentials not configured")

        try:
            # Try to use existing token
            if os.path.exists(self.token_path):
                self._client = auth.client_from_token_file(
                    self.token_path,
                    self.api_key
                )
            else:
                # Need to authenticate
                self._client = auth.client_from_manual_flow(
                    self.api_key,
                    self.secret,
                    self.callback_url,
                    self.token_path
                )

            logger.info("Schwab client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Schwab client: {e}")
            raise

    def get_accounts(self) -> List[Dict]:
        """Get list of Schwab accounts"""
        if not self._client:
            self.initialize()

        try:
            response = self._client.get_accounts()
            accounts = response.json()
            return accounts
        except Exception as e:
            logger.error(f"Error fetching accounts: {e}")
            raise

    def get_portfolio(self, account_id: str) -> Portfolio:
        """Get portfolio holdings for an account"""
        if not self._client:
            self.initialize()

        try:
            # Get account details with positions
            response = self._client.get_account(
                account_id,
                fields=client.Client.Account.Fields.POSITIONS
            )
            account_data = response.json()

            # Parse holdings
            holdings = []
            cash_balance = 0.0

            if 'securitiesAccount' in account_data:
                account = account_data['securitiesAccount']

                # Get cash balance
                if 'currentBalances' in account:
                    cash_balance = account['currentBalances'].get('cashBalance', 0.0)

                # Parse positions
                if 'positions' in account:
                    for position in account['positions']:
                        instrument = position.get('instrument', {})
                        symbol = instrument.get('symbol', 'UNKNOWN')

                        holding = Holding(
                            symbol=symbol,
                            quantity=position.get('longQuantity', 0.0),
                            cost_basis=position.get('averagePrice', 0.0) * position.get('longQuantity', 0.0),
                            current_price=position.get('marketValue', 0.0) / position.get('longQuantity', 1.0),
                            market_value=position.get('marketValue', 0.0),
                            asset_class=instrument.get('assetType', 'equity').lower()
                        )
                        holdings.append(holding)

            portfolio = Portfolio(
                account_id=account_id,
                account_name=account_data.get('securitiesAccount', {}).get('accountNumber'),
                holdings=holdings,
                cash_balance=cash_balance,
                total_value=sum(h.market_value or 0 for h in holdings) + cash_balance,
                last_updated=datetime.now()
            )

            return portfolio

        except Exception as e:
            logger.error(f"Error fetching portfolio for account {account_id}: {e}")
            raise

    def get_transactions(
        self,
        account_id: str,
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> List[Transaction]:
        """Get transaction history for an account"""
        if not self._client:
            self.initialize()

        if end_date is None:
            end_date = datetime.now()

        try:
            response = self._client.get_transactions(
                account_id,
                start_date=start_date,
                end_date=end_date
            )
            transactions_data = response.json()

            transactions = []
            for txn in transactions_data:
                transaction_type = txn.get('type', '').lower()

                # Map Schwab transaction types to our types
                type_mapping = {
                    'trade': 'buy' if txn.get('netAmount', 0) < 0 else 'sell',
                    'dividend': 'dividend',
                    'split': 'split'
                }

                transaction = Transaction(
                    date=datetime.fromisoformat(txn.get('transactionDate').replace('Z', '+00:00')),
                    symbol=txn.get('transactionItem', {}).get('instrument', {}).get('symbol', 'UNKNOWN'),
                    transaction_type=type_mapping.get(transaction_type, transaction_type),
                    quantity=abs(txn.get('transactionItem', {}).get('amount', 0.0)),
                    price=txn.get('transactionItem', {}).get('price', 0.0),
                    amount=txn.get('netAmount', 0.0),
                    fees=txn.get('fees', {}).get('commission', 0.0),
                    notes=txn.get('description')
                )
                transactions.append(transaction)

            return transactions

        except Exception as e:
            logger.error(f"Error fetching transactions: {e}")
            raise


# Global instance
schwab_client = SchwabClient()
