from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import logging
import os
from dotenv import load_dotenv

from models.portfolio import (
    Portfolio, PerformanceMetrics, AttributionResult,
    RiskMetrics, HoldingsAnalysis, StressTestScenario, StressTestResult, Holding, Transaction
)
from services.schwab_client import schwab_client, SCHWAB_AVAILABLE
from services.performance import performance_service
from services.risk import risk_service, STRESS_SCENARIOS
from services.holdings import holdings_service
from services.market_data import market_data_service
from services.csv_parser import csv_parser
from services.ticker_validator import ticker_validator
import database

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Portfolio Analytics API",
    description="Comprehensive portfolio management and analytics platform",
    version="1.0.0"
)

# CORS middleware
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class DateRangeQuery(BaseModel):
    start_date: datetime
    end_date: Optional[datetime] = None
    benchmark: str = "^GSPC"


class PortfolioRequest(BaseModel):
    account_id: str


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Portfolio Analytics API",
        "version": "1.0.0",
        "schwab_available": SCHWAB_AVAILABLE
    }


# Health check
@app.get("/health")
async def health():
    return {"status": "healthy"}


# Schwab Integration Endpoints
@app.get("/api/accounts")
async def get_accounts():
    """Get list of Schwab accounts"""
    if not SCHWAB_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Schwab integration not available. Install schwab-py package."
        )

    try:
        accounts = schwab_client.get_accounts()
        return {"accounts": accounts}
    except Exception as e:
        logger.error(f"Error fetching accounts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio/{account_id}", response_model=Portfolio)
async def get_portfolio(account_id: str):
    """Get portfolio holdings for an account"""
    try:
        # Try to get transactions from database first
        db_transactions = database.get_transactions(account_id)

        if db_transactions:
            # Calculate portfolio from transactions
            return _calculate_portfolio_from_transactions(account_id, db_transactions)

        # Fall back to Schwab API
        if SCHWAB_AVAILABLE:
            portfolio = schwab_client.get_portfolio(account_id)
            return portfolio

        # Return mock data for development/testing
        return _get_mock_portfolio(account_id)

    except Exception as e:
        logger.error(f"Error fetching portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/transactions/{account_id}")
async def get_transactions(
    account_id: str,
    days: int = Query(default=365, description="Number of days of history")
):
    """Get transaction history"""
    try:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        # Try database first
        db_transactions = database.get_transactions(
            account_id,
            start_date.isoformat(),
            end_date.isoformat()
        )

        if db_transactions:
            # Convert to Transaction objects
            transactions = [
                Transaction(
                    date=datetime.fromisoformat(t['date']),
                    symbol=t['symbol'],
                    transaction_type=t['transaction_type'],
                    quantity=t['quantity'],
                    price=t['price'],
                    amount=t['amount'],
                    fees=t['fees'],
                    notes=t.get('notes', '')
                )
                for t in db_transactions
            ]
            return {"transactions": transactions}

        # Fall back to Schwab API
        if SCHWAB_AVAILABLE:
            transactions = schwab_client.get_transactions(account_id, start_date, end_date)
            return {"transactions": transactions}

        # Return empty for mock data
        return {"transactions": []}

    except Exception as e:
        logger.error(f"Error fetching transactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Performance Analytics Endpoints
@app.post("/api/performance/{account_id}", response_model=PerformanceMetrics)
async def calculate_performance(
    account_id: str,
    date_range: DateRangeQuery
):
    """Calculate performance metrics"""
    try:
        portfolio = await get_portfolio(account_id)
        transactions_response = await get_transactions(
            account_id,
            days=(datetime.now(timezone.utc) - date_range.start_date).days
        )
        transactions = transactions_response.get("transactions", [])

        logger.info(f"Performance endpoint called:")
        logger.info(f"  Account: {account_id}")
        logger.info(f"  Portfolio total_value: ${portfolio.total_value:,.2f}")
        logger.info(f"  Portfolio holdings count: {len(portfolio.holdings)}")
        logger.info(f"  Transactions retrieved: {len(transactions)}")
        if transactions:
            logger.info(f"  First transaction: {transactions[0].symbol} on {transactions[0].date}")
            logger.info(f"  Last transaction: {transactions[-1].symbol} on {transactions[-1].date}")

        end_date = date_range.end_date or datetime.now(timezone.utc)

        metrics = performance_service.calculate_performance_metrics(
            portfolio,
            transactions,
            date_range.start_date,
            end_date,
            date_range.benchmark
        )

        return metrics
    except Exception as e:
        logger.error(f"Error calculating performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/attribution/{account_id}", response_model=AttributionResult)
async def calculate_attribution(
    account_id: str,
    date_range: DateRangeQuery
):
    """Calculate return attribution"""
    try:
        portfolio = await get_portfolio(account_id)
        transactions_response = await get_transactions(
            account_id,
            days=(datetime.now(timezone.utc) - date_range.start_date).days
        )
        transactions = transactions_response.get("transactions", [])

        end_date = date_range.end_date or datetime.now(timezone.utc)

        attribution = performance_service.calculate_attribution(
            portfolio,
            transactions,
            date_range.start_date,
            end_date
        )

        return attribution
    except Exception as e:
        logger.error(f"Error calculating attribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Risk Analytics Endpoints
@app.post("/api/risk/{account_id}", response_model=RiskMetrics)
async def calculate_risk(
    account_id: str,
    date_range: DateRangeQuery
):
    """Calculate risk metrics"""
    try:
        portfolio = await get_portfolio(account_id)
        end_date = date_range.end_date or datetime.now(timezone.utc)

        metrics = risk_service.calculate_risk_metrics(
            portfolio,
            date_range.start_date,
            end_date,
            date_range.benchmark
        )

        return metrics
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stress-test/{account_id}", response_model=List[StressTestResult])
async def run_stress_tests(account_id: str):
    """Run predefined stress test scenarios"""
    try:
        portfolio = await get_portfolio(account_id)

        results = risk_service.run_scenario_analysis(
            portfolio,
            STRESS_SCENARIOS
        )

        return results
    except Exception as e:
        logger.error(f"Error running stress tests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stress-test/{account_id}/custom", response_model=StressTestResult)
async def run_custom_stress_test(
    account_id: str,
    scenario: StressTestScenario
):
    """Run custom stress test scenario"""
    try:
        portfolio = await get_portfolio(account_id)
        result = risk_service.stress_test(portfolio, scenario)
        return result
    except Exception as e:
        logger.error(f"Error running custom stress test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Holdings Analysis Endpoints
@app.get("/api/holdings/{account_id}", response_model=HoldingsAnalysis)
async def analyze_holdings(account_id: str):
    """Analyze portfolio holdings"""
    try:
        portfolio = await get_portfolio(account_id)
        analysis = holdings_service.analyze_holdings(portfolio)
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing holdings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Market Data Endpoints
@app.get("/api/market/prices")
async def get_current_prices(symbols: str = Query(..., description="Comma-separated symbols")):
    """Get current market prices"""
    try:
        symbol_list = [s.strip() for s in symbols.split(",")]
        prices = market_data_service.get_current_prices(symbol_list)
        return {"prices": prices}
    except Exception as e:
        logger.error(f"Error fetching prices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/info/{symbol}")
async def get_stock_info(symbol: str):
    """Get detailed stock information"""
    try:
        info = market_data_service.get_stock_info(symbol)
        return info
    except Exception as e:
        logger.error(f"Error fetching stock info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# CSV Import Endpoints
@app.post("/api/upload-csv/{account_id}")
async def upload_csv(account_id: str, file: UploadFile = File(...)):
    """Upload CSV file with transactions"""
    try:
        # Read file content
        content = await file.read()
        file_content = content.decode('utf-8')

        # Parse CSV
        transactions = csv_parser.parse_transactions_csv(file_content)

        # Auto-generate offsetting cash transactions to keep cash balance at zero
        all_transactions = []
        for txn in transactions:
            all_transactions.append(txn)

            txn_type = txn['transaction_type']
            if txn_type == 'buy':
                # For buy: deposit cash to fund the purchase
                cash_amount = txn['quantity'] * txn['price'] + txn['fees']
                cash_txn = {
                    'date': txn['date'],
                    'symbol': 'CASH',
                    'transaction_type': 'deposit',
                    'quantity': cash_amount,
                    'price': 1.0,
                    'amount': cash_amount,
                    'fees': 0.0,
                    'notes': f"Auto-deposit to fund {txn['symbol']} purchase"
                }
                all_transactions.append(cash_txn)
            elif txn_type == 'sell':
                # For sell: withdraw cash from the sale proceeds
                cash_amount = txn['quantity'] * txn['price'] - txn['fees']
                cash_txn = {
                    'date': txn['date'],
                    'symbol': 'CASH',
                    'transaction_type': 'withdrawal',
                    'quantity': cash_amount,
                    'price': 1.0,
                    'amount': -cash_amount,
                    'fees': 0.0,
                    'notes': f"Auto-withdrawal of {txn['symbol']} sale proceeds"
                }
                all_transactions.append(cash_txn)
            elif txn_type == 'dividend':
                # For dividend: withdraw the dividend cash
                cash_amount = txn['quantity'] * txn['price']
                cash_txn = {
                    'date': txn['date'],
                    'symbol': 'CASH',
                    'transaction_type': 'withdrawal',
                    'quantity': cash_amount,
                    'price': 1.0,
                    'amount': -cash_amount,
                    'fees': 0.0,
                    'notes': f"Auto-withdrawal of {txn['symbol']} dividend"
                }
                all_transactions.append(cash_txn)

        # Save to database
        inserted = database.save_transactions(account_id, all_transactions)

        # Create or update portfolio record
        database.get_or_create_portfolio(account_id, f"Portfolio {account_id}")

        return {
            "success": True,
            "transactions_imported": len(transactions),
            "total_transactions_saved": inserted,
            "message": f"Successfully imported {len(transactions)} transactions (with auto cash management)"
        }

    except ValueError as e:
        logger.error(f"CSV validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error uploading CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download-template")
async def download_template():
    """Download CSV template file"""
    template = csv_parser.generate_template_csv()
    return Response(
        content=template,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=transactions_template.csv"}
    )


@app.delete("/api/transactions/{account_id}")
async def delete_transactions(account_id: str):
    """Delete all transactions for an account"""
    try:
        deleted = database.delete_all_transactions(account_id)
        return {
            "success": True,
            "transactions_deleted": deleted,
            "message": f"Deleted {deleted} transactions"
        }
    except Exception as e:
        logger.error(f"Error deleting transactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/transaction/{account_id}")
async def add_transaction(account_id: str, transaction: dict):
    """Add a single transaction manually"""
    try:
        # Validate required fields
        required_fields = ['date', 'symbol', 'type', 'quantity', 'price']
        missing = [f for f in required_fields if f not in transaction]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {', '.join(missing)}"
            )

        # Validate ticker symbol for stock transactions (unless skip_validation is true)
        txn_type = transaction['type'].lower()
        symbol = transaction['symbol'].strip().upper()
        skip_validation = transaction.get('skip_validation', False)

        if txn_type in ['buy', 'sell', 'dividend', 'split'] and not skip_validation:
            validation = ticker_validator.validate_ticker(symbol)
            if not validation['valid']:
                raise HTTPException(status_code=400, detail=validation['error'])

        # Parse and validate date
        try:
            date = datetime.fromisoformat(transaction['date'])
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

        # Validate numeric fields
        try:
            quantity = float(transaction['quantity'])
            price = float(transaction['price'])
            fees = float(transaction.get('fees', 0))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid numeric values")

        if quantity <= 0:
            raise HTTPException(status_code=400, detail="Quantity must be positive")
        if price < 0:
            raise HTTPException(status_code=400, detail="Price cannot be negative")

        # Calculate amount
        if txn_type == 'buy':
            amount = -(quantity * price + fees)
        elif txn_type == 'sell':
            amount = quantity * price - fees
        elif txn_type == 'dividend':
            amount = quantity * price
        elif txn_type == 'deposit':
            amount = quantity * price
        elif txn_type == 'withdrawal':
            amount = -(quantity * price)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid transaction type: {txn_type}"
            )

        # Save to database
        txn_data = {
            'date': date.isoformat(),
            'symbol': symbol,
            'transaction_type': txn_type,
            'quantity': quantity,
            'price': price,
            'amount': amount,
            'fees': fees,
            'notes': transaction.get('notes', '')
        }

        # Auto-generate offsetting cash transaction to keep cash balance at zero
        transactions_to_save = [txn_data]

        if txn_type == 'buy':
            # For buy: deposit cash to fund the purchase
            cash_amount = quantity * price + fees
            cash_txn = {
                'date': date.isoformat(),
                'symbol': 'CASH',
                'transaction_type': 'deposit',
                'quantity': cash_amount,
                'price': 1.0,
                'amount': cash_amount,
                'fees': 0.0,
                'notes': f'Auto-deposit to fund {symbol} purchase'
            }
            transactions_to_save.append(cash_txn)
        elif txn_type == 'sell':
            # For sell: withdraw cash from the sale proceeds
            cash_amount = quantity * price - fees
            cash_txn = {
                'date': date.isoformat(),
                'symbol': 'CASH',
                'transaction_type': 'withdrawal',
                'quantity': cash_amount,
                'price': 1.0,
                'amount': -cash_amount,
                'fees': 0.0,
                'notes': f'Auto-withdrawal of {symbol} sale proceeds'
            }
            transactions_to_save.append(cash_txn)
        elif txn_type == 'dividend':
            # For dividend: withdraw the dividend cash
            cash_amount = quantity * price
            cash_txn = {
                'date': date.isoformat(),
                'symbol': 'CASH',
                'transaction_type': 'withdrawal',
                'quantity': cash_amount,
                'price': 1.0,
                'amount': -cash_amount,
                'fees': 0.0,
                'notes': f'Auto-withdrawal of {symbol} dividend'
            }
            transactions_to_save.append(cash_txn)

        database.save_transactions(account_id, transactions_to_save)
        database.get_or_create_portfolio(account_id)

        return {
            "success": True,
            "message": f"Transaction added successfully (with {len(transactions_to_save)} total transactions)",
            "transaction": txn_data,
            "auto_cash_transactions": len(transactions_to_save) - 1
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding transaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/transaction-list/{account_id}")
async def get_transaction_list(account_id: str, limit: int = Query(default=100, le=500)):
    """Get list of all transactions for display"""
    try:
        transactions = database.get_transactions(account_id)

        # Limit results
        transactions = transactions[:limit] if limit else transactions

        return {
            "success": True,
            "count": len(transactions),
            "transactions": transactions
        }
    except Exception as e:
        logger.error(f"Error fetching transaction list: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/validate-ticker/{symbol}")
async def validate_ticker(symbol: str):
    """Validate a ticker symbol"""
    try:
        result = ticker_validator.validate_ticker(symbol)
        return result
    except Exception as e:
        logger.error(f"Error validating ticker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/debug/portfolio/{account_id}")
async def debug_portfolio(account_id: str):
    """Debug endpoint to see portfolio and transaction details"""
    try:
        # Get portfolio
        portfolio = await get_portfolio(account_id)

        # Get all transactions
        db_transactions = database.get_transactions(account_id)

        return {
            "portfolio": {
                "account_id": portfolio.account_id,
                "total_value": portfolio.total_value,
                "cash_balance": portfolio.cash_balance,
                "holdings_count": len(portfolio.holdings),
                "holdings": [
                    {
                        "symbol": h.symbol,
                        "quantity": h.quantity,
                        "market_value": h.market_value,
                        "current_price": h.current_price
                    }
                    for h in portfolio.holdings
                ],
                "inception_date": portfolio.inception_date.isoformat() if portfolio.inception_date else None
            },
            "transactions": {
                "count": len(db_transactions),
                "transactions": [
                    {
                        "date": t['date'],
                        "symbol": t['symbol'],
                        "type": t['transaction_type'],
                        "quantity": t['quantity'],
                        "price": t['price'],
                        "amount": t['amount']
                    }
                    for t in db_transactions[:20]  # First 20 only
                ]
            }
        }
    except Exception as e:
        logger.error(f"Debug error: {e}")
        return {"error": str(e)}


# Helper functions
def _calculate_portfolio_from_transactions(account_id: str, transactions_data: List[dict]) -> Portfolio:
    """Calculate current portfolio holdings from transaction history"""
    from collections import defaultdict

    # Build holdings from transactions
    holdings_dict = defaultdict(lambda: {'quantity': 0.0, 'cost_basis': 0.0})
    cash_balance = 0.0
    inception_date = None

    for txn in transactions_data:
        symbol = txn['symbol']
        txn_type = txn['transaction_type']
        quantity = txn['quantity']
        price = txn['price']
        fees = txn['fees']
        txn_date = datetime.fromisoformat(txn['date'])

        if inception_date is None or txn_date < inception_date:
            inception_date = txn_date

        if txn_type == 'buy':
            holdings_dict[symbol]['quantity'] += quantity
            holdings_dict[symbol]['cost_basis'] += (quantity * price + fees)
            cash_balance -= (quantity * price + fees)
        elif txn_type == 'sell':
            holdings_dict[symbol]['quantity'] -= quantity
            # Reduce cost basis proportionally
            if holdings_dict[symbol]['quantity'] > 0:
                cost_per_share = holdings_dict[symbol]['cost_basis'] / (holdings_dict[symbol]['quantity'] + quantity)
                holdings_dict[symbol]['cost_basis'] -= (quantity * cost_per_share)
            else:
                holdings_dict[symbol]['cost_basis'] = 0
            cash_balance += (quantity * price - fees)
        elif txn_type == 'dividend':
            cash_balance += txn['amount']
        elif txn_type == 'deposit':
            cash_balance += txn['amount']
        elif txn_type == 'withdrawal':
            cash_balance += txn['amount']  # amount is already negative

    # Get current prices for all symbols (exclude CASH placeholder)
    symbols = [sym for sym, data in holdings_dict.items() if data['quantity'] > 0 and sym != 'CASH']
    current_prices = market_data_service.get_current_prices(symbols) if symbols else {}

    # Create holdings list
    holdings = []
    total_value = 0.0

    for symbol, data in holdings_dict.items():
        if data['quantity'] > 0 and symbol != 'CASH':  # Only include positions we still hold (exclude CASH)
            current_price = current_prices.get(symbol)

            # If market price is unavailable or invalid, fall back to cost basis per share
            if not current_price or current_price <= 0:
                current_price = data['cost_basis'] / data['quantity'] if data['quantity'] > 0 else 0.0
                logger.warning(f"Market price unavailable for {symbol}, using cost basis per share: ${current_price:.2f}")

            market_value = data['quantity'] * current_price

            # Get stock info for additional data
            try:
                info = market_data_service.get_stock_info(symbol)
                sector = info.get('sector')
                industry = info.get('industry')
                country = info.get('country')
                market_cap = info.get('market_cap')
                pe_ratio = info.get('pe_ratio')
                pb_ratio = info.get('pb_ratio')
                beta = info.get('beta')
            except:
                sector = None
                industry = None
                country = None
                market_cap = None
                pe_ratio = None
                pb_ratio = None
                beta = None

            holding = Holding(
                symbol=symbol,
                quantity=data['quantity'],
                cost_basis=data['cost_basis'],
                current_price=current_price,
                market_value=market_value,
                sector=sector,
                industry=industry,
                country=country,
                asset_class='equity',
                market_cap=market_cap,
                pe_ratio=pe_ratio,
                pb_ratio=pb_ratio,
                beta=beta
            )
            holdings.append(holding)
            total_value += market_value

    # Get portfolio info from database
    portfolio_info = database.get_or_create_portfolio(account_id)

    return Portfolio(
        account_id=account_id,
        account_name=portfolio_info['account_name'],
        holdings=holdings,
        transactions=[],
        cash_balance=cash_balance,
        total_value=total_value + cash_balance,
        inception_date=inception_date or datetime.now(timezone.utc),
        last_updated=datetime.now(timezone.utc)
    )


def _get_mock_portfolio(account_id: str) -> Portfolio:
    """Generate mock portfolio for testing"""
    from models.portfolio import Holding

    # Create sample holdings
    holdings = [
        Holding(
            symbol="AAPL",
            quantity=100,
            cost_basis=15000,
            current_price=190.50,
            market_value=19050,
            sector="Technology",
            industry="Consumer Electronics",
            country="United States",
            market_cap=2.9e12,
            pe_ratio=31.5,
            pb_ratio=45.2,
            beta=1.25
        ),
        Holding(
            symbol="MSFT",
            quantity=50,
            cost_basis=18000,
            current_price=410.20,
            market_value=20510,
            sector="Technology",
            industry="Software",
            country="United States",
            market_cap=3.1e12,
            pe_ratio=35.8,
            pb_ratio=12.1,
            beta=0.92
        ),
        Holding(
            symbol="GOOGL",
            quantity=75,
            cost_basis=10000,
            current_price=142.50,
            market_value=10687.50,
            sector="Communication Services",
            industry="Internet Content",
            country="United States",
            market_cap=1.8e12,
            pe_ratio=27.3,
            pb_ratio=6.8,
            beta=1.05
        ),
    ]

    total_value = sum(h.market_value for h in holdings)

    return Portfolio(
        account_id=account_id,
        account_name="Mock Account",
        holdings=holdings,
        transactions=[],
        cash_balance=5000,
        total_value=total_value + 5000,
        inception_date=datetime.now(timezone.utc) - timedelta(days=365),
        last_updated=datetime.now(timezone.utc)
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")

    uvicorn.run(app, host=host, port=port)
