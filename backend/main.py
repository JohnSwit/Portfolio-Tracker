from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import logging
import os
from dotenv import load_dotenv

from models.portfolio import (
    Portfolio, PerformanceMetrics, AttributionResult,
    RiskMetrics, HoldingsAnalysis, StressTestScenario, StressTestResult
)
from services.schwab_client import schwab_client, SCHWAB_AVAILABLE
from services.performance import performance_service
from services.risk import risk_service, STRESS_SCENARIOS
from services.holdings import holdings_service
from services.market_data import market_data_service

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
    if not SCHWAB_AVAILABLE:
        # Return mock data for development/testing
        return _get_mock_portfolio(account_id)

    try:
        portfolio = schwab_client.get_portfolio(account_id)
        return portfolio
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

        if SCHWAB_AVAILABLE:
            transactions = schwab_client.get_transactions(account_id, start_date, end_date)
        else:
            transactions = []  # Mock empty transactions

        return {"transactions": transactions}
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


# Helper functions
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
