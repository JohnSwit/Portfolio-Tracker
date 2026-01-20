# Portfolio Analytics Tool

A comprehensive portfolio management and analytics platform with real-time data integration from Schwab API and financial databases. Built with Python (FastAPI) backend and React (TypeScript) frontend.

## Features

### Performance Measurement
- **Time-Weighted Return (TWR)**: Eliminates the effect of cash flows to measure manager performance
- **Money-Weighted Return (IRR)**: Internal rate of return accounting for cash flow timing
- **Multiple Period Analysis**: 1M, 3M, 6M, 1Y, YTD comparisons
- **Benchmark Comparison**: Compare against major indices (S&P 500, Russell 2000, NASDAQ, custom benchmarks)
- **Attribution Analysis**:
  - Sector-level attribution
  - Country attribution
  - Stock-level contribution
  - Factor-based attribution

### Risk Analytics
- **Volatility Measures**:
  - Daily and annual volatility
  - Downside volatility (semi-deviation)

- **Risk-Adjusted Returns**:
  - Sharpe Ratio
  - Sortino Ratio
  - Calmar Ratio
  - Information Ratio

- **Market Risk Metrics**:
  - Beta (market sensitivity)
  - Alpha (excess returns)
  - Correlation with benchmark
  - Tracking error

- **Drawdown Analysis**:
  - Maximum drawdown
  - Current drawdown
  - Drawdown duration

- **Value at Risk (VaR)**:
  - 95% and 99% VaR
  - Conditional VaR (Expected Shortfall)
  - Historical simulation method

- **Stress Testing**:
  - 2008 Financial Crisis scenario
  - COVID-19 Crash scenario
  - Tech Bubble Burst scenario
  - Rising Interest Rates scenario
  - Custom scenario support

- **Factor Exposures**:
  - Market factor
  - Size (small vs large cap)
  - Value vs Growth
  - Momentum
  - Quality

### Holdings Analysis
- **Concentration Metrics**:
  - Top 10 and Top 20 concentration
  - Herfindahl Index
  - Active Share vs benchmark

- **Exposure Breakdown**:
  - Sector exposure
  - Country/geographic exposure
  - Industry exposure
  - Asset class allocation

- **Liquidity Analysis**:
  - Average daily trading volume
  - Weighted liquidity score
  - Illiquid holdings percentage
  - Days to liquidate calculation

- **Factor Characteristics**:
  - Average and median market cap
  - P/E and P/B ratios
  - Weighted portfolio beta
  - Value, Growth, Momentum, Quality scores

### Data Integration
- **Schwab API Integration**: Real-time portfolio data, holdings, and transactions
- **Market Data**: Live prices and historical data via yfinance
- **Automatic Enrichment**: Stocks automatically enriched with sector, industry, fundamentals

## Tech Stack

### Backend
- **Framework**: FastAPI (Python)
- **Financial Libraries**: pandas, numpy, scipy
- **Market Data**: yfinance, schwab-py
- **Analytics**: empyrical, pyfolio
- **API**: RESTful endpoints with automatic OpenAPI documentation

### Frontend
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **Routing**: React Router
- **HTTP Client**: Axios

## Installation

### Prerequisites
- Python 3.9+
- Node.js 18+
- npm or yarn

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
```

Edit `.env` with your configuration:
```env
# Schwab API Configuration (optional - uses mock data if not configured)
SCHWAB_API_KEY=your_schwab_api_key
SCHWAB_SECRET=your_schwab_secret
SCHWAB_CALLBACK_URL=https://localhost:8080/callback

# Market Data API (optional)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# Database
DATABASE_URL=sqlite:///./portfolio.db

# Application
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000
```

5. Run the backend:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

API documentation at `http://localhost:8000/docs`

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create environment file (optional):
```bash
# Create .env file
echo "VITE_API_URL=http://localhost:8000" > .env
```

4. Run the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:3000`

## Usage

### Getting Started

1. **Start both backend and frontend servers**
2. **Access the dashboard** at `http://localhost:3000`
3. **Connect Schwab account** (optional) or use mock data
4. **Explore analytics**:
   - Dashboard: Portfolio overview and key metrics
   - Performance: Returns, attribution, and benchmark comparison
   - Risk: Volatility, VaR, stress tests, and factor exposures
   - Holdings: Concentration, exposures, and liquidity

### Schwab API Integration

To connect your Schwab account:

1. **Register for Schwab Developer Account**:
   - Visit https://developer.schwab.com
   - Create an application
   - Get your API key and secret

2. **Configure credentials** in `backend/.env`

3. **Authenticate**:
   - The first time you run the backend, it will prompt for authentication
   - Follow the OAuth flow in your browser
   - Token will be saved for future use

4. **Access your portfolio**:
   - The API will automatically fetch your holdings and transactions
   - Data refreshes based on cache settings

### Using Mock Data

If Schwab API is not configured, the application uses realistic mock data:
- 3 sample holdings (AAPL, MSFT, GOOGL)
- Realistic prices and fundamentals
- Sample transactions

This is perfect for testing and development.

## API Endpoints

### Portfolio
- `GET /api/portfolio/{account_id}` - Get portfolio holdings
- `GET /api/transactions/{account_id}` - Get transaction history

### Performance
- `POST /api/performance/{account_id}` - Calculate performance metrics
- `POST /api/attribution/{account_id}` - Calculate return attribution

### Risk
- `POST /api/risk/{account_id}` - Calculate risk metrics
- `POST /api/stress-test/{account_id}` - Run predefined stress tests
- `POST /api/stress-test/{account_id}/custom` - Run custom scenario

### Holdings
- `GET /api/holdings/{account_id}` - Analyze holdings

### Market Data
- `GET /api/market/prices` - Get current prices
- `GET /api/market/info/{symbol}` - Get stock information

## Project Structure

```
Portfolio-Tracker/
├── backend/
│   ├── models/              # Pydantic data models
│   │   └── portfolio.py     # Portfolio, holdings, metrics models
│   ├── services/            # Business logic
│   │   ├── market_data.py   # Market data fetching
│   │   ├── schwab_client.py # Schwab API integration
│   │   ├── performance.py   # Performance calculations
│   │   ├── risk.py          # Risk analytics
│   │   └── holdings.py      # Holdings analysis
│   ├── main.py              # FastAPI application
│   └── requirements.txt     # Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── components/      # Reusable React components
│   │   │   ├── Layout.tsx
│   │   │   ├── Card.tsx
│   │   │   └── MetricCard.tsx
│   │   ├── pages/           # Page components
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Performance.tsx
│   │   │   ├── Risk.tsx
│   │   │   └── Holdings.tsx
│   │   ├── services/        # API clients
│   │   │   └── api.ts
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts
│
└── README.md
```

## Financial Calculations

### Time-Weighted Return (TWR)
Eliminates cash flow impact by linking sub-period returns:
```
TWR = [(1 + R₁) × (1 + R₂) × ... × (1 + Rₙ)] - 1
```

### Money-Weighted Return (IRR)
Solves for rate that makes NPV = 0:
```
NPV = Σ(CF_t / (1 + IRR)^t) = 0
```

### Sharpe Ratio
Risk-adjusted return:
```
Sharpe = (Rₚ - Rբ) / σₚ
```

### Value at Risk (VaR)
Maximum expected loss at confidence level:
```
VaR₉₅ = -Percentile(returns, 5%)
```

### Beta
Sensitivity to market movements:
```
β = Cov(Rₚ, Rₘ) / Var(Rₘ)
```

## Development

### Running Tests
```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

### Building for Production

**Backend:**
```bash
# The backend runs as-is with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm run build
# Outputs to frontend/dist
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Market data powered by yfinance
- Schwab API integration via schwab-py
- Financial calculations using empyrical and scipy
- UI components built with Recharts

## Support

For issues, questions, or contributions:
- GitHub Issues: [your-repo-url]/issues
- Documentation: [your-docs-url]

## Roadmap

- [ ] Add more benchmark options (international indices)
- [ ] Implement Monte Carlo simulation
- [ ] Add tax lot tracking
- [ ] Multi-currency support
- [ ] Portfolio optimization (efficient frontier)
- [ ] Real-time WebSocket data updates
- [ ] Mobile app
- [ ] PDF report generation
- [ ] Email alerts for risk thresholds

## Security Notes

- Never commit API keys or secrets
- Use environment variables for sensitive data
- Schwab tokens are stored locally (add to .gitignore)
- HTTPS recommended for production deployments
- Consider using a secrets manager for production
