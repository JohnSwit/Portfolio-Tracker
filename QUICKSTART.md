# Quick Start Guide

Get up and running with Portfolio Analytics Tool in minutes!

## Quick Installation

### 1. Clone or download the repository

### 2. Quick start (Unix/Mac)
```bash
chmod +x start.sh
./start.sh
```

### 3. Manual start

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**Frontend (in a new terminal):**
```bash
cd frontend
npm install
npm run dev
```

### 4. Access the application

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## First Time Setup

### Without Schwab API (Mock Data)
The application works out of the box with realistic mock data. No configuration needed!

### With Schwab API (Real Data)

1. Get Schwab API credentials:
   - Visit https://developer.schwab.com
   - Register and create an app
   - Copy your API Key and Secret

2. Configure backend:
   ```bash
   cd backend
   cp .env.example .env
   # Edit .env and add your credentials
   ```

3. First authentication:
   - The backend will prompt you to authenticate via browser
   - Complete the OAuth flow
   - Your token will be saved for future use

## Features Overview

### Dashboard
- Portfolio overview
- Holdings breakdown
- Sector allocation
- Key metrics at a glance

### Performance
- Time-weighted and money-weighted returns
- Benchmark comparison (S&P 500)
- Sector attribution analysis
- Multiple time periods (1M, 3M, 6M, 1Y, YTD)

### Risk Analytics
- Volatility metrics
- Sharpe and Sortino ratios
- Value at Risk (VaR)
- Stress test scenarios
- Factor exposures
- Maximum drawdown analysis

### Holdings Analysis
- Concentration metrics
- Sector/Industry/Country exposure
- Liquidity analysis
- Factor characteristics
- Portfolio heatmaps

## Key Metrics Explained

**Time-Weighted Return (TWR)**: Measures performance independent of cash flows. Best for comparing portfolio managers.

**Money-Weighted Return (IRR)**: Accounts for timing of cash flows. Shows actual investor experience.

**Sharpe Ratio**: Risk-adjusted return. Higher is better. > 1.0 is good, > 2.0 is excellent.

**Beta**: Sensitivity to market. 1.0 = market average, > 1.0 = more volatile, < 1.0 = less volatile.

**VaR (95%)**: Maximum expected loss with 95% confidence. Example: 5% VaR means you could lose up to 5% on a bad day.

**Max Drawdown**: Largest peak-to-trough decline. Shows worst historical loss.

## Common Use Cases

### 1. Portfolio Check-up
1. Go to Dashboard
2. Check total value and overall gain/loss
3. Review top holdings and sector allocation

### 2. Performance Analysis
1. Go to Performance page
2. Select time period
3. Compare vs S&P 500 benchmark
4. Review sector attribution

### 3. Risk Assessment
1. Go to Risk page
2. Check Sharpe ratio (> 1.0 is good)
3. Review max drawdown
4. Run stress tests to see potential losses

### 4. Portfolio Rebalancing
1. Go to Holdings page
2. Check concentration metrics
3. Review sector/industry exposure
4. Look for over-concentration in top holdings

## Troubleshooting

**Backend won't start:**
- Check Python version (3.9+ required)
- Verify all dependencies installed
- Check port 8000 is available

**Frontend won't start:**
- Check Node version (18+ required)
- Run `npm install` again
- Check port 3000 is available

**No data showing:**
- Check backend is running (http://localhost:8000/health)
- Open browser console for errors
- Verify API connection in Network tab

**Schwab API errors:**
- Check credentials in .env file
- Verify token hasn't expired
- Re-authenticate if needed

## Next Steps

1. Explore all four main sections
2. Try different time periods in Performance
3. Run stress tests in Risk section
4. Customize benchmarks
5. Connect real Schwab account (optional)

## Getting Help

- Check full README.md for detailed documentation
- Review API docs at http://localhost:8000/docs
- Check browser console for frontend errors
- Review backend logs for API errors

## Tips

- Use mock data first to understand features
- Compare multiple time periods for better context
- Pay attention to risk-adjusted returns, not just raw returns
- Review stress tests regularly
- Monitor concentration to avoid over-exposure
- Check liquidity before making large changes

Enjoy analyzing your portfolio! 📊
