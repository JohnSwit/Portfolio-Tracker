# Implementation Examples

## How to Integrate Institutional Analytics into Your Application

### Quick Start (3 Steps)

#### 1. Apply Database Schema

```bash
# Run the schema migration
cd backend
sqlite3 ../portfolios.db < database_analytics_schema.sql
```

#### 2. Update Your Database Initialization

Add to `backend/database.py`:

```python
def init_analytics_tables():
    """Initialize analytics tables"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Read and execute the schema file
        with open('database_analytics_schema.sql', 'r') as f:
            schema_sql = f.read()
            cursor.executescript(schema_sql)

        conn.commit()
        logger.info("Analytics tables initialized")

# Call during startup
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing database...")
    database.init_database()
    database.migrate_database()
    database.init_analytics_tables()  # Add this line
```

#### 3. Start Using Analytics

```python
# Example: Add active share to portfolio endpoint
from services import analytics
from services.analytics_data_service import analytics_data_service

@app.get("/api/portfolio/{account_id}/analytics")
async def get_portfolio_analytics(
    account_id: str,
    period: str = "1Y"
):
    # Get current portfolio
    portfolio = await get_portfolio(account_id)

    # Compute active share
    portfolio_weights = analytics.compute_weights(
        portfolio.holdings,
        portfolio.total_value
    )

    # For benchmark, you can use equal-weight S&P 500 or fetch actual holdings
    # Simplified: use empty benchmark for now (shows 100% active)
    active_share = analytics.compute_active_share(
        portfolio_weights,
        {}  # Empty benchmark = all holdings are active
    )

    # Concentration metrics
    concentration = analytics.compute_concentration_metrics(
        portfolio.holdings,
        portfolio.total_value
    )

    return {
        "active_share": active_share,
        "concentration": concentration,
        "top5_weight": concentration["top5_weight"],
        "top10_weight": concentration["top10_weight"],
        "hhi": concentration["hhi_holdings"],
        "num_holdings": concentration["num_holdings"]
    }
```

---

## Full Integration Examples

### Example 1: Add Liquidity Analysis to Portfolio View

```python
from services import liquidity
from services.analytics_data_service import analytics_data_service

@app.get("/api/portfolio/{account_id}/liquidity")
async def get_portfolio_liquidity(account_id: str):
    """Get liquidity analysis for portfolio"""

    # Get portfolio
    portfolio = await get_portfolio(account_id)

    # Get volume data for all symbols
    symbols = [h.symbol for h in portfolio.holdings]
    volume_data = analytics_data_service.get_volume_data(symbols, lookback_days=20)

    # Compute liquidity profile
    liq_profile = liquidity.compute_portfolio_liquidity_profile(
        portfolio.holdings,
        volume_data,
        max_volume_participation=0.20
    )

    return {
        "weighted_avg_days_to_exit": liq_profile["weighted_avg_days_to_exit"],
        "liquidity_distribution": liq_profile["liquidity_distribution"],
        "most_illiquid_positions": liq_profile["most_illiquid_positions"][:5],
        "position_details": liq_profile["position_liquidity"]
    }
```

### Example 2: Add Tracking Error to Performance Endpoint

```python
from services import tracking_error
from services.analytics_data_service import analytics_data_service
from datetime import datetime, timedelta

@app.get("/api/portfolio/{account_id}/performance/advanced")
async def get_advanced_performance(
    account_id: str,
    period: str = "1Y",
    benchmark: str = "SPY"
):
    """Advanced performance with tracking error and IR"""

    # Date range
    end_date = datetime.now()
    period_days = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "3Y": 1095}
    start_date = end_date - timedelta(days=period_days.get(period, 365))

    # Get returns data
    portfolio_returns = analytics_data_service.compute_portfolio_returns(
        account_id, start_date, end_date
    )
    benchmark_returns = analytics_data_service.get_benchmark_returns(
        benchmark, start_date, end_date
    )

    # Compute active returns
    active_rets = tracking_error.compute_active_returns(
        portfolio_returns,
        benchmark_returns
    )

    # Tracking error and IR
    te = tracking_error.compute_tracking_error(active_rets)
    ir = tracking_error.compute_information_ratio(active_rets)

    # Portfolio and benchmark returns
    portfolio_total_return = (1 + portfolio_returns).prod() - 1
    benchmark_total_return = (1 + benchmark_returns).prod() - 1

    return {
        "period": period,
        "portfolio_return": portfolio_total_return,
        "benchmark_return": benchmark_total_return,
        "active_return": portfolio_total_return - benchmark_total_return,
        "tracking_error": te,
        "information_ratio": ir,
        "sharpe_ratio": None  # Can add if you have risk-free rate
    }
```

### Example 3: Add Factor Analysis

```python
from services import factor_analysis
from services.analytics_data_service import analytics_data_service

@app.get("/api/portfolio/{account_id}/factors")
async def get_factor_analysis(
    account_id: str,
    period: str = "1Y"
):
    """Factor exposure analysis"""

    # Date range
    end_date = datetime.now()
    period_days = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "3Y": 1095}
    start_date = end_date - timedelta(days=period_days.get(period, 365))

    # Get returns
    portfolio_returns = analytics_data_service.compute_portfolio_returns(
        account_id, start_date, end_date
    )
    factor_returns = analytics_data_service.get_factor_returns(
        start_date, end_date
    )

    # Estimate factor exposures
    exposures = factor_analysis.estimate_factor_exposures_regression(
        portfolio_returns,
        factor_returns
    )

    # Identify tilts
    tilts = factor_analysis.identify_factor_tilts(
        exposures["exposures"],
        significance_threshold=0.10
    )

    # Validate model
    model_quality = factor_analysis.validate_factor_model(
        exposures["r_squared"],
        exposures["residual_vol_annualized"]
    )

    return {
        "exposures": exposures["exposures"],
        "alpha_annualized": exposures["alpha_annualized"],
        "r_squared": exposures["r_squared"],
        "significant_tilts": tilts["significant_tilts"],
        "tilt_direction": tilts["tilt_direction"],
        "model_quality": model_quality["quality_score"]
    }
```

### Example 4: Add Drawdown Analysis

```python
from services import drawdown
from services.analytics_data_service import analytics_data_service

@app.get("/api/portfolio/{account_id}/risk/drawdown")
async def get_drawdown_analysis(
    account_id: str,
    period: str = "1Y",
    benchmark: str = "SPY"
):
    """Drawdown and capture ratio analysis"""

    # Date range
    end_date = datetime.now()
    period_days = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "3Y": 1095, "ITD": 3650}
    start_date = end_date - timedelta(days=period_days.get(period, 365))

    # Get returns
    portfolio_returns = analytics_data_service.compute_portfolio_returns(
        account_id, start_date, end_date
    )
    benchmark_returns = analytics_data_service.get_benchmark_returns(
        benchmark, start_date, end_date
    )

    # All drawdown metrics
    dd_metrics = drawdown.compute_all_drawdown_metrics(
        portfolio_returns,
        benchmark_returns
    )

    # Calmar ratio
    calmar = drawdown.compute_calmar_ratio(portfolio_returns)

    return {
        "max_drawdown": dd_metrics["max_drawdown"],
        "max_drawdown_start": dd_metrics["max_drawdown_start"],
        "max_drawdown_end": dd_metrics["max_drawdown_end"],
        "max_drawdown_duration_days": dd_metrics["max_drawdown_duration_days"],
        "recovery_date": dd_metrics["recovery_date"],
        "current_drawdown": dd_metrics["current_drawdown"],
        "upside_capture": dd_metrics.get("upside_capture"),
        "downside_capture": dd_metrics.get("downside_capture"),
        "calmar_ratio": calmar
    }
```

### Example 5: Full CIO Dashboard (Ready to Use)

```python
@app.get("/api/dashboard/cio/{account_id}/full")
async def get_full_cio_dashboard(
    account_id: str,
    period: str = Query(default="1Y"),
    benchmark: str = Query(default="SPY")
):
    """
    Full CIO dashboard with all analytics

    This endpoint aggregates all institutional analytics
    """
    from services.dashboard import build_cio_dashboard
    from services.analytics_data_service import analytics_data_service

    # Get portfolio
    portfolio = await get_portfolio(account_id)

    # Date range
    end_date = datetime.now()
    period_days = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "3Y": 1095, "ITD": 3650}
    start_date = end_date - timedelta(days=period_days.get(period, 365))

    # Get all required data
    portfolio_returns = analytics_data_service.compute_portfolio_returns(
        account_id, start_date, end_date
    )
    benchmark_returns = analytics_data_service.get_benchmark_returns(
        benchmark, start_date, end_date
    )
    factor_returns = analytics_data_service.get_factor_returns(
        start_date, end_date
    )

    symbols = [h.symbol for h in portfolio.holdings]
    volume_data = analytics_data_service.get_volume_data(symbols)

    # Build dashboard
    dashboard = build_cio_dashboard(
        portfolio,
        portfolio_returns,
        benchmark_returns,
        None,  # benchmark_holdings - can add later
        volume_data,
        factor_returns,
        period,
        benchmark
    )

    return dashboard
```

---

## Frontend Integration Examples

### React Component Example

```tsx
// components/PortfolioAnalytics.tsx
import React, { useEffect, useState } from 'react';

interface AnalyticsData {
  active_share: number;
  tracking_error: number;
  information_ratio: number;
  concentration: {
    top5_weight: number;
    top10_weight: number;
    hhi_holdings: number;
  };
}

export const PortfolioAnalytics: React.FC<{ accountId: string }> = ({ accountId }) => {
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`/api/portfolio/${accountId}/analytics`)
      .then(res => res.json())
      .then(data => {
        setAnalytics(data);
        setLoading(false);
      });
  }, [accountId]);

  if (loading) return <div>Loading analytics...</div>;
  if (!analytics) return <div>No analytics available</div>;

  return (
    <div className="analytics-dashboard">
      <h2>Portfolio Analytics</h2>

      <div className="metric-card">
        <h3>Active Share</h3>
        <p className="metric-value">{(analytics.active_share * 100).toFixed(1)}%</p>
        <p className="metric-description">
          {analytics.active_share > 0.6 ? 'Highly Active' :
           analytics.active_share > 0.2 ? 'Moderately Active' : 'Closet Indexer'}
        </p>
      </div>

      <div className="metric-card">
        <h3>Tracking Error</h3>
        <p className="metric-value">{(analytics.tracking_error * 100).toFixed(2)}%</p>
        <p className="metric-description">
          {analytics.tracking_error > 0.05 ? 'High Active Risk' : 'Low Active Risk'}
        </p>
      </div>

      <div className="metric-card">
        <h3>Information Ratio</h3>
        <p className="metric-value">{analytics.information_ratio?.toFixed(2) || 'N/A'}</p>
        <p className="metric-description">
          {analytics.information_ratio > 0.75 ? 'Excellent' :
           analytics.information_ratio > 0.5 ? 'Good' : 'Poor'}
        </p>
      </div>

      <div className="metric-card">
        <h3>Concentration</h3>
        <p className="metric-value">Top 5: {(analytics.concentration.top5_weight * 100).toFixed(1)}%</p>
        <p className="metric-value">Top 10: {(analytics.concentration.top10_weight * 100).toFixed(1)}%</p>
      </div>
    </div>
  );
};
```

### Dashboard Component with All Analytics

```tsx
// components/CIODashboard.tsx
import React, { useEffect, useState } from 'react';
import { LineChart, BarChart } from 'recharts'; // or your preferred chart library

export const CIODashboard: React.FC<{ accountId: string }> = ({ accountId }) => {
  const [dashboard, setDashboard] = useState(null);
  const [period, setPeriod] = useState('1Y');
  const [benchmark, setBenchmark] = useState('SPY');

  useEffect(() => {
    fetch(`/api/dashboard/cio/${accountId}/full?period=${period}&benchmark=${benchmark}`)
      .then(res => res.json())
      .then(data => setDashboard(data));
  }, [accountId, period, benchmark]);

  if (!dashboard) return <div>Loading dashboard...</div>;

  return (
    <div className="cio-dashboard">
      {/* Period Selector */}
      <div className="controls">
        <select value={period} onChange={(e) => setPeriod(e.target.value)}>
          <option value="1M">1 Month</option>
          <option value="3M">3 Months</option>
          <option value="6M">6 Months</option>
          <option value="1Y">1 Year</option>
          <option value="3Y">3 Years</option>
          <option value="ITD">Inception</option>
        </select>
      </div>

      {/* Key Metrics */}
      <div className="key-metrics-grid">
        <MetricCard
          title="Active Share"
          value={`${(dashboard.active_share * 100).toFixed(1)}%`}
          change={dashboard.active_return}
        />
        <MetricCard
          title="Tracking Error"
          value={`${(dashboard.tracking_error * 100).toFixed(2)}%`}
        />
        <MetricCard
          title="Information Ratio"
          value={dashboard.information_ratio?.toFixed(2) || 'N/A'}
        />
      </div>

      {/* Performance Attribution */}
      <div className="attribution-section">
        <h3>Performance Attribution</h3>
        <BarChart data={[
          { name: 'Allocation', value: dashboard.attribution.allocation_effect },
          { name: 'Selection', value: dashboard.attribution.selection_effect },
          { name: 'Interaction', value: dashboard.attribution.interaction_effect }
        ]} />
      </div>

      {/* Top Contributors/Detractors */}
      <div className="contributors-grid">
        <div>
          <h4>Top Contributors</h4>
          <ul>
            {dashboard.attribution.top_contributors.slice(0, 5).map(c => (
              <li key={c.symbol}>
                {c.symbol}: +{(c.contribution * 100).toFixed(2)}%
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h4>Top Detractors</h4>
          <ul>
            {dashboard.attribution.top_detractors.slice(0, 5).map(c => (
              <li key={c.symbol}>
                {c.symbol}: {(c.contribution * 100).toFixed(2)}%
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Factor Exposures */}
      <div className="factor-exposures">
        <h3>Factor Tilts</h3>
        <BarChart data={[
          { name: 'Market', value: dashboard.factor_exposures.market_beta },
          { name: 'Value', value: dashboard.factor_exposures.value_factor },
          { name: 'Growth', value: dashboard.factor_exposures.growth_factor },
          { name: 'Momentum', value: dashboard.factor_exposures.momentum_factor },
        ]} />
      </div>

      {/* Liquidity Profile */}
      <div className="liquidity-section">
        <h3>Liquidity Analysis</h3>
        <p>Avg Days to Exit: {dashboard.liquidity.weighted_avg_days_to_exit.toFixed(1)}</p>
        <p>Highly Liquid: {dashboard.liquidity.pct_highly_liquid.toFixed(1)}%</p>
      </div>
    </div>
  );
};
```

---

## Testing Your Implementation

### 1. Test Basic Analytics

```bash
# After applying schema and adding endpoints, test:
curl http://localhost:8000/api/portfolio/ACCOUNT_ID/analytics

# Should return:
# {
#   "active_share": 0.42,
#   "concentration": {...},
#   "top5_weight": 0.35
# }
```

### 2. Test CIO Dashboard

```bash
curl "http://localhost:8000/api/dashboard/cio/ACCOUNT_ID/full?period=1Y&benchmark=SPY"
```

### 3. Run Unit Tests

```bash
# All analytics tests should pass
pytest tests/test_analytics.py tests/test_tracking_error.py tests/test_attribution.py -v
```

---

## Deployment Checklist

- [ ] Apply database schema (`database_analytics_schema.sql`)
- [ ] Add `analytics_data_service.py` to services
- [ ] Update `main.py` with new endpoints (start with simple ones)
- [ ] Test with your existing portfolios
- [ ] Add caching for expensive calculations
- [ ] Set up scheduled jobs to compute daily returns
- [ ] Add error handling and logging
- [ ] Create frontend components
- [ ] Test with real data
- [ ] Deploy!

---

## Performance Tips

1. **Cache Aggressively**: Store computed metrics in `analytics_cache` table
2. **Batch Computations**: Compute all analytics in one pass
3. **Use Background Jobs**: Compute returns daily via cron job
4. **Limit History**: Default to 1Y, only compute longer periods on demand
5. **Pagination**: For position-level data, paginate results

---

## Next Steps

1. **Start Simple**: Begin with active share and concentration (no historical data needed)
2. **Add Returns Tracking**: Set up daily job to compute and store portfolio returns
3. **Integrate Benchmark Data**: Fetch SPY returns daily
4. **Add More Analytics**: Gradually add tracking error, drawdown, etc.
5. **Build Dashboard**: Create comprehensive frontend view

The analytics modules are ready to use - you just need to wire them into your existing application!
