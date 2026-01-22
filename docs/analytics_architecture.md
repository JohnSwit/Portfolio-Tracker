# Institutional Analytics Architecture

## Current State Assessment

### Existing Infrastructure

**Data Models** (`backend/models/portfolio.py`):
- ✅ `Holding`: symbol, quantity, cost_basis, market_value, sector, industry, country
- ✅ `Transaction`: complete transaction history with is_synthetic flag
- ✅ `Portfolio`: holdings, transactions, cash_balance, total_value
- ✅ `PerformanceMetrics`: includes benchmark_return, active_return, tracking_error, information_ratio (stubs)
- ✅ `AttributionResult`: sector/country/stock/factor attribution (stubs)
- ✅ `RiskMetrics`: volatility, drawdowns, VaR, beta, alpha, correlation

**Services**:
- ✅ `market_data.py`: yfinance integration for prices and benchmark data
- ✅ `performance_v2.py`: TWR, MWR, Simple Return calculations (recently fixed)
- ✅ `risk.py`: Risk metrics including benchmark comparison
- ✅ Holdings already have sector/industry/country data

**API Endpoints**:
- ✅ `/api/performance/{account_id}`: Performance metrics
- ✅ `/api/attribution/{account_id}`: Attribution (stub)
- ✅ `/api/risk/{account_id}`: Risk analytics
- ✅ `/api/holdings/{account_id}`: Holdings analysis

### Gaps to Fill

1. **Benchmark Support**: Needs explicit benchmark configuration and composition loading
2. **Active Share**: Not implemented
3. **Tracking Error**: Stub exists, needs proper calculation
4. **Factor Analysis**: Not implemented
5. **Brinson Attribution**: Stub exists, needs full implementation
6. **Concentration Metrics**: Partial (HHI mentioned in models)
7. **Liquidity Metrics**: Model fields exist (avg_daily_volume), calculation missing
8. **CIO Dashboard**: No dedicated UI endpoint

## Implementation Plan

### Phase 1: Core Analytics Engine

**New Module**: `backend/services/analytics.py`

Core calculations:
- `compute_weights(holdings, benchmark_holdings) -> Dict`: Portfolio and benchmark weights
- `compute_active_share(port_weights, bench_weights) -> float`: Active share metric
- `compute_tracking_error(port_returns, bench_returns) -> Dict`: Realized TE + components
- `compute_concentration_metrics(holdings) -> Dict`: HHI, top N concentration
- `compute_drawdowns(returns) -> Dict`: Max drawdown, current drawdown, duration
- `compute_capture_ratios(port_returns, bench_returns) -> Dict`: Up/down capture

**New Module**: `backend/services/attribution.py`

- `brinson_attribution(port_holdings, bench_holdings, returns) -> Dict`: Full Brinson model
- `compute_contribution_to_return(holdings, returns) -> List`: Position-level attribution

**Enhanced Module**: `backend/services/benchmark.py` (new)

- `load_benchmark(ticker) -> Dict`: Load benchmark composition from source
- `get_benchmark_holdings(ticker, date) -> List[Holding]`: Benchmark constituents
- `get_benchmark_returns(ticker, start, end) -> Series`: Benchmark return series

### Phase 2: Factor Analysis

**New Module**: `backend/services/factors.py`

Pragmatic approach (Option C - ETF regression):
- Use factor ETFs: SPY (market), IWM (size), IWD (value), IWF (growth), MTUM (momentum), QUAL (quality), USMV (low vol)
- Regress portfolio and benchmark returns against factor returns
- Compute active factor tilts = beta_portfolio - beta_benchmark

Functions:
- `compute_factor_exposures(returns, factor_returns) -> Dict`: Regression-based factor betas
- `compute_active_factor_tilts(port_factors, bench_factors) -> Dict`: Active tilts

### Phase 3: Liquidity & Implementation

**Enhanced**: `backend/services/market_data.py`

Add methods:
- `get_volume_data(symbols, start, end) -> DataFrame`: Historical volume data
- `compute_adv(volume_data) -> Dict`: Average daily volume (20-day and 60-day)

**New Module**: `backend/services/liquidity.py`

- `compute_liquidity_metrics(holdings, volume_data, config) -> Dict`: Days-to-exit analysis
- `flag_liquidity_warnings(holdings, volume_data, thresholds) -> List`: Illiquid positions

### Phase 4: CIO Dashboard

**New Endpoint**: `/api/cio-dashboard/{account_id}`

Response model: `CIODashboard`

Includes:
- Active share
- Tracking error (realized)
- Top 5 contributors/detractors to return
- Top 5 contributors to active risk
- Factor tilts (active) summary
- Sector active weights table
- Liquidity warning flags
- Current relative drawdown
- Period selector (1M, 3M, 6M, YTD, 1Y, 3Y)

## Data Models

### New Models (add to `models/portfolio.py`):

```python
class BenchmarkConfig(BaseModel):
    """Benchmark configuration"""
    ticker: str
    name: str
    composition_source: str = "index"  # "index", "etf", "manual"
    rebalance_frequency: str = "quarterly"

class ActiveMetrics(BaseModel):
    """Active management metrics"""
    active_share: float
    tracking_error: float
    information_ratio: float
    active_risk_contributors: List[Dict[str, float]]

class ConcentrationMetrics(BaseModel):
    """Portfolio concentration measures"""
    top5_weight: float
    top10_weight: float
    hhi_holdings: float
    hhi_sectors: float
    avg_correlation_60d: Optional[float] = None
    avg_correlation_252d: Optional[float] = None

class LiquidityMetrics(BaseModel):
    """Liquidity and implementation metrics"""
    days_to_exit_stats: Dict[str, float]  # percentiles
    pct_illiquid_3d: float  # % in positions > 3 days to exit
    pct_illiquid_5d: float
    pct_illiquid_10d: float
    warnings: List[Dict[str, any]]

class FactorExposures(BaseModel):
    """Factor exposures"""
    market_beta: float
    size_tilt: float  # SMB
    value_tilt: float  # HML
    momentum_tilt: float
    quality_tilt: float
    low_vol_tilt: float
    active_tilts: Dict[str, float]  # vs benchmark

class CIODashboard(BaseModel):
    """CIO summary dashboard"""
    period: str
    as_of_date: datetime

    # Key metrics
    active_share: float
    tracking_error: float
    information_ratio: float

    # Attribution
    top_contributors: List[Dict[str, float]]  # symbol, contribution
    top_detractors: List[Dict[str, float]]
    attribution_summary: Dict[str, float]  # allocation, selection, interaction

    # Risk
    active_risk_contributors: List[Dict[str, float]]
    relative_drawdown: float
    downside_capture: float
    upside_capture: float

    # Positioning
    sector_active_weights: Dict[str, float]
    factor_tilts: Dict[str, float]
    concentration: ConcentrationMetrics
    liquidity: LiquidityMetrics
```

## Configuration

Add to `backend/config.py` (or `.env`):

```python
# Analytics configuration
BENCHMARK_TICKER = "SPY"
ANNUALIZATION_FACTOR = 252
DAYS_TO_EXIT_FRACTION = 0.20  # Use 20% of ADV
LIQUIDITY_THRESHOLDS = {"warning": 3, "alert": 5, "critical": 10}  # days

# Factor ETFs for regression
FACTOR_ETFS = {
    "market": "SPY",
    "size": "IWM",
    "value": "IWD",
    "growth": "IWF",
    "momentum": "MTUM",
    "quality": "QUAL",
    "low_vol": "USMV"
}
```

## Testing Strategy

### Unit Tests (`tests/test_analytics.py`)

Test each calculation function with known inputs/outputs:
- Active share: manual portfolio vs benchmark → verify formula
- Tracking error: synthetic return series → verify annualization
- Brinson attribution: toy example → verify sum = active return
- Drawdown: crafted return series with known max → verify detection
- Days to exit: holdings + volume → verify calculation

### Golden Fixtures (`tests/fixtures/`)

- `sample_portfolio.json`: 10 holdings with sectors
- `sample_benchmark.json`: Matching benchmark composition
- `sample_returns.csv`: Daily return series (100 days)
- `sample_volumes.csv`: Volume data for holdings

### Integration Tests

- Full CIO dashboard API call with sample data
- Verify no crashes on missing data (handle gracefully)
- Performance: 200 holdings, 1260 days (~5 years) should complete in <30s

## Dependencies

Existing (already in requirements.txt):
- pandas, numpy, scipy: ✅ Core analytics
- yfinance: ✅ Market data

Potentially add (lightweight):
- `scikit-learn` (optional, for factor regression if needed beyond scipy)
  - Alternative: Use scipy.stats.linregress (already available)

**Decision**: Stick with existing dependencies. Use scipy for regressions.

## Performance Considerations

- Cache benchmark compositions (refresh quarterly)
- Cache factor ETF return series (refresh daily)
- Pre-compute daily portfolio returns and store in database for dashboards
- Use vectorized pandas operations throughout
- Limit rolling correlation windows to 252 days max

## Error Handling & Data Quality

- **Missing benchmark data**: Fallback to cash (0% return) and warn
- **Missing sector/industry**: Group as "Unknown" and warn in dashboard
- **Missing volume data**: Set days_to_exit = NaN and flag
- **Corporate actions**: Assume adjusted prices (document in analytics.md)
- **Stale prices**: Warn if last updated > 1 day

## UI Considerations

For the CIO Dashboard:
- **Framework**: Assume existing frontend (React or similar)
- **New Route**: `/dashboard/cio` or similar
- **Components**:
  - Metric cards (active share, TE, IR)
  - Sector weights heatmap (active vs benchmark)
  - Attribution waterfall chart
  - Top contributors/detractors table
  - Factor tilt radar/bar chart
  - Liquidity gauge/buckets
  - Relative drawdown chart

**Style**: Clean, professional, "Bloomberg-lite" aesthetic
- Dark mode support
- Exportable to PDF/CSV
- Period selector dropdown

## Rollout Plan

### Commit 1: Foundation
- Create `analytics.py` with core weight/active share functions
- Add unit tests
- Update models with new classes

### Commit 2: Tracking Error & Risk
- Implement tracking error calculation
- Add realized TE to risk service
- Tests for TE

### Commit 3: Attribution
- Full Brinson attribution
- Contribution to return by position
- Tests with toy example

### Commit 4: Concentration & Drawdown
- Concentration metrics (HHI, top N)
- Drawdown analysis (max, relative)
- Capture ratios
- Tests

### Commit 5: Liquidity
- Volume data ingestion
- Days-to-exit calculation
- Liquidity warnings
- Tests

### Commit 6: Factor Analysis
- ETF regression approach
- Active factor tilts
- Tests with synthetic data

### Commit 7: CIO Dashboard
- New API endpoint
- CIODashboard model
- Integration of all analytics
- Tests

### Commit 8: Documentation
- Update README with usage guide
- Add formula definitions to docs/
- Example API calls

## Timeline Estimate

- Foundation: 1-2 hours
- Analytics core: 2-3 hours
- Attribution + Tests: 1-2 hours
- Liquidity + Factors: 1-2 hours
- Dashboard endpoint: 1 hour
- Testing + Polish: 1-2 hours
- Documentation: 1 hour

**Total: 8-13 hours of implementation**

## Success Criteria

✅ All unit tests pass
✅ CIO dashboard returns complete data for sample portfolio
✅ No crashes on missing data (graceful degradation)
✅ Performance: <5s for dashboard with 100 holdings, 1Y of data
✅ Documentation clear enough for institutional user to understand metrics
✅ Code review ready (small commits, clear)
