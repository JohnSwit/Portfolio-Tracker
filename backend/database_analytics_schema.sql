-- Database schema for institutional analytics
-- Add these tables to your existing database

-- Historical portfolio returns (daily)
CREATE TABLE IF NOT EXISTS portfolio_returns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id TEXT NOT NULL,
    date DATE NOT NULL,
    daily_return REAL NOT NULL,
    portfolio_value REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(account_id, date)
);

-- Historical position-level returns
CREATE TABLE IF NOT EXISTS position_returns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    daily_return REAL NOT NULL,
    position_value REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(account_id, symbol, date)
);

-- Benchmark returns cache
CREATE TABLE IF NOT EXISTS benchmark_returns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    benchmark_ticker TEXT NOT NULL,
    date DATE NOT NULL,
    daily_return REAL NOT NULL,
    close_price REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(benchmark_ticker, date)
);

-- Factor ETF returns cache
CREATE TABLE IF NOT EXISTS factor_returns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_name TEXT NOT NULL,  -- Market, Value, Growth, etc.
    etf_ticker TEXT NOT NULL,   -- SPY, IWD, IWF, etc.
    date DATE NOT NULL,
    daily_return REAL NOT NULL,
    close_price REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(factor_name, date)
);

-- Volume data cache (for liquidity analysis)
CREATE TABLE IF NOT EXISTS volume_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    volume INTEGER NOT NULL,
    avg_volume_20d REAL,  -- 20-day rolling average
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date)
);

-- Portfolio snapshots (for tracking over time)
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id TEXT NOT NULL,
    snapshot_date DATE NOT NULL,
    total_value REAL NOT NULL,
    cash_balance REAL NOT NULL,
    holdings_json TEXT NOT NULL,  -- JSON array of holdings
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(account_id, snapshot_date)
);

-- Analytics cache (store computed metrics)
CREATE TABLE IF NOT EXISTS analytics_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id TEXT NOT NULL,
    metric_type TEXT NOT NULL,  -- 'active_share', 'tracking_error', etc.
    period TEXT NOT NULL,        -- '1M', '3M', '1Y', etc.
    metric_value REAL,
    metric_json TEXT,            -- For complex metrics
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    UNIQUE(account_id, metric_type, period)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_portfolio_returns_account_date
    ON portfolio_returns(account_id, date DESC);

CREATE INDEX IF NOT EXISTS idx_position_returns_account_symbol_date
    ON position_returns(account_id, symbol, date DESC);

CREATE INDEX IF NOT EXISTS idx_benchmark_returns_ticker_date
    ON benchmark_returns(benchmark_ticker, date DESC);

CREATE INDEX IF NOT EXISTS idx_factor_returns_factor_date
    ON factor_returns(factor_name, date DESC);

CREATE INDEX IF NOT EXISTS idx_volume_data_symbol_date
    ON volume_data(symbol, date DESC);

CREATE INDEX IF NOT EXISTS idx_analytics_cache_lookup
    ON analytics_cache(account_id, metric_type, period);
