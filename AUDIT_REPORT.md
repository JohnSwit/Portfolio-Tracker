# Portfolio Tracker Performance Calculation Audit Report

## Executive Summary
**CRITICAL BUG IDENTIFIED**: Synthetic cash transactions are polluting return calculations.

## STEP 1: REPO AUDIT FINDINGS

### A) Transaction Schema & Sign Conventions

**Location**: `backend/models/portfolio.py`

**Schema**:
```python
class Transaction(BaseModel):
    date: datetime
    symbol: str
    transaction_type: str  # buy, sell, dividend, split, deposit, withdrawal
    quantity: float
    price: float
    amount: float  # SIGNED: negative for outflows, positive for inflows
    fees: float = 0.0
    notes: Optional[str] = None
```

**Sign Convention** (found in `backend/main.py` lines 588-600):
- **BUY**: `amount = -(quantity * price + fees)` → NEGATIVE (cash out)
- **SELL**: `amount = quantity * price - fees` → POSITIVE (cash in)
- **DIVIDEND**: `amount = quantity * price` → POSITIVE (cash in)
- **DEPOSIT**: `amount = quantity * price` → POSITIVE (cash in)
- **WITHDRAWAL**: `amount = -(quantity * price)` → NEGATIVE (cash out)

✅ Sign convention is correct and consistent.

### B) Price Data Storage & Frequency

**Location**: `backend/services/market_data.py`

**Source**: yfinance library
**Frequency**: Daily (interval="1d")
**Storage**: Fetched on-demand, no persistent storage
**Adjustment**: Uses adjusted close prices (includes splits/dividends)

**Current Implementation**:
- `get_current_prices()`: Fetches latest prices
- `get_price_data()`: Fetches historical daily OHLCV data
- Forward-fill and backward-fill for missing dates (weekends/holidays)

⚠️ Issue: No caching, fetches on every calculation (performance problem)

### C) Holdings/Valuation Computation

**Location**: `backend/main.py` lines 760-830 (`_calculate_portfolio_from_transactions`)

**Process**:
1. Reconstruct holdings from transaction history
2. Track quantities using buy/sell transactions
3. Calculate cost basis (weighted average)
4. Fetch current market prices
5. Calculate market values: `quantity * current_price`
6. Sum all holdings + cash_balance = total_value

**Cash Balance Calculation** (lines 778-799):
```python
cash_balance = 0.0
for txn in transactions:
    if txn_type == 'buy':
        cash_balance += (-(quantity * price + fees))
    elif txn_type == 'sell':
        cash_balance += (quantity * price - fees)
    elif txn_type == 'dividend':
        cash_balance += txn['amount']
    elif txn_type == 'deposit':
        cash_balance += txn['amount']
    elif txn_type == 'withdrawal':
        cash_balance += txn['amount']
```

✅ Holdings computation looks correct.
⚠️ Cash balance includes synthetic transactions (see section D).

### D) **AUTO CASH DEPOSIT/WITHDRAW LOGIC** ⚠️ **CRITICAL BUG SOURCE**

**Location**: `backend/main.py` lines 458-499 (CSV upload) and 622-663 (manual add)

**Logic**:
When a user adds a transaction (buy/sell/dividend), the system AUTOMATICALLY creates a matching CASH transaction to ensure cash_balance stays at zero.

**For BUY** (lines 459-471):
```python
# User transaction: BUY 100 AAPL @ $150
# System auto-creates:
{
    'symbol': 'CASH',
    'transaction_type': 'deposit',
    'amount': (quantity * price + fees),  # POSITIVE
    'notes': 'Auto-deposit to fund AAPL purchase'
}
```

**For SELL** (lines 473-485):
```python
# User transaction: SELL 50 AAPL @ $170
# System auto-creates:
{
    'symbol': 'CASH',
    'transaction_type': 'withdrawal',
    'amount': -(quantity * price - fees),  # NEGATIVE
    'notes': 'Auto-withdrawal of AAPL sale proceeds'
}
```

**For DIVIDEND** (lines 487-499):
```python
# User transaction: DIVIDEND AAPL $1.24
# System auto-creates:
{
    'symbol': 'CASH',
    'transaction_type': 'withdrawal',
    'amount': -(quantity * price),  # NEGATIVE
    'notes': 'Auto-withdrawal of AAPL dividend'
}
```

**Storage**: Synthetic CASH transactions are stored in the same `transactions` table with NO FLAG to distinguish them from real deposits/withdrawals.

**Detection Method**: Currently identified by:
- `symbol == 'CASH'`
- `notes` starts with "Auto-"

⚠️ **PROBLEM**: This is FRAGILE and INCORRECT because:
1. Users might want to legitimately track cash as a position
2. Notes field might be edited
3. No schema-level distinction between real and synthetic flows

### E) Current TWR/MWR Implementation

**Location**: `backend/services/performance_v2.py`

**TWR Implementation** (lines 405-461):
- Uses chain-linking of sub-period returns
- **Attempts** to filter CASH transactions: `if t.symbol != 'CASH'`
- **Line 441**: `flows_on_date = sum([t.amount for t in transactions if t.symbol != 'CASH' and ...])`

**MWR Implementation** (lines 523-569):
- Uses IRR/XIRR calculation
- **Attempts** to filter CASH transactions: `if t.symbol != 'CASH'`
- Solves NPV equation using Brent's method

**Current Filtering**:
```python
# In _get_portfolio_cash_flows (lines 641-677):
for txn in transactions:
    if txn.symbol == 'CASH':
        cash_txns_excluded += 1
        continue
    if start_date < txn_date <= end_date:
        flows.append((txn_date, txn.amount))
```

**PROBLEMS**:
1. ✅ Sign error FIXED (we corrected lines 447 and 507)
2. ❌ **Filtering by `symbol == 'CASH'` excludes ALL cash transactions, including legitimate deposits/withdrawals**
3. ❌ **If user has real deposits/withdrawals (not auto-generated), they are lost**
4. ❌ **Semantic confusion: CASH transactions are treated as "internal" when they should be "external flows"**

## ROOT CAUSE ANALYSIS

### The Bug Chain:

1. **Design Decision**: System auto-creates offsetting CASH transactions to keep cash_balance = 0
2. **Implementation**: These synthetic transactions are stored identically to real transactions
3. **Attempted Fix**: Code tries to exclude them by filtering `symbol == 'CASH'`
4. **Failure Mode**: This filter is too broad and conceptually wrong

### Impact on Returns:

**Scenario 1: User has no real deposits/withdrawals (only buys/sells)**
- Current code: Works somewhat (but confusing)
- Auto CASH transactions cancel out
- Returns might be close to correct

**Scenario 2: User has real deposits/withdrawals**
- Current code: BREAKS COMPLETELY
- Real deposits/withdrawals are lost in filtering
- TWR/MWR calculations will be wildly wrong

**Scenario 3: Complex transaction history**
- Auto-generated CASH for buys/sells mixed with real deposits
- Impossible to distinguish correctly
- Returns are unpredictable

## RECOMMENDATIONS

### Must Fix:
1. Add `is_synthetic` boolean flag to transaction schema
2. Tag all auto-generated CASH transactions with `is_synthetic = TRUE`
3. Update performance calculations to exclude `is_synthetic` flows, not all CASH flows
4. Implement proper external flow detection

### Should Fix:
5. Add caching for historical prices
6. Add caching for daily valuation series
7. Implement incremental updates on new transactions
8. Add comprehensive unit tests

### Nice to Have:
9. Consider removing auto-CASH feature entirely (force users to track deposits explicitly)
10. Add data migration to backfill `is_synthetic` flag on existing transactions

## CRITICAL EDGE CASE

If user's transaction data includes REAL deposit/withdrawal transactions (not auto-generated), the current code will incorrectly exclude them from performance calculations, leading to completely wrong returns.

**Test Case Needed**: Portfolio with:
- Initial deposit: $10,000
- Buy: 100 shares @ $100
- Price increases to $150
- Expected TWR: 50% (price gain)
- Current code might show 0% or error (depending on filtering)

## FILES REQUIRING CHANGES

1. `backend/models/portfolio.py` - Add `is_synthetic` field
2. `backend/database.py` - Update schema with migration
3. `backend/main.py` - Tag synthetic transactions on creation
4. `backend/services/performance_v2.py` - Update filtering logic
5. NEW: `backend/services/performance_engine.py` - Canonical engine
6. NEW: `tests/test_performance.py` - Comprehensive tests
