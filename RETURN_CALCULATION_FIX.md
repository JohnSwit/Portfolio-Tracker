# Return Calculation Fix - Complete Analysis

## Executive Summary

✅ **ROOT CAUSE IDENTIFIED AND FIXED**

Your returns were incorrect due to **wrong sign interpretation** of transaction amounts when calculating external cash flows.

## The Problem

### Sign Convention Mismatch

**Transaction amounts** (stored in database):
- `BUY`: amount = **negative** (e.g., -$10,000)
  - Represents cash leaving your account to purchase securities

- `SELL`: amount = **positive** (e.g., +$5,000)
  - Represents cash entering your account from selling securities

**External flow interpretation** (for performance calculations):
- `BUY`: should be **positive contribution** (+$10,000)
  - You're investing money into the portfolio

- `SELL`: should be **negative withdrawal** (-$5,000)
  - You're taking money out of the portfolio

### The Bug

The old code used transaction amounts **directly** as external flows:

```python
flows.append((txn_date, txn.amount))  # WRONG!
```

This treated:
- BUYs as money LEAVING (negative) → incorrect!
- SELLs as money ENTERING (positive) → incorrect!

This broke Modified Dietz and MWR calculations completely.

### Example Impact

**Simple scenario:** Invest $10K, grows to $11.14K

**Expected return:** 11.4%

**With bug:**
```
R = ($11,140 - $0 - (-$10,000)) / ($0 + (-$10,000))
R = $21,140 / -$10,000 = -211.4%  ❌ COMPLETELY WRONG
```

**After fix:**
```
R = ($11,140 - $0 - $10,000) / ($0 + $10,000)
R = $1,140 / $10,000 = 11.4%  ✅ CORRECT
```

## The Solution

### Code Changes

**File:** `backend/services/performance_v2.py`

**Function:** `_get_portfolio_cash_flows()` and `_get_security_cash_flows()`

**Fix:** Flip the sign when converting to external flows:

```python
# OLD (WRONG):
flows.append((txn_date, txn.amount))

# NEW (CORRECT):
external_flow = -txn.amount  # Flip the sign!
flows.append((txn_date, external_flow))
```

### Why This Works

```
Transaction Amount → External Flow

BUY  $10K: amount = -$10,000 → external_flow = -(-$10,000) = +$10,000 ✓
SELL $5K:  amount = +$5,000  → external_flow = -(+$5,000)  = -$5,000  ✓
```

## What Was Fixed

✅ **Modified Dietz (Simple Return):** Now correctly treats BUY/SELL as contributions/withdrawals
✅ **MWR/IRR:** Cash flows have correct signs for IRR calculation
✅ **Security-level returns:** Same fix applied to individual stock calculations
✅ **TWR:** Already correct (uses internal amounts, which is proper for TWR)

## Testing

### Automated Test Results

**Test 1: Simple Scenario** ✅ PASS
```
Scenario: Invest $10,000, grows to $11,140
Expected: 11.4%
Result: 11.4%
```

**Test 2: Your Real Portfolio**
- Requires live market data to test properly
- See below for how to test with your actual data

## How to Test With Your Real Data

### Step 1: Start the Backend

```bash
cd /home/user/Portfolio-Tracker/backend
python main.py
```

The backend will:
1. Run database migrations automatically
2. Load your 109 transactions
3. Be ready to calculate returns with real market prices

### Step 2: Upload Your Transactions (if not already done)

```bash
curl -X POST "http://localhost:8000/api/upload_csv/my_portfolio" \
  -F "file=@user_transactions.csv"
```

### Step 3: Calculate Performance

```bash
# Get portfolio performance for all time periods
curl "http://localhost:8000/api/performance/my_portfolio" | jq

# Or get specific time period
curl "http://localhost:8000/api/portfolio/my_portfolio" | jq
```

### Step 4: Verify Results

The returns should now be:
- ✅ Reasonable (typically -50% to +100% for yearly returns)
- ✅ Consistent across timeframes (longer periods smooth out volatility)
- ✅ Matching your expectations based on market conditions

### What to Check

**1. Sign Check:**
- All returns should have reasonable magnitudes
- No more crazy values like -211% or +500%

**2. Sanity Check:**
- If you've been mostly buying and holding tech stocks from 2023-2025, expect positive returns
- Recent periods (1M, 3M) may vary based on market conditions
- Longer periods (1Y, 3Y) should show your overall performance

**3. Consistency Check:**
- TWR should be close to Simple Return if cash flows are small
- MWR may differ if you had bad timing on contributions
- All three should be in same ballpark (within 10-20 percentage points)

## Understanding Your Results

### Modified Dietz (Simple Return)
- **Fast approximation** of return
- Weights cash flows by how long they were invested
- Good for quick checks

### TWR (Time-Weighted Return)
- **True market performance** measure
- Removes impact of your contribution timing
- What a fund manager would report

### MWR (Money-Weighted Return / IRR)
- **Your personal return** including timing effects
- Accounts for when you added/removed money
- What you actually earned

## Common Issues

### Issue: Returns still look wrong

**Check 1:** Are you using real market data?
- The test script uses fallback prices (cost basis)
- Real calculations need actual market prices via yfinance
- Start the backend to fetch real prices

**Check 2:** Are transactions dated correctly?
- Check your CSV for wrong dates (future dates, typos)
- Run: `python diagnose_returns.py` to see transaction summary

**Check 3:** Are there synthetic transactions polluting the data?
- Synthetic CASH transactions should have `is_synthetic=1`
- Run migration again if needed: `python -c "import database; database.migrate_database()"`

### Issue: Some stocks show N/A for returns

**Cause:** Stock has no transactions in the period being calculated

**Solution:** This is normal - some stocks are buy-and-hold for full period

### Issue: TWR very different from Simple Return

**Possible causes:**
1. Large cash flows during the period (TWR excludes impact, Simple includes)
2. Volatile period with transactions at bad/good times
3. One of the calculations has a bug (verify with the other)

## Files Modified

```
backend/services/performance_v2.py  (External flow sign fix)
diagnose_returns.py                 (Diagnostic tool)
test_real_data_returns.py           (Test suite)
RETURN_CALCULATION_FIX.md           (This file)
```

## Next Steps

1. **Start the backend** to test with real data
2. **Check a known period** where you know the approximate return
3. **Verify all timeframes** (1M, 3M, 6M, YTD, 1Y, etc.)
4. **Compare securities** - individual stock returns should make sense

## Technical Details

### Why TWR Doesn't Need the Sign Flip

TWR calculation uses transaction amounts to **back out** the transaction impact from portfolio values:

```python
value_end_before_flows = calculate_value(date)  # Includes bought securities
flows_on_date = sum(t.amount)                   # BUY = negative
value_end = value_end_before_flows + flows_on_date  # Backs out the buy

# For BUY: value_end = value_with_securities + (-purchase_amount) = value_before_buy ✓
```

This is **different** from external flow perspective and is **correct** for TWR.

### Modified Dietz Formula

```
R = (End Value - Start Value - Net Contributions) / (Start Value + Weighted Contributions)

Where:
- Net Contributions = sum of external flows (with correct signs!)
- Weighted Contributions = time-weighted sum of flows
- Weight = (Period Length - Days Since Flow) / Period Length
```

### MWR/IRR Formula

```
NPV(r) = -Start Value + sum(Flow_i / (1+r)^(t_i)) + End Value / (1+r)^T = 0

Solve for r where:
- Flow_i = external flow on date i (with correct signs!)
- t_i = years from start to flow date
- T = years from start to end
```

## Success Criteria

✅ Simple test scenario gives 11.4% (verified)
✅ No negative returns over 100%
✅ No positive returns over 1000%
✅ TWR, MWR, and Simple Return are within reasonable range of each other
✅ Returns match your intuition about market performance

## Support

If returns still seem incorrect after testing with real data:

1. Run diagnostic: `python diagnose_returns.py`
2. Check the output for anomalies
3. Verify transaction data is clean (no duplicate dates, reasonable amounts)
4. Check that market prices are being fetched successfully

---

**Summary:** The sign flip fix corrects the fundamental misinterpretation of BUY/SELL transactions. Your returns should now be accurate when tested with real market data through the backend API.
