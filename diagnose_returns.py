#!/usr/bin/env python3
"""
Comprehensive diagnostic for return calculation issues
"""
import sys
sys.path.append('/home/user/Portfolio-Tracker/backend')

import sqlite3
import pandas as pd
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# First, let's analyze the user's actual data
def analyze_user_data():
    """Analyze the actual transaction data to understand the portfolio"""

    conn = sqlite3.connect('portfolio.db')

    # Get all transactions
    query = "SELECT * FROM transactions WHERE account_id = 'test_user' ORDER BY date, id"
    df = pd.read_sql_query(query, conn)
    conn.close()

    print("="*80)
    print("USER DATA ANALYSIS")
    print("="*80)

    print(f"\nTotal transactions: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Transaction types: {df['transaction_type'].value_counts().to_dict()}")

    # Group by symbol
    print("\n" + "="*80)
    print("HOLDINGS ANALYSIS")
    print("="*80)

    holdings = defaultdict(lambda: {'quantity': 0, 'cost_basis': 0, 'num_buys': 0, 'num_sells': 0})

    for _, row in df.iterrows():
        if row['symbol'] == 'CASH':
            continue

        symbol = row['symbol']

        if row['transaction_type'] == 'buy':
            holdings[symbol]['quantity'] += row['quantity']
            holdings[symbol]['cost_basis'] += abs(row['amount'])
            holdings[symbol]['num_buys'] += 1
        elif row['transaction_type'] == 'sell':
            old_qty = holdings[symbol]['quantity']
            holdings[symbol]['quantity'] -= row['quantity']
            if holdings[symbol]['quantity'] > 0:
                # Reduce cost basis proportionally
                cost_per_share = holdings[symbol]['cost_basis'] / old_qty
                holdings[symbol]['cost_basis'] -= row['quantity'] * cost_per_share
            else:
                holdings[symbol]['cost_basis'] = 0
            holdings[symbol]['num_sells'] += 1

    # Show holdings
    for symbol in sorted(holdings.keys())[:5]:  # Top 5
        h = holdings[symbol]
        if h['quantity'] > 0:
            avg_cost = h['cost_basis'] / h['quantity'] if h['quantity'] > 0 else 0
            print(f"\n{symbol}:")
            print(f"  Quantity: {h['quantity']:,.0f} shares")
            print(f"  Cost Basis: ${h['cost_basis']:,.2f}")
            print(f"  Avg Cost: ${avg_cost:.2f}/share")
            print(f"  Transactions: {h['num_buys']} buys, {h['num_sells']} sells")

    # Calculate 1-year period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    print("\n" + "="*80)
    print("1-YEAR PERIOD ANALYSIS")
    print("="*80)
    print(f"Start: {start_date.date()}")
    print(f"End: {end_date.date()}")

    # Transactions in this period
    df['date_parsed'] = pd.to_datetime(df['date'])
    period_df = df[(df['date_parsed'] >= start_date) & (df['date_parsed'] <= end_date)]

    print(f"\nTransactions in 1-year period: {len(period_df)}")
    print(f"  By type: {period_df['transaction_type'].value_counts().to_dict()}")

    # Exclude CASH transactions
    non_cash_df = period_df[period_df['symbol'] != 'CASH']
    print(f"\nNon-CASH transactions: {len(non_cash_df)}")

    # Analyze as external flows
    total_flow = non_cash_df['amount'].sum()
    print(f"  Sum of amounts: ${total_flow:,.2f}")

    # By type
    for txn_type in ['buy', 'sell']:
        type_df = non_cash_df[non_cash_df['transaction_type'] == txn_type]
        if len(type_df) > 0:
            type_sum = type_df['amount'].sum()
            print(f"  {txn_type.upper()}: {len(type_df)} transactions, sum=${type_sum:,.2f}")

    print("\n" + "="*80)
    print("EXTERNAL FLOW INTERPRETATION")
    print("="*80)

    print("\nCurrent interpretation (using transaction amount as-is):")
    print("  BUY: amount = negative (cash out)")
    print("  SELL: amount = positive (cash in)")
    print("  → Total external flow (wrong?): ${:,.2f}".format(total_flow))

    print("\nAlternative interpretation (BUY/SELL as contributions/withdrawals):")
    buy_sum = non_cash_df[non_cash_df['transaction_type'] == 'buy']['amount'].sum()
    sell_sum = non_cash_df[non_cash_df['transaction_type'] == 'sell']['amount'].sum()

    # Flip signs: BUY should be positive contribution, SELL should be negative withdrawal
    flipped_buy = -buy_sum  # Make positive
    flipped_sell = -sell_sum  # Make negative

    print(f"  BUY (flipped): ${flipped_buy:,.2f} (positive contribution)")
    print(f"  SELL (flipped): ${flipped_sell:,.2f} (negative withdrawal)")
    print(f"  → Total net contribution: ${flipped_buy + flipped_sell:,.2f}")

    return df

def diagnose_calculation():
    """Diagnose what the calculation is actually doing"""

    print("\n" + "="*80)
    print("CALCULATION LOGIC DIAGNOSIS")
    print("="*80)

    # Check what the current code considers as external flows
    conn = sqlite3.connect('portfolio.db')
    cursor = conn.cursor()

    # Get non-synthetic transactions
    cursor.execute("""
        SELECT date, symbol, transaction_type, amount, is_synthetic
        FROM transactions
        WHERE account_id = 'test_user'
        AND is_synthetic = 0
        ORDER BY date
        LIMIT 10
    """)

    rows = cursor.fetchall()

    print("\nFirst 10 EXTERNAL transactions (is_synthetic=0):")
    for row in rows:
        print(f"  {row[0]} {row[1]:6s} {row[2]:10s} amt=${row[3]:12,.2f}")

    # Count synthetic vs non-synthetic
    cursor.execute("""
        SELECT is_synthetic, COUNT(*), SUM(amount)
        FROM transactions
        WHERE account_id = 'test_user'
        GROUP BY is_synthetic
    """)

    rows = cursor.fetchall()
    print("\nSynthetic vs External summary:")
    for row in rows:
        is_syn = "SYNTHETIC" if row[0] == 1 else "EXTERNAL"
        print(f"  {is_syn}: {row[1]} transactions, sum=${row[2]:,.2f}")

    conn.close()

    print("\n" + "="*80)
    print("FUNDAMENTAL QUESTION")
    print("="*80)

    print("\nWhat should 'external flow' mean for this portfolio?")
    print("\nOption A: Portfolio = All assets (cash + securities)")
    print("  - BUY: Just reallocation, not external flow")
    print("  - SELL: Just reallocation, not external flow")
    print("  - Only DEPOSIT/WITHDRAWAL are external flows")
    print("  - Problem: User has NO deposit/withdrawal transactions!")

    print("\nOption B: Portfolio = Invested assets only (securities)")
    print("  - BUY: Money going INTO invested portfolio = +contribution")
    print("  - SELL: Money going OUT OF invested portfolio = -withdrawal")
    print("  - Need to flip transaction amount signs!")
    print("  - This makes sense for user's use case!")

    print("\nRecommendation: Use Option B")
    print("  Transform: external_flow = -transaction.amount")
    print("  This treats BUYs as contributions, SELLs as withdrawals")

def test_simple_scenario():
    """Test with a simple manual scenario"""

    print("\n" + "="*80)
    print("SIMPLE TEST SCENARIO")
    print("="*80)

    print("\nScenario:")
    print("  Jan 1, 2024: Buy 100 shares @ $100 = $10,000 invested")
    print("  Jan 1, 2025: Shares worth $111.40 each = $11,140")
    print("  Expected return: 11.4%")

    print("\nModified Dietz formula:")
    print("  R = (End - Start - NetFlows) / (Start + WeightedFlows)")

    print("\nCurrent approach (using amount as-is):")
    start = 0
    end = 11140
    net_flow = -10000  # Buy amount
    weighted_flow = -10000 * 1.0

    if start + weighted_flow != 0:
        ret = (end - start - net_flow) / (start + weighted_flow)
        print(f"  Start={start}, End={end}")
        print(f"  NetFlow={net_flow} (buy treated as negative)")
        print(f"  WeightedFlow={weighted_flow}")
        print(f"  R = ({end} - {start} - ({net_flow})) / ({start} + ({weighted_flow}))")
        print(f"  R = {end - start - net_flow} / {start + weighted_flow}")
        print(f"  R = {ret:.4f} = {ret*100:.2f}%")
        print(f"  ❌ WRONG! Should be 11.4%")
    else:
        print("  ❌ Division by zero!")

    print("\nCorrected approach (flip sign for external flows):")
    net_flow = 10000  # Treat buy as positive contribution
    weighted_flow = 10000 * 1.0

    ret = (end - start - net_flow) / (start + weighted_flow)
    print(f"  Start={start}, End={end}")
    print(f"  NetFlow={net_flow} (buy treated as positive contribution)")
    print(f"  WeightedFlow={weighted_flow}")
    print(f"  R = ({end} - {start} - {net_flow}) / ({start} + {weighted_flow})")
    print(f"  R = {end - start - net_flow} / {start + weighted_flow}")
    print(f"  R = {ret:.4f} = {ret*100:.2f}%")
    print(f"  ✅ CORRECT!")

def main():
    print("="*80)
    print("PORTFOLIO RETURN CALCULATION DIAGNOSTIC")
    print("="*80)
    print("\nThis script diagnoses why returns are incorrect.")
    print("Specifically: Why 1-year return shows 51.79% instead of 11.4%")

    # Analyze user's data
    df = analyze_user_data()

    # Diagnose calculation logic
    diagnose_calculation()

    # Test simple scenario
    test_simple_scenario()

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("\nThe core issue is:")
    print("  Current code treats transaction.amount as external flow directly")
    print("  For BUYs: amount is negative → treated as money leaving")
    print("  For SELLs: amount is positive → treated as money coming in")
    print("  This is BACKWARDS for an 'invested portfolio' interpretation!")

    print("\nThe fix:")
    print("  For external flows, use: -transaction.amount")
    print("  This flips BUYs to positive contributions")
    print("  And flips SELLs to negative withdrawals")
    print("  Matching the intuitive interpretation!")

    print("\nNext step: Implement this fix in performance_v2.py")

if __name__ == '__main__':
    main()
