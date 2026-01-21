#!/usr/bin/env python3
"""
Simplified calculation test without yfinance dependency
"""
import sys
import os
sys.path.append('/home/user/Portfolio-Tracker/backend')

# Mock the market_data module before importing anything else
import sys
from unittest.mock import Mock
mock_market_data = Mock()
mock_market_data.get_current_prices = Mock(return_value={})
mock_market_data.get_price_data = Mock(return_value=None)
sys.modules['services.market_data'] = Mock()
sys.modules['services.market_data'].market_data_service = mock_market_data

import sqlite3
import pandas as pd
from datetime import datetime, timezone, timedelta
from models.portfolio import Transaction, Portfolio, Holding
from collections import defaultdict

def load_transactions_from_csv(csv_path: str, account_id: str = 'test_user'):
    """Load transactions from CSV and insert into database"""

    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} transactions from CSV")

    # Connect to database
    conn = sqlite3.connect('/home/user/Portfolio-Tracker/portfolio.db')
    cursor = conn.cursor()

    # Clear existing data for this account
    print(f"Clearing existing data for account '{account_id}'...")
    cursor.execute("DELETE FROM transactions WHERE account_id = ?", (account_id,))
    cursor.execute("DELETE FROM portfolios WHERE account_id = ?", (account_id,))

    # Parse and insert transactions
    print("Inserting transactions...")
    inserted = 0

    for idx, row in df.iterrows():
        try:
            # Parse date
            date_str = str(row['date'])
            date_obj = pd.to_datetime(date_str)

            symbol = str(row['symbol']).upper().strip()
            txn_type = str(row['type']).lower().strip()
            quantity = float(row['quantity']) if not pd.isna(row['quantity']) else 0.0
            price = float(row['price']) if not pd.isna(row['price']) else 0.0
            fees = float(row['fees']) if 'fees' in row and not pd.isna(row['fees']) else 0.0

            # Calculate amount
            if txn_type == 'buy':
                amount = -(quantity * price + fees)
            elif txn_type == 'sell':
                amount = quantity * price - fees
            else:
                amount = quantity * price

            cursor.execute("""
                INSERT INTO transactions
                (account_id, date, symbol, transaction_type, quantity, price, amount, fees, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                account_id,
                date_obj.strftime('%Y-%m-%d'),
                symbol,
                txn_type,
                quantity,
                price,
                amount,
                fees,
                ''
            ))
            inserted += 1

        except Exception as e:
            print(f"Error at row {idx}: {e}")
            continue

    # Create portfolio record
    earliest_date = df['date'].min()
    cursor.execute("""
        INSERT INTO portfolios (account_id, account_name, cash_balance, inception_date)
        VALUES (?, ?, ?, ?)
    """, (account_id, 'Test Portfolio', 0.0, str(earliest_date)))

    conn.commit()
    conn.close()

    print(f"✓ Inserted {inserted} transactions\n")
    return inserted

def analyze_transactions(account_id: str):
    """Analyze transactions to find issues"""

    conn = sqlite3.connect('/home/user/Portfolio-Tracker/portfolio.db')

    # Get all transactions
    df = pd.read_sql_query("""
        SELECT date, symbol, transaction_type, quantity, price, amount, fees
        FROM transactions
        WHERE account_id = ?
        ORDER BY date, id
    """, conn, params=(account_id,))

    conn.close()

    print("="*80)
    print("TRANSACTION ANALYSIS")
    print("="*80)

    print(f"\nTotal transactions: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nUnique symbols: {df['symbol'].nunique()}")
    print(f"Symbols: {sorted(df['symbol'].unique())}")

    print(f"\nTransaction types:")
    print(df['transaction_type'].value_counts())

    print(f"\nTransaction summary by symbol:")
    for symbol in sorted(df['symbol'].unique())[:10]:  # First 10 symbols
        symbol_df = df[df['symbol'] == symbol]
        buys = symbol_df[symbol_df['transaction_type'] == 'buy']
        sells = symbol_df[symbol_df['transaction_type'] == 'sell']

        total_bought = buys['quantity'].sum() if len(buys) > 0 else 0
        total_sold = sells['quantity'].sum() if len(sells) > 0 else 0
        net_qty = total_bought - total_sold

        avg_buy_price = (buys['price'] * buys['quantity']).sum() / buys['quantity'].sum() if len(buys) > 0 and buys['quantity'].sum() > 0 else 0
        avg_sell_price = (sells['price'] * sells['quantity']).sum() / sells['quantity'].sum() if len(sells) > 0 and sells['quantity'].sum() > 0 else 0

        print(f"\n  {symbol}:")
        print(f"    Bought: {total_bought:,.0f} shares @ avg ${avg_buy_price:.2f}" if total_bought > 0 else f"    No buys")
        print(f"    Sold:   {total_sold:,.0f} shares @ avg ${avg_sell_price:.2f}" if total_sold > 0 else f"    No sells")
        print(f"    Current holding: {net_qty:,.0f} shares")

    print("\n" + "="*80)
    print("CHECKING FOR COMMON ISSUES")
    print("="*80)

    # Check for negative prices
    neg_prices = df[df['price'] < 0]
    if len(neg_prices) > 0:
        print(f"\n⚠️  WARNING: Found {len(neg_prices)} transactions with negative prices!")
        print(neg_prices[['date', 'symbol', 'transaction_type', 'price']].head())
    else:
        print(f"\n✓ No negative prices found")

    # Check for zero quantities on buy/sell
    zero_qty = df[((df['transaction_type'] == 'buy') | (df['transaction_type'] == 'sell')) & (df['quantity'] == 0)]
    if len(zero_qty) > 0:
        print(f"\n⚠️  WARNING: Found {len(zero_qty)} buy/sell transactions with zero quantity!")
        print(zero_qty[['date', 'symbol', 'transaction_type', 'quantity']].head())
    else:
        print(f"✓ No zero quantities on buy/sell found")

    # Check for very large quantities (possible data entry errors)
    large_qty = df[df['quantity'] > 1000000]
    if len(large_qty) > 0:
        print(f"\n⚠️  INFO: Found {len(large_qty)} transactions with very large quantities (>1M shares):")
        for _, row in large_qty[['date', 'symbol', 'transaction_type', 'quantity', 'price']].iterrows():
            print(f"    {row['date']} {row['symbol']:6s} {row['transaction_type']:4s} {row['quantity']:,.0f} @ ${row['price']:.2f}")

    # Check amount calculations
    print(f"\n✓ Checking amount calculations...")
    calc_errors = 0
    for idx, row in df.iterrows():
        if row['transaction_type'] == 'buy':
            expected = -(row['quantity'] * row['price'] + row['fees'])
            if abs(row['amount'] - expected) > 0.02:  # Allow 2 cent rounding
                print(f"  ⚠️  Amount mismatch at row {idx}: expected {expected:.2f}, got {row['amount']:.2f}")
                calc_errors += 1
        elif row['transaction_type'] == 'sell':
            expected = row['quantity'] * row['price'] - row['fees']
            if abs(row['amount'] - expected) > 0.02:
                print(f"  ⚠️  Amount mismatch at row {idx}: expected {expected:.2f}, got {row['amount']:.2f}")
                calc_errors += 1

    if calc_errors == 0:
        print(f"✓ All amount calculations correct")
    else:
        print(f"⚠️  Found {calc_errors} amount calculation errors")

    return df

def check_calculation_logic():
    """Check key parts of the calculation logic"""

    print("\n" + "="*80)
    print("EXAMINING CALCULATION CODE")
    print("="*80)

    # Read the performance_v2.py file
    perf_file = '/home/user/Portfolio-Tracker/backend/services/performance_v2.py'
    with open(perf_file, 'r') as f:
        code = f.read()

    print("\nKey findings:")

    # Check for inception date handling
    if 'portfolio_inception' in code and 'adjust' in code.lower():
        print("✓ Code includes portfolio inception date adjustment")
    else:
        print("⚠️  Code may not properly handle portfolio inception date")

    # Check for historical price usage
    if '_fetch_historical_prices' in code:
        print("✓ Code fetches historical prices")
    else:
        print("⚠️  Code may not use historical prices")

    # Check cash flow handling
    if "symbol != 'CASH'" in code or 'CASH' in code:
        print("✓ Code handles CASH transactions")
    else:
        print("⚠️  Code may not filter CASH transactions")

    # Look for potential bugs
    print("\nPotential issues to investigate:")

    # Check TWR calculation
    if 'value_end_before_flows - flows_on_date' in code:
        print("  ℹ️  TWR calculation subtracts flows from end value")
        print("     → This should be CORRECT for TWR (removes the flow impact)")

    # Check amount sign convention
    if 'BUY: negative amount' in code or 'cash outflow' in code:
        print("  ℹ️  Code expects BUY transactions to have negative amounts (cash outflow)")

    if 'SELL: positive amount' in code or 'cash inflow' in code:
        print("  ℹ️  Code expects SELL transactions to have positive amounts (cash inflow)")

def main():
    print("="*80)
    print("SIMPLIFIED PORTFOLIO CALCULATION TEST")
    print("="*80)

    # Load transactions
    csv_path = '/home/user/Portfolio-Tracker/user_transactions.csv'
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found!")
        return

    inserted = load_transactions_from_csv(csv_path)

    if inserted == 0:
        print("ERROR: No transactions were inserted!")
        return

    # Analyze transactions
    df = analyze_transactions('test_user')

    # Check calculation logic
    check_calculation_logic()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review any warnings or errors above")
    print("2. If data looks correct, the issue is likely in the calculation formulas")
    print("3. Key areas to check:")
    print("   - TWR calculation (chain-linking sub-periods)")
    print("   - MWR/IRR calculation (cash flow timing)")
    print("   - Historical price fetching and usage")
    print("   - Start value calculation (should use historical prices)")

if __name__ == '__main__':
    main()
