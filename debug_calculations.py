#!/usr/bin/env python3
"""
Debug performance calculations with real user data
"""
import sys
sys.path.append('/home/user/Portfolio-Tracker/backend')

import sqlite3
import pandas as pd
from datetime import datetime, timezone
from models.portfolio import Transaction, Portfolio, Holding
from services.performance_v2 import performance_calculator
from services.market_data import market_data_service
from collections import defaultdict

def load_transactions_from_csv(csv_path: str, account_id: str = 'test_user'):
    """Load transactions from CSV and insert into database"""

    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} transactions from CSV")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())

    # Connect to database
    conn = sqlite3.connect('/home/user/Portfolio-Tracker/portfolio.db')
    cursor = conn.cursor()

    # Clear existing data for this account
    print(f"\nClearing existing data for account '{account_id}'...")
    cursor.execute("DELETE FROM transactions WHERE account_id = ?", (account_id,))
    cursor.execute("DELETE FROM portfolios WHERE account_id = ?", (account_id,))

    # Parse and insert transactions
    print("\nParsing transactions...")
    inserted = 0
    errors = 0

    for idx, row in df.iterrows():
        try:
            # Parse date - handle different formats
            date_str = str(row['date'])
            try:
                date_obj = pd.to_datetime(date_str)
            except:
                print(f"  Warning: Could not parse date '{date_str}' at row {idx}, skipping")
                errors += 1
                continue

            symbol = str(row['symbol']).upper().strip()
            txn_type = str(row['type']).lower().strip()
            quantity = float(row['quantity']) if not pd.isna(row['quantity']) else 0.0
            price = float(row['price']) if not pd.isna(row['price']) else 0.0
            fees = float(row['fees']) if 'fees' in row and not pd.isna(row['fees']) else 0.0
            notes = str(row['notes']) if 'notes' in row and not pd.isna(row['notes']) else ''

            # Calculate amount based on transaction type
            if txn_type == 'buy':
                amount = -(quantity * price + fees)  # Negative for outflow
            elif txn_type == 'sell':
                amount = quantity * price - fees  # Positive for inflow
            elif txn_type == 'dividend':
                amount = quantity * price  # Positive for income
            else:
                print(f"  Warning: Unknown transaction type '{txn_type}' at row {idx}")
                amount = quantity * price

            # Insert transaction
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
                notes
            ))
            inserted += 1

            if inserted <= 5:
                print(f"  ✓ {date_obj.date()} {txn_type:4s} {quantity:8.2f} {symbol:6s} @ ${price:8.2f} = ${amount:12.2f}")

        except Exception as e:
            print(f"  Error at row {idx}: {e}")
            errors += 1
            continue

    # Create portfolio record
    earliest_date = df['date'].min()
    cursor.execute("""
        INSERT INTO portfolios (account_id, account_name, cash_balance, inception_date)
        VALUES (?, ?, ?, ?)
    """, (account_id, 'Test Portfolio', 0.0, str(earliest_date)))

    conn.commit()
    conn.close()

    print(f"\n✓ Inserted {inserted} transactions")
    if errors > 0:
        print(f"  {errors} errors encountered")

    return inserted, errors

def reconstruct_portfolio_from_db(account_id: str):
    """Reconstruct portfolio and transactions from database"""

    conn = sqlite3.connect('/home/user/Portfolio-Tracker/portfolio.db')
    cursor = conn.cursor()

    # Get transactions
    cursor.execute("""
        SELECT date, symbol, transaction_type, quantity, price, amount, fees, notes
        FROM transactions
        WHERE account_id = ?
        ORDER BY date, id
    """, (account_id,))

    rows = cursor.fetchall()
    conn.close()

    print(f"\nReconstructed {len(rows)} transactions from database")

    # Convert to Transaction objects
    transactions = []
    for row in rows:
        txn = Transaction(
            date=datetime.fromisoformat(row[0] + 'T00:00:00+00:00'),
            symbol=row[1],
            transaction_type=row[2],
            quantity=row[3],
            price=row[4],
            amount=row[5],
            fees=row[6],
            notes=row[7]
        )
        transactions.append(txn)

    # Reconstruct holdings
    holdings_dict = defaultdict(lambda: {'quantity': 0.0, 'cost_basis': 0.0})

    for txn in transactions:
        if txn.symbol == 'CASH':
            continue

        if txn.transaction_type == 'buy':
            holdings_dict[txn.symbol]['quantity'] += txn.quantity
            holdings_dict[txn.symbol]['cost_basis'] += abs(txn.amount)
        elif txn.transaction_type == 'sell':
            old_qty = holdings_dict[txn.symbol]['quantity']
            holdings_dict[txn.symbol]['quantity'] -= txn.quantity
            if holdings_dict[txn.symbol]['quantity'] > 0:
                cost_per_share = holdings_dict[txn.symbol]['cost_basis'] / old_qty
                holdings_dict[txn.symbol]['cost_basis'] -= (txn.quantity * cost_per_share)
            else:
                holdings_dict[txn.symbol]['cost_basis'] = 0

    # Get current prices
    symbols = [s for s in holdings_dict.keys() if holdings_dict[s]['quantity'] > 0]
    print(f"\nFetching current prices for {len(symbols)} symbols: {symbols}")
    current_prices = market_data_service.get_current_prices(symbols) if symbols else {}
    print(f"Got prices for {len(current_prices)} symbols")

    # Create holdings objects
    holdings = []
    total_value = 0.0

    for symbol, data in holdings_dict.items():
        if data['quantity'] > 0:
            current_price = current_prices.get(symbol, 0.0)
            market_value = data['quantity'] * current_price if current_price > 0 else data['cost_basis']

            holding = Holding(
                symbol=symbol,
                quantity=data['quantity'],
                cost_basis=data['cost_basis'],
                current_price=current_price,
                market_value=market_value
            )
            holdings.append(holding)
            total_value += market_value

            print(f"  {symbol:6s}: {data['quantity']:10.2f} shares @ ${current_price:8.2f} = ${market_value:12,.2f}")

    # Create portfolio
    portfolio = Portfolio(
        account_id=account_id,
        account_name='Test Portfolio',
        holdings=holdings,
        transactions=transactions,
        cash_balance=0.0,
        total_value=total_value,
        inception_date=transactions[0].date if transactions else None
    )

    print(f"\nPortfolio total value: ${total_value:,.2f}")
    print(f"Number of holdings: {len(holdings)}")

    return portfolio, transactions

def test_calculations(portfolio, transactions):
    """Run performance calculations and display results"""

    print("\n" + "="*80)
    print("RUNNING PERFORMANCE CALCULATIONS")
    print("="*80)

    results = performance_calculator.calculate_performance_for_all_periods(
        portfolio, transactions
    )

    print("\n" + "="*80)
    print("PORTFOLIO-LEVEL RESULTS")
    print("="*80)

    for period, metrics in results['portfolio'].items():
        print(f"\n{period}:")
        print(f"  Start Value: ${metrics['start_value']:,.2f}")
        print(f"  End Value:   ${metrics['end_value']:,.2f}")
        print(f"  Simple Return: {metrics['simple_return']*100:.2f}%" if metrics['simple_return'] is not None else "  Simple Return: N/A")
        print(f"  TWR:          {metrics['twr']*100:.2f}%" if metrics['twr'] is not None else "  TWR: N/A")
        print(f"  MWR:          {metrics['mwr']*100:.2f}%" if metrics['mwr'] is not None else "  MWR: N/A")

    print("\n" + "="*80)
    print("SECURITY-LEVEL RESULTS (First 5 symbols)")
    print("="*80)

    for i, (symbol, periods) in enumerate(list(results['securities'].items())[:5]):
        print(f"\n{symbol}:")
        for period, metrics in periods.items():
            simple = f"{metrics['simple_return']*100:.2f}%" if metrics['simple_return'] is not None else "N/A"
            twr = f"{metrics['twr']*100:.2f}%" if metrics['twr'] is not None else "N/A"
            mwr = f"{metrics['mwr']*100:.2f}%" if metrics['mwr'] is not None else "N/A"
            print(f"  {period:6s}: Simple={simple:8s} TWR={twr:8s} MWR={mwr:8s}")

    return results

def main():
    print("="*80)
    print("PORTFOLIO CALCULATION DEBUGGER")
    print("="*80)

    # Step 1: Load transactions
    print("\nStep 1: Loading transactions from CSV...")
    csv_path = '/home/user/Portfolio-Tracker/user_transactions.csv'
    inserted, errors = load_transactions_from_csv(csv_path)

    if inserted == 0:
        print("ERROR: No transactions were inserted!")
        return

    # Step 2: Reconstruct portfolio
    print("\nStep 2: Reconstructing portfolio from database...")
    portfolio, transactions = reconstruct_portfolio_from_db('test_user')

    if not transactions:
        print("ERROR: No transactions found!")
        return

    # Step 3: Run calculations
    print("\nStep 3: Running performance calculations...")
    results = test_calculations(portfolio, transactions)

    print("\n" + "="*80)
    print("DEBUGGING COMPLETE")
    print("="*80)
    print("\nIf you see issues above, they need to be fixed in the calculation logic.")

if __name__ == '__main__':
    main()
