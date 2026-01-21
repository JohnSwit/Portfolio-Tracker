#!/usr/bin/env python3
"""
Script to add test transaction data directly to the database.
This is useful for testing calculations without running the full API.
"""

import sqlite3
from datetime import datetime

# Test transactions for portfolio
test_transactions = [
    {
        'account_id': 'test_account',
        'date': '2024-01-15',
        'symbol': 'AAPL',
        'transaction_type': 'buy',
        'quantity': 100,
        'price': 150.00,
        'amount': 15000.00,
        'fees': 9.99,
        'notes': 'Initial AAPL purchase'
    },
    {
        'account_id': 'test_account',
        'date': '2024-01-15',
        'symbol': 'CASH',
        'transaction_type': 'deposit',
        'quantity': 0,
        'price': 0,
        'amount': 15009.99,
        'fees': 0,
        'notes': 'Cash deposit for AAPL purchase'
    },
    {
        'account_id': 'test_account',
        'date': '2024-02-20',
        'symbol': 'MSFT',
        'transaction_type': 'buy',
        'quantity': 50,
        'price': 380.00,
        'amount': 19000.00,
        'fees': 9.99,
        'notes': 'MSFT purchase'
    },
    {
        'account_id': 'test_account',
        'date': '2024-02-20',
        'symbol': 'CASH',
        'transaction_type': 'deposit',
        'quantity': 0,
        'price': 0,
        'amount': 19009.99,
        'fees': 0,
        'notes': 'Cash deposit for MSFT purchase'
    },
    {
        'account_id': 'test_account',
        'date': '2024-03-10',
        'symbol': 'AAPL',
        'transaction_type': 'sell',
        'quantity': 25,
        'price': 170.00,
        'amount': 4250.00,
        'fees': 9.99,
        'notes': 'Partial AAPL sale'
    },
    {
        'account_id': 'test_account',
        'date': '2024-03-10',
        'symbol': 'CASH',
        'transaction_type': 'withdrawal',
        'quantity': 0,
        'price': 0,
        'amount': 4240.01,
        'fees': 0,
        'notes': 'Cash withdrawal from AAPL sale'
    },
    {
        'account_id': 'test_account',
        'date': '2024-04-01',
        'symbol': 'AAPL',
        'transaction_type': 'dividend',
        'quantity': 0,
        'price': 1.24,
        'amount': 93.00,
        'fees': 0,
        'notes': 'AAPL quarterly dividend (75 shares * $1.24)'
    },
    {
        'account_id': 'test_account',
        'date': '2024-04-01',
        'symbol': 'CASH',
        'transaction_type': 'withdrawal',
        'quantity': 0,
        'price': 0,
        'amount': 93.00,
        'fees': 0,
        'notes': 'Cash withdrawal from dividend'
    },
]

def add_test_data():
    """Add test transactions to the database."""

    # Connect to database
    conn = sqlite3.connect('portfolio.db')
    cursor = conn.cursor()

    # First, clear any existing test data
    print("Clearing existing test_account data...")
    cursor.execute("DELETE FROM transactions WHERE account_id = 'test_account'")
    cursor.execute("DELETE FROM portfolios WHERE account_id = 'test_account'")

    # Insert test transactions
    print(f"\nInserting {len(test_transactions)} test transactions...")
    for txn in test_transactions:
        cursor.execute("""
            INSERT INTO transactions
            (account_id, date, symbol, transaction_type, quantity, price, amount, fees, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            txn['account_id'],
            txn['date'],
            txn['symbol'],
            txn['transaction_type'],
            txn['quantity'],
            txn['price'],
            txn['amount'],
            txn['fees'],
            txn['notes']
        ))
        print(f"  ✓ {txn['date']} - {txn['transaction_type']} {txn['quantity']} {txn['symbol']} @ ${txn['price']}")

    # Create portfolio record
    print("\nCreating portfolio record...")
    cursor.execute("""
        INSERT INTO portfolios (account_id, account_name, cash_balance, inception_date)
        VALUES (?, ?, ?, ?)
    """, ('test_account', 'Test Portfolio', 0.0, '2024-01-15'))

    # Commit changes
    conn.commit()
    conn.close()

    print("\n✅ Test data added successfully!")
    print("\nYou can now test the calculations by running:")
    print("  curl http://localhost:8000/api/portfolio/test_account")
    print("\nOr view transactions:")
    print("  curl http://localhost:8000/api/transactions/test_account")

def view_current_data():
    """Display current transactions in the database."""

    conn = sqlite3.connect('portfolio.db')
    cursor = conn.cursor()

    cursor.execute("""
        SELECT date, symbol, transaction_type, quantity, price, amount, fees
        FROM transactions
        WHERE account_id = 'test_account'
        ORDER BY date, id
    """)

    rows = cursor.fetchall()

    if rows:
        print("\nCurrent test_account transactions:")
        print("-" * 80)
        for row in rows:
            print(f"{row[0]} | {row[2]:10s} | {row[1]:6s} | Qty: {row[3]:6.2f} | Price: ${row[4]:7.2f} | Amount: ${row[5]:9.2f}")
    else:
        print("\nNo transactions found for test_account")

    conn.close()

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--view':
        view_current_data()
    else:
        add_test_data()
        print("\nTo view the data you just added, run:")
        print("  python add_test_data.py --view")
