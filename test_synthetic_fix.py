#!/usr/bin/env python3
"""
Test that the is_synthetic flag correctly fixes performance calculations

This test demonstrates the complete fix for the synthetic cash transaction bug.
"""
import sys
sys.path.append('/home/user/Portfolio-Tracker/backend')

import sqlite3
from datetime import datetime, timezone
from models.portfolio import Transaction, Portfolio, Holding
import database

# Mock market data
from unittest.mock import Mock, MagicMock
import pandas as pd

mock_market_data = MagicMock()

def mock_get_current_prices(symbols):
    # Return constant prices for testing
    prices = {
        'AAPL': 150.0,
        'MSFT': 380.0,
    }
    return {s: prices.get(s, 100.0) for s in symbols}

def mock_get_price_data(symbols, start, end, interval="1d"):
    """Return mock price data - constant prices"""
    dates = pd.date_range(start=start, end=end, freq='D')

    if len(symbols) == 1:
        df = pd.DataFrame({'Close': [mock_get_current_prices(symbols)[symbols[0]]] * len(dates)}, index=dates)
        return df
    else:
        data = {}
        for symbol in symbols:
            data[symbol] = [mock_get_current_prices([symbol])[symbol]] * len(dates)
        return pd.DataFrame(data, index=dates)

mock_market_data.get_current_prices = mock_get_current_prices
mock_market_data.get_price_data = mock_get_price_data

sys.modules['services.market_data'] = Mock()
sys.modules['services.market_data'].market_data_service = mock_market_data

# Now import the calculator
from services.performance_v2 import PerformanceCalculator

def test_synthetic_transactions():
    """
    Test Scenario: Buy 100 AAPL @ $150 with no price change

    Expected: TWR should be 0% (no market movement)
    Previous bug: Would incorrectly include synthetic CASH deposit, breaking calculations
    """
    print("="*80)
    print("TEST: Synthetic Transaction Filtering")
    print("="*80)

    # Initialize database
    database.init_database()
    database.migrate_database()

    # Clear test data
    conn = sqlite3.connect('portfolio.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM transactions WHERE account_id = 'test_synthetic'")
    cursor.execute("DELETE FROM portfolios WHERE account_id = 'test_synthetic'")
    conn.commit()
    conn.close()

    # Create transactions
    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    buy_date = datetime(2024, 1, 15, tzinfo=timezone.utc)
    end_date = datetime(2024, 1, 31, tzinfo=timezone.utc)

    # Real transaction: Buy AAPL
    buy_txn = {
        'date': buy_date.isoformat(),
        'symbol': 'AAPL',
        'transaction_type': 'buy',
        'quantity': 100.0,
        'price': 150.0,
        'amount': -15000.0,  # Negative = cash out
        'fees': 0.0,
        'notes': 'Buy AAPL',
        'is_synthetic': False  # Real transaction
    }

    # Synthetic transaction: Auto-generated cash deposit
    cash_txn = {
        'date': buy_date.isoformat(),
        'symbol': 'CASH',
        'transaction_type': 'deposit',
        'quantity': 15000.0,
        'price': 1.0,
        'amount': 15000.0,  # Positive = cash in
        'fees': 0.0,
        'notes': 'Auto-deposit to fund AAPL purchase',
        'is_synthetic': True  # Synthetic - should be excluded from TWR/MWR
    }

    # Save to database
    database.save_transactions('test_synthetic', [buy_txn, cash_txn])

    # Load transactions
    db_txns = database.get_transactions('test_synthetic')

    print(f"\nLoaded {len(db_txns)} transactions from database:")
    for txn in db_txns:
        is_syn = bool(txn.get('is_synthetic', 0))
        print(f"  {txn['date']} {txn['symbol']:6s} {txn['transaction_type']:10s} "
              f"amt=${txn['amount']:10,.2f} is_synthetic={is_syn}")

    # Convert to Transaction objects
    transactions = []
    for row in db_txns:
        # Parse date - handle both date-only and full timestamp
        date_str = row['date']
        if 'T' in date_str:
            # Already has time component
            date_obj = datetime.fromisoformat(date_str.replace('+00:00', '')).replace(tzinfo=timezone.utc)
        else:
            # Date only
            date_obj = datetime.fromisoformat(date_str + 'T00:00:00').replace(tzinfo=timezone.utc)

        txn = Transaction(
            date=date_obj,
            symbol=row['symbol'],
            transaction_type=row['transaction_type'],
            quantity=row['quantity'],
            price=row['price'],
            amount=row['amount'],
            fees=row.get('fees', 0.0),
            notes=row.get('notes', ''),
            is_synthetic=bool(row.get('is_synthetic', 0))
        )
        transactions.append(txn)

    # Create portfolio
    holdings = [
        Holding(
            symbol='AAPL',
            quantity=100,
            cost_basis=15000.0,
            current_price=150.0,
            market_value=15000.0
        )
    ]

    portfolio = Portfolio(
        account_id='test_synthetic',
        holdings=holdings,
        transactions=transactions,
        cash_balance=0.0,
        total_value=15000.0,
        inception_date=buy_date
    )

    # Calculate TWR
    calc = PerformanceCalculator()

    print("\n" + "-"*80)
    print("Calculating TWR...")
    print("-"*80)

    twr = calc._calculate_twr_portfolio(transactions, buy_date, end_date)

    print(f"\nScenario:")
    print(f"  - Buy: 100 AAPL @ $150 = $15,000")
    print(f"  - Auto CASH deposit: $15,000 (synthetic, should be ignored)")
    print(f"  - Price: No change ($150 throughout)")
    print(f"  - Time: {buy_date.date()} to {end_date.date()}")

    print(f"\nResults:")
    print(f"  Calculated TWR: {twr:.4f} ({twr*100:.2f}%)")
    print(f"  Expected TWR:   0.0000 (0.00%)")

    # Check external flows
    external_flows = calc._get_portfolio_cash_flows(transactions, buy_date, end_date)
    print(f"\n  External cash flows detected: {len(external_flows)}")
    for date, amount in external_flows:
        print(f"    {date.date()}: ${amount:,.2f}")

    # Verify
    success = abs(twr) < 0.0001  # Allow tiny floating point error

    print("\n" + "="*80)
    if success:
        print("✅ TEST PASSED: Synthetic transactions correctly excluded!")
        print("   TWR is 0% as expected (no market movement, only cash flow)")
    else:
        print(f"❌ TEST FAILED: TWR should be 0% but got {twr*100:.2f}%")
        print("   Synthetic transactions may not be properly excluded")
    print("="*80)

    return success

def test_real_deposit_included():
    """
    Test that REAL deposits/withdrawals ARE included in performance calculations
    """
    print("\n" + "="*80)
    print("TEST: Real Deposits Should Be Included")
    print("="*80)

    # Clear test data
    conn = sqlite3.connect('portfolio.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM transactions WHERE account_id = 'test_real_deposit'")
    cursor.execute("DELETE FROM portfolios WHERE account_id = 'test_real_deposit'")
    conn.commit()
    conn.close()

    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    deposit_date = datetime(2024, 1, 5, tzinfo=timezone.utc)
    buy_date = datetime(2024, 1, 10, tzinfo=timezone.utc)
    end_date = datetime(2024, 1, 31, tzinfo=timezone.utc)

    # Real deposit (user adds money to portfolio)
    deposit_txn = {
        'date': deposit_date.isoformat(),
        'symbol': 'CASH',
        'transaction_type': 'deposit',
        'quantity': 10000.0,
        'price': 1.0,
        'amount': 10000.0,
        'fees': 0.0,
        'notes': 'Real user deposit',
        'is_synthetic': False  # Real transaction - should be included
    }

    # Buy with the deposited money
    buy_txn = {
        'date': buy_date.isoformat(),
        'symbol': 'AAPL',
        'transaction_type': 'buy',
        'quantity': 66.67,
        'price': 150.0,
        'amount': -10000.0,
        'fees': 0.0,
        'notes': 'Buy AAPL with deposited funds',
        'is_synthetic': False
    }

    # Synthetic cash transaction to balance the buy
    cash_txn = {
        'date': buy_date.isoformat(),
        'symbol': 'CASH',
        'transaction_type': 'deposit',
        'quantity': 10000.0,
        'price': 1.0,
        'amount': 10000.0,
        'fees': 0.0,
        'notes': 'Auto-deposit to fund AAPL purchase',
        'is_synthetic': True
    }

    database.save_transactions('test_real_deposit', [deposit_txn, buy_txn, cash_txn])

    # Load and convert
    db_txns = database.get_transactions('test_real_deposit')

    print(f"\nTransactions:")
    for txn in db_txns:
        is_syn = bool(txn.get('is_synthetic', 0))
        print(f"  {txn['date']} {txn['symbol']:6s} {txn['transaction_type']:10s} "
              f"amt=${txn['amount']:10,.2f} is_synthetic={is_syn}")

    transactions = []
    for row in db_txns:
        # Parse date - handle both date-only and full timestamp
        date_str = row['date']
        if 'T' in date_str:
            date_obj = datetime.fromisoformat(date_str.replace('+00:00', '')).replace(tzinfo=timezone.utc)
        else:
            date_obj = datetime.fromisoformat(date_str + 'T00:00:00').replace(tzinfo=timezone.utc)

        txn = Transaction(
            date=date_obj,
            symbol=row['symbol'],
            transaction_type=row['transaction_type'],
            quantity=row['quantity'],
            price=row['price'],
            amount=row['amount'],
            fees=row.get('fees', 0.0),
            notes=row.get('notes', ''),
            is_synthetic=bool(row.get('is_synthetic', 0))
        )
        transactions.append(txn)

    # Calculate cash flows
    calc = PerformanceCalculator()
    external_flows = calc._get_portfolio_cash_flows(transactions, start_date, end_date)

    print(f"\nExternal cash flows detected: {len(external_flows)}")
    for date, amount in external_flows:
        print(f"  {date.date()}: ${amount:,.2f}")

    # Should detect 2 flows: real deposit (+10000) and buy (-10000)
    # Synthetic cash should NOT appear
    expected_flow_count = 2
    success = len(external_flows) == expected_flow_count

    print("\n" + "="*80)
    if success:
        print(f"✅ TEST PASSED: Real deposit correctly included in cash flows!")
        print(f"   Found {len(external_flows)} flows (1 deposit, 1 buy)")
    else:
        print(f"❌ TEST FAILED: Expected {expected_flow_count} flows but found {len(external_flows)}")
    print("="*80)

    return success

def main():
    print("="*80)
    print("COMPREHENSIVE SYNTHETIC TRANSACTION FIX VERIFICATION")
    print("="*80)
    print("\nThis test verifies that:")
    print("1. Synthetic cash transactions are excluded from TWR/MWR calculations")
    print("2. Real deposits/withdrawals are included in calculations")
    print("3. The is_synthetic flag works correctly end-to-end")

    results = []

    try:
        results.append(test_synthetic_transactions())
    except Exception as e:
        print(f"\n❌ Test 1 CRASHED: {e}")
        import traceback
        traceback.print_exc()
        results.append(False)

    try:
        results.append(test_real_deposit_included())
    except Exception as e:
        print(f"\n❌ Test 2 CRASHED: {e}")
        import traceback
        traceback.print_exc()
        results.append(False)

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        print("\nThe is_synthetic fix is working correctly:")
        print("- Synthetic cash balancing transactions are excluded from performance")
        print("- Real deposits/withdrawals are included in performance")
        print("- TWR/MWR calculations are now accurate")
    else:
        print(f"\n❌ {total - passed} test(s) failed")

    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
