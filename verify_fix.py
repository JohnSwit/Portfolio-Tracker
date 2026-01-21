#!/usr/bin/env python3
"""
Verify that the TWR calculation fix works correctly
"""
import sys
sys.path.append('/home/user/Portfolio-Tracker/backend')

# Mock the yfinance dependency
from unittest.mock import Mock, MagicMock
import pandas as pd
from datetime import datetime, timedelta

mock_market_data = MagicMock()

# Mock price data - all stocks at $10 for simplicity
def mock_get_current_prices(symbols):
    return {symbol: 10.0 for symbol in symbols}

def mock_get_price_data(symbols, start, end, interval="1d"):
    """Return mock price data - constant $10 price"""
    dates = pd.date_range(start=start, end=end, freq='D')
    data = {}

    if len(symbols) == 1:
        # Single symbol
        df = pd.DataFrame({
            'Close': [10.0] * len(dates)
        }, index=dates)
        return df
    else:
        # Multiple symbols - return MultiIndex DataFrame
        for symbol in symbols:
            data[symbol] = pd.Series([10.0] * len(dates), index=dates)
        df = pd.DataFrame(data)
        return df

mock_market_data.get_current_prices = mock_get_current_prices
mock_market_data.get_price_data = mock_get_price_data

sys.modules['services.market_data'] = Mock()
sys.modules['services.market_data'].market_data_service = mock_market_data

# Now import the calculator
from services.performance_v2 import PerformanceCalculator
from models.portfolio import Transaction, Portfolio, Holding
from datetime import datetime, timezone

def test_simple_buy_scenario():
    """Test: Buy with no price change should give 0% TWR"""
    print("\n" + "="*70)
    print("TEST 1: Buy transaction with no price change")
    print("="*70)

    # Create simple transaction history
    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    buy_date = datetime(2024, 1, 15, tzinfo=timezone.utc)
    end_date = datetime(2024, 1, 31, tzinfo=timezone.utc)

    transactions = [
        # Initial position: 10 shares @ $10
        Transaction(
            date=start_date,
            symbol='TEST',
            transaction_type='buy',
            quantity=10,
            price=10.0,
            amount=-100.0,  # $100 outflow
            fees=0.0
        ),
        # Buy more: 5 shares @ $10
        Transaction(
            date=buy_date,
            symbol='TEST',
            transaction_type='buy',
            quantity=5,
            price=10.0,
            amount=-50.0,  # $50 outflow
            fees=0.0
        )
    ]

    # Create portfolio
    holdings = [
        Holding(
            symbol='TEST',
            quantity=15,
            cost_basis=150.0,
            current_price=10.0,
            market_value=150.0
        )
    ]

    portfolio = Portfolio(
        account_id='test',
        holdings=holdings,
        transactions=transactions,
        cash_balance=0.0,
        total_value=150.0,
        inception_date=start_date
    )

    # Calculate TWR
    calc = PerformanceCalculator()
    result = calc._calculate_twr_portfolio(transactions, start_date, end_date)

    print(f"Initial: 10 shares @ $10 = $100")
    print(f"Action:  Buy 5 shares @ $10 = $50 (on {buy_date.date()})")
    print(f"Final:   15 shares @ $10 = $150")
    print(f"Price:   No change ($10 throughout)")
    print(f"\nCalculated TWR: {result:.4f} ({result*100:.2f}%)")
    print(f"Expected TWR:   0.0000 (0.00%)")

    if abs(result) < 0.0001:  # Allow small floating point error
        print("✅ PASS: TWR is correctly 0%")
        return True
    else:
        print(f"❌ FAIL: TWR should be 0% but got {result*100:.2f}%")
        return False

def test_simple_sell_scenario():
    """Test: Sell with no price change should give 0% TWR"""
    print("\n" + "="*70)
    print("TEST 2: Sell transaction with no price change")
    print("="*70)

    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    sell_date = datetime(2024, 1, 15, tzinfo=timezone.utc)
    end_date = datetime(2024, 1, 31, tzinfo=timezone.utc)

    transactions = [
        # Initial position: 10 shares @ $10
        Transaction(
            date=start_date,
            symbol='TEST',
            transaction_type='buy',
            quantity=10,
            price=10.0,
            amount=-100.0,
            fees=0.0
        ),
        # Sell: 5 shares @ $10
        Transaction(
            date=sell_date,
            symbol='TEST',
            transaction_type='sell',
            quantity=5,
            price=10.0,
            amount=50.0,  # $50 inflow
            fees=0.0
        )
    ]

    holdings = [
        Holding(
            symbol='TEST',
            quantity=5,
            cost_basis=50.0,
            current_price=10.0,
            market_value=50.0
        )
    ]

    portfolio = Portfolio(
        account_id='test',
        holdings=holdings,
        transactions=transactions,
        cash_balance=0.0,
        total_value=50.0,
        inception_date=start_date
    )

    calc = PerformanceCalculator()
    result = calc._calculate_twr_portfolio(transactions, start_date, end_date)

    print(f"Initial: 10 shares @ $10 = $100")
    print(f"Action:  Sell 5 shares @ $10 = $50 (on {sell_date.date()})")
    print(f"Final:   5 shares @ $10 = $50")
    print(f"Price:   No change ($10 throughout)")
    print(f"\nCalculated TWR: {result:.4f} ({result*100:.2f}%)")
    print(f"Expected TWR:   0.0000 (0.00%)")

    if abs(result) < 0.0001:
        print("✅ PASS: TWR is correctly 0%")
        return True
    else:
        print(f"❌ FAIL: TWR should be 0% but got {result*100:.2f}%")
        return False

def test_price_increase_with_buy():
    """Test: Price increase with buy should show correct TWR"""
    print("\n" + "="*70)
    print("TEST 3: Buy transaction with price increase")
    print("="*70)

    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    buy_date = datetime(2024, 1, 15, tzinfo=timezone.utc)
    end_date = datetime(2024, 1, 31, tzinfo=timezone.utc)

    # Override mock to provide price increase
    def mock_get_price_data_with_increase(symbols, start, end, interval="1d"):
        dates = pd.date_range(start=start, end=end, freq='D')
        prices = []
        for date in dates:
            if date < pd.Timestamp(buy_date):
                prices.append(10.0)  # $10 before buy
            else:
                prices.append(12.0)  # $12 after buy (20% increase)

        if len(symbols) == 1:
            return pd.DataFrame({'Close': prices}, index=dates)
        else:
            data = {symbol: prices for symbol in symbols}
            return pd.DataFrame(data, index=dates)

    mock_market_data.get_price_data = mock_get_price_data_with_increase
    mock_market_data.get_current_prices = lambda s: {sym: 12.0 for sym in s}

    transactions = [
        # Initial: 10 shares @ $10
        Transaction(
            date=start_date,
            symbol='TEST',
            transaction_type='buy',
            quantity=10,
            price=10.0,
            amount=-100.0,
            fees=0.0
        ),
        # Buy: 5 shares @ $12 (after price increase)
        Transaction(
            date=buy_date,
            symbol='TEST',
            transaction_type='buy',
            quantity=5,
            price=12.0,
            amount=-60.0,
            fees=0.0
        )
    ]

    holdings = [
        Holding(
            symbol='TEST',
            quantity=15,
            cost_basis=160.0,
            current_price=12.0,
            market_value=180.0
        )
    ]

    portfolio = Portfolio(
        account_id='test',
        holdings=holdings,
        transactions=transactions,
        cash_balance=0.0,
        total_value=180.0,
        inception_date=start_date
    )

    calc = PerformanceCalculator()
    result = calc._calculate_twr_portfolio(transactions, start_date, end_date)

    print(f"Initial: 10 shares @ $10 = $100")
    print(f"Price increases to $12 (+20%)")
    print(f"Action:  Buy 5 shares @ $12 = $60")
    print(f"Final:   15 shares @ $12 = $180")
    print(f"\nExpected TWR: ~20% (price increased 20%)")
    print(f"Calculated TWR: {result:.4f} ({result*100:.2f}%)")

    # TWR should be approximately 20% (the price increase)
    if 0.15 < result < 0.25:  # Allow some range
        print("✅ PASS: TWR is approximately 20%")
        return True
    else:
        print(f"❌ FAIL: TWR should be ~20% but got {result*100:.2f}%")
        return False

def main():
    print("="*70)
    print("TWR CALCULATION FIX VERIFICATION")
    print("="*70)
    print("\nTesting the fixes to lines 445 and 508 of performance_v2.py")
    print("These tests use mock price data to verify calculation logic.")

    results = []

    try:
        results.append(test_simple_buy_scenario())
    except Exception as e:
        print(f"❌ Test 1 CRASHED: {e}")
        import traceback
        traceback.print_exc()
        results.append(False)

    try:
        results.append(test_simple_sell_scenario())
    except Exception as e:
        print(f"❌ Test 2 CRASHED: {e}")
        import traceback
        traceback.print_exc()
        results.append(False)

    try:
        results.append(test_price_increase_with_buy())
    except Exception as e:
        print(f"❌ Test 3 CRASHED: {e}")
        import traceback
        traceback.print_exc()
        results.append(False)

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✅ ALL TESTS PASSED! The TWR calculation fix is working correctly.")
    else:
        print(f"\n❌ {total - passed} test(s) failed. There may still be issues.")

    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
