#!/usr/bin/env python3
"""
Test to verify TWR calculation bug

Scenario:
- Start with $100 in stock
- Buy $50 more stock (at same price, so no market movement)
- End with $150 in stock

Expected TWR: 0% (no market movement, only cash flow)
Current bug: Will show incorrect return due to sign error
"""

# Simple example
print("="*60)
print("TWR CALCULATION BUG DEMONSTRATION")
print("="*60)

print("\nScenario:")
print("- Start: Own 10 shares @ $10/share = $100")
print("- Action: Buy 5 more shares @ $10/share = $50")
print("- End: Own 15 shares @ $10/share = $150")
print("- Market movement: 0% (price stayed at $10)")
print("\nExpected TWR: 0% (no price change)")

print("\n" + "-"*60)
print("CURRENT CODE LOGIC (performance_v2.py line 445):")
print("-"*60)

value_start = 100.0
value_end_including_flow = 150.0  # This is what _calculate_portfolio_value_at_date returns

# In our system, BUY transactions have NEGATIVE amount (cash outflow)
buy_amount = -(5 * 10)  # -50 (5 shares @ $10, cash leaving account)
flows_on_date = buy_amount  # -50

# Current code (LINE 445):
value_end_current = value_end_including_flow - flows_on_date
period_return_current = (value_end_current - value_start) / value_start

print(f"\nvalue_start = ${value_start}")
print(f"value_end_including_flow = ${value_end_including_flow}")
print(f"flows_on_date (buy amount) = ${flows_on_date}")
print(f"\nCurrent calculation (line 445):")
print(f"  value_end = value_end_including_flow - flows_on_date")
print(f"  value_end = {value_end_including_flow} - ({flows_on_date})")
print(f"  value_end = {value_end_current}")
print(f"\n  TWR = (value_end - value_start) / value_start")
print(f"  TWR = ({value_end_current} - {value_start}) / {value_start}")
print(f"  TWR = {period_return_current:.2%}")
print(f"\n  ❌ BUG: Shows {period_return_current:.2%} when it should be 0%!")

print("\n" + "-"*60)
print("CORRECT LOGIC:")
print("-"*60)

# To back out a buy, we need to subtract the VALUE added (which is positive)
# Since amount = -50 for a buy, we need to ADD it (double negative)
value_end_correct = value_end_including_flow + flows_on_date
period_return_correct = (value_end_correct - value_start) / value_start

print(f"\nCorrect calculation:")
print(f"  value_end = value_end_including_flow + flows_on_date")
print(f"  value_end = {value_end_including_flow} + ({flows_on_date})")
print(f"  value_end = {value_end_correct}")
print(f"\n  TWR = (value_end - value_start) / value_start")
print(f"  TWR = ({value_end_correct} - {value_start}) / {value_start}")
print(f"  TWR = {period_return_correct:.2%}")
print(f"\n  ✓ CORRECT: Shows {period_return_correct:.2%}")

print("\n" + "="*60)
print("ANOTHER EXAMPLE - SELL Transaction:")
print("="*60)

print("\nScenario:")
print("- Start: Own 10 shares @ $10/share = $100")
print("- Action: Sell 5 shares @ $10/share = $50")
print("- End: Own 5 shares @ $10/share = $50")
print("- Market movement: 0% (price stayed at $10)")
print("\nExpected TWR: 0% (no price change)")

value_start2 = 100.0
value_end_including_flow2 = 50.0

# SELL transactions have POSITIVE amount (cash inflow)
sell_amount = (5 * 10)  # +50 (5 shares @ $10, cash entering account)
flows_on_date2 = sell_amount  # +50

# Current code:
value_end_current2 = value_end_including_flow2 - flows_on_date2
period_return_current2 = (value_end_current2 - value_start2) / value_start2

print(f"\n❌ Current code (WRONG):")
print(f"  value_end = {value_end_including_flow2} - {flows_on_date2} = {value_end_current2}")
print(f"  TWR = ({value_end_current2} - {value_start2}) / {value_start2} = {period_return_current2:.2%}")

# Correct:
value_end_correct2 = value_end_including_flow2 + flows_on_date2
period_return_correct2 = (value_end_correct2 - value_start2) / value_start2

print(f"\n✓ Correct calculation:")
print(f"  value_end = {value_end_including_flow2} + {flows_on_date2} = {value_end_correct2}")
print(f"  TWR = ({value_end_correct2} - {value_start2}) / {value_start2} = {period_return_correct2:.2%}")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
print("\nThe bug is on lines 445 and 506 of performance_v2.py:")
print("\n  WRONG: value_end = value_end_before_flows - flows_on_date")
print("  RIGHT: value_end = value_end_before_flows + flows_on_date")
print("\nThis affects BOTH portfolio-level and security-level TWR calculations!")
print("="*60)
