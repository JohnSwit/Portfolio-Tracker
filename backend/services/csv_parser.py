import csv
import io
from datetime import datetime
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class CSVParser:
    """Parse and validate CSV transaction files"""

    @staticmethod
    def parse_transactions_csv(file_content: str) -> List[Dict]:
        """
        Parse CSV file content into transactions

        Expected CSV format:
        date,symbol,type,quantity,price,fees,notes

        Example:
        2024-01-15,AAPL,buy,100,150.00,9.99,Initial purchase
        2024-02-20,MSFT,buy,50,380.00,9.99,
        2024-03-10,AAPL,sell,25,170.00,9.99,Partial sale

        Supported transaction types: buy, sell, dividend, split
        """
        transactions = []
        reader = csv.DictReader(io.StringIO(file_content))

        required_fields = ['date', 'symbol', 'type', 'quantity', 'price']

        for row_num, row in enumerate(reader, start=2):  # Start at 2 (1 is header)
            try:
                # Validate required fields
                missing_fields = [field for field in required_fields if not row.get(field)]
                if missing_fields:
                    raise ValueError(f"Row {row_num}: Missing required fields: {', '.join(missing_fields)}")

                # Parse date
                date_str = row['date'].strip()
                try:
                    # Try multiple date formats
                    for date_format in ['%Y-%m-%d', '%m/%d/%Y', '%m/%d/%y', '%Y/%m/%d']:
                        try:
                            date = datetime.strptime(date_str, date_format)
                            break
                        except ValueError:
                            continue
                    else:
                        raise ValueError(f"Invalid date format. Use YYYY-MM-DD or MM/DD/YYYY")
                except ValueError as e:
                    raise ValueError(f"Row {row_num}: {str(e)}")

                # Parse transaction type
                txn_type = row['type'].strip().lower()
                valid_types = ['buy', 'sell', 'dividend', 'split', 'deposit', 'withdrawal']
                if txn_type not in valid_types:
                    raise ValueError(f"Row {row_num}: Invalid transaction type '{txn_type}'. Must be one of: {', '.join(valid_types)}")

                # Parse numeric fields
                try:
                    quantity = float(row['quantity'].strip())
                    price = float(row['price'].strip())
                    fees = float(row.get('fees', '0').strip() or '0')
                except ValueError as e:
                    raise ValueError(f"Row {row_num}: Invalid number format - {str(e)}")

                # Validate positive values
                if quantity <= 0:
                    raise ValueError(f"Row {row_num}: Quantity must be positive")
                if price < 0:
                    raise ValueError(f"Row {row_num}: Price cannot be negative")
                if fees < 0:
                    raise ValueError(f"Row {row_num}: Fees cannot be negative")

                # Calculate amount
                if txn_type == 'buy':
                    amount = -(quantity * price + fees)  # Negative for cash outflow
                elif txn_type == 'sell':
                    amount = quantity * price - fees  # Positive for cash inflow
                elif txn_type == 'dividend':
                    amount = quantity * price  # Dividend amount
                elif txn_type == 'deposit':
                    amount = quantity * price  # Cash deposit
                elif txn_type == 'withdrawal':
                    amount = -(quantity * price)  # Cash withdrawal
                else:
                    amount = 0.0

                transaction = {
                    'date': date.isoformat(),
                    'symbol': row['symbol'].strip().upper(),
                    'transaction_type': txn_type,
                    'quantity': quantity,
                    'price': price,
                    'amount': amount,
                    'fees': fees,
                    'notes': row.get('notes', '').strip()
                }

                transactions.append(transaction)

            except Exception as e:
                logger.error(f"Error parsing CSV row {row_num}: {e}")
                raise ValueError(f"Row {row_num}: {str(e)}")

        if not transactions:
            raise ValueError("No valid transactions found in CSV file")

        logger.info(f"Successfully parsed {len(transactions)} transactions from CSV")
        return transactions

    @staticmethod
    def generate_template_csv() -> str:
        """Generate a CSV template file"""
        template = """date,symbol,type,quantity,price,fees,notes
2024-01-15,AAPL,buy,100,150.00,9.99,Initial purchase
2024-02-20,MSFT,buy,50,380.00,9.99,Technology position
2024-03-10,AAPL,sell,25,170.00,9.99,Partial sale
2024-04-01,AAPL,dividend,0,1.24,0.00,Quarterly dividend
"""
        return template

    @staticmethod
    def validate_csv_format(file_content: str) -> Dict[str, any]:
        """Validate CSV format and return validation results"""
        try:
            # Try to parse
            transactions = CSVParser.parse_transactions_csv(file_content)

            return {
                'valid': True,
                'transaction_count': len(transactions),
                'date_range': {
                    'start': min(t['date'] for t in transactions),
                    'end': max(t['date'] for t in transactions)
                },
                'symbols': list(set(t['symbol'] for t in transactions)),
                'errors': []
            }

        except Exception as e:
            return {
                'valid': False,
                'transaction_count': 0,
                'errors': [str(e)]
            }


csv_parser = CSVParser()
