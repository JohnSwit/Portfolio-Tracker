import sqlite3
import json
from datetime import datetime
from typing import List, Optional
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

DATABASE_PATH = "portfolio.db"


@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_database():
    """Initialize database tables"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Transactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id TEXT NOT NULL,
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                transaction_type TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                amount REAL NOT NULL,
                fees REAL DEFAULT 0.0,
                notes TEXT,
                is_synthetic INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Portfolios table (for custom portfolio data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id TEXT UNIQUE NOT NULL,
                account_name TEXT,
                cash_balance REAL DEFAULT 0.0,
                inception_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_account
            ON transactions(account_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_date
            ON transactions(date)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_symbol
            ON transactions(symbol)
        """)

        conn.commit()
        logger.info("Database initialized successfully")


def migrate_database():
    """Apply database migrations for existing databases"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Check if is_synthetic column exists
        cursor.execute("PRAGMA table_info(transactions)")
        columns = [col[1] for col in cursor.fetchall()]

        if 'is_synthetic' not in columns:
            logger.info("Adding is_synthetic column to transactions table...")
            cursor.execute("""
                ALTER TABLE transactions
                ADD COLUMN is_synthetic INTEGER DEFAULT 0
            """)

            # Backfill: mark existing CASH transactions as synthetic if they match the pattern
            cursor.execute("""
                UPDATE transactions
                SET is_synthetic = 1
                WHERE symbol = 'CASH'
                  AND (notes LIKE 'Auto-%' OR notes LIKE 'auto-%')
            """)

            rows_updated = cursor.rowcount
            logger.info(f"Marked {rows_updated} existing CASH transactions as synthetic")

        conn.commit()
        logger.info("Database migration completed successfully")


def save_transactions(account_id: str, transactions: List[dict]) -> int:
    """Save transactions to database"""
    with get_db() as conn:
        cursor = conn.cursor()

        inserted = 0
        for txn in transactions:
            cursor.execute("""
                INSERT INTO transactions
                (account_id, date, symbol, transaction_type, quantity, price, amount, fees, notes, is_synthetic)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                account_id,
                txn['date'],
                txn['symbol'],
                txn['transaction_type'],
                txn['quantity'],
                txn['price'],
                txn['amount'],
                txn.get('fees', 0.0),
                txn.get('notes', ''),
                1 if txn.get('is_synthetic', False) else 0
            ))
            inserted += 1

        conn.commit()
        logger.info(f"Saved {inserted} transactions for account {account_id}")
        return inserted


def get_transactions(account_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[dict]:
    """Get transactions from database"""
    with get_db() as conn:
        cursor = conn.cursor()

        query = "SELECT * FROM transactions WHERE account_id = ?"
        params = [account_id]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date ASC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        transactions = []
        for row in rows:
            transactions.append({
                'id': row['id'],
                'date': row['date'],
                'symbol': row['symbol'],
                'transaction_type': row['transaction_type'],
                'quantity': row['quantity'],
                'price': row['price'],
                'amount': row['amount'],
                'fees': row['fees'],
                'notes': row['notes'],
                'is_synthetic': row['is_synthetic'] if 'is_synthetic' in row.keys() else 0
            })

        return transactions


def delete_all_transactions(account_id: str) -> int:
    """Delete all transactions for an account"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM transactions WHERE account_id = ?", (account_id,))
        deleted = cursor.rowcount
        conn.commit()
        logger.info(f"Deleted {deleted} transactions for account {account_id}")
        return deleted


def get_or_create_portfolio(account_id: str, account_name: str = None) -> dict:
    """Get or create portfolio record"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Try to get existing
        cursor.execute("SELECT * FROM portfolios WHERE account_id = ?", (account_id,))
        row = cursor.fetchone()

        if row:
            return {
                'account_id': row['account_id'],
                'account_name': row['account_name'],
                'cash_balance': row['cash_balance'],
                'inception_date': row['inception_date']
            }

        # Create new
        cursor.execute("""
            INSERT INTO portfolios (account_id, account_name, cash_balance, inception_date)
            VALUES (?, ?, ?, ?)
        """, (account_id, account_name or f"Portfolio {account_id}", 0.0, datetime.now().isoformat()))

        conn.commit()

        return {
            'account_id': account_id,
            'account_name': account_name or f"Portfolio {account_id}",
            'cash_balance': 0.0,
            'inception_date': datetime.now().isoformat()
        }


def update_portfolio_cash(account_id: str, cash_balance: float):
    """Update portfolio cash balance"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE portfolios
            SET cash_balance = ?, updated_at = CURRENT_TIMESTAMP
            WHERE account_id = ?
        """, (cash_balance, account_id))
        conn.commit()


# Initialize database on module load
init_database()
