"""
Script to apply analytics database schema

Run this to add analytics tables to your database:
    python apply_analytics_schema.py
"""

import sqlite3
import os

def apply_analytics_schema():
    """Apply the analytics schema to the database"""

    # Database path
    db_path = os.path.join(os.path.dirname(__file__), '..', 'portfolios.db')

    # Schema file path
    schema_path = os.path.join(os.path.dirname(__file__), 'database_analytics_schema.sql')

    # Read schema
    print(f"Reading schema from: {schema_path}")
    with open(schema_path, 'r') as f:
        schema_sql = f.read()

    # Apply schema
    print(f"Applying schema to: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.executescript(schema_sql)
        conn.commit()
        print("✅ Analytics schema applied successfully!")

        # Verify tables were created
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table'
            AND name IN (
                'portfolio_returns',
                'position_returns',
                'benchmark_returns',
                'factor_returns',
                'volume_data',
                'portfolio_snapshots',
                'analytics_cache'
            )
            ORDER BY name
        """)

        tables = cursor.fetchall()
        print(f"\nCreated {len(tables)} analytics tables:")
        for table in tables:
            print(f"  - {table[0]}")

    except Exception as e:
        print(f"❌ Error applying schema: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    apply_analytics_schema()
