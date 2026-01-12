#!/usr/bin/env python3
"""
SochDB SQL Query Examples

Demonstrates SQL support in SochDB:
- CREATE TABLE, INSERT, UPDATE, DELETE
- SELECT with WHERE, ORDER BY, LIMIT
- Transactions with SQL
- Schema management
"""

import sys
import os
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sochdb-python', 'src'))

from sochdb import Database


def create_tables(db: Database) -> None:
    """Create sample tables using SQL."""
    print("\n📝 Creating Tables with SQL")
    print("=" * 60)
    
    # Create users table
    result = db.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            age INTEGER,
            created_at TEXT
        )
    """)
    print("✓ Created 'users' table")
    
    # Create posts table
    result = db.execute("""
        CREATE TABLE posts (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            title TEXT NOT NULL,
            content TEXT,
            likes INTEGER DEFAULT 0,
            published_at TEXT
        )
    """)
    print("✓ Created 'posts' table")


def insert_data(db: Database) -> None:
    """Insert data using SQL."""
    print("\n📥 Inserting Data with SQL")
    print("=" * 60)
    
    # Insert users
    users = [
        (1, 'Alice', 'alice@example.com', 30, '2024-01-01'),
        (2, 'Bob', 'bob@example.com', 25, '2024-01-02'),
        (3, 'Charlie', 'charlie@example.com', 35, '2024-01-03'),
        (4, 'Diana', 'diana@example.com', 28, '2024-01-04'),
    ]
    
    for user_id, name, email, age, created_at in users:
        db.execute(f"""
            INSERT INTO users (id, name, email, age, created_at)
            VALUES ({user_id}, '{name}', '{email}', {age}, '{created_at}')
        """)
        print(f"  ✓ Inserted user: {name}")
    
    # Insert posts
    posts = [
        (1, 1, 'First Post', 'Hello World!', 10, '2024-01-05'),
        (2, 1, 'Second Post', 'SochDB is awesome', 25, '2024-01-06'),
        (3, 2, 'Bob\'s Thoughts', 'SQL queries are easy', 15, '2024-01-07'),
        (4, 3, 'Charlie\'s Guide', 'Database tips', 30, '2024-01-08'),
        (5, 3, 'Advanced Topics', 'Performance tuning', 50, '2024-01-09'),
    ]
    
    for post_id, user_id, title, content, likes, published_at in posts:
        db.execute(f"""
            INSERT INTO posts (id, user_id, title, content, likes, published_at)
            VALUES ({post_id}, {user_id}, '{title}', '{content}', {likes}, '{published_at}')
        """)
        print(f"  ✓ Inserted post: {title}")


def select_queries(db: Database) -> None:
    """Run various SELECT queries."""
    print("\n🔍 Running SELECT Queries")
    print("=" * 60)
    
    # Simple SELECT
    print("\n1. Select all users:")
    result = db.execute("SELECT * FROM users")
    print(f"   Found {len(result.rows)} users")
    for row in result.rows:
        print(f"   - {row['name']} ({row['email']})")
    
    # SELECT with WHERE clause
    print("\n2. Users older than 28:")
    result = db.execute("SELECT name, age FROM users WHERE age > 28")
    for row in result.rows:
        print(f"   - {row['name']}: {row['age']} years old")
    
    # SELECT with ORDER BY
    print("\n3. Posts ordered by likes (descending):")
    result = db.execute("""
        SELECT title, likes FROM posts 
        ORDER BY likes DESC
    """)
    for row in result.rows:
        print(f"   - {row['title']}: {row['likes']} likes")
    
    # SELECT with LIMIT
    print("\n4. Top 3 most liked posts:")
    result = db.execute("""
        SELECT title, likes FROM posts 
        ORDER BY likes DESC 
        LIMIT 3
    """)
    for row in result.rows:
        print(f"   - {row['title']}: {row['likes']} likes")
    
    # Aggregate functions (if supported)
    print("\n5. Count total posts:")
    result = db.execute("SELECT COUNT(*) as total FROM posts")
    if result.rows:
        print(f"   Total posts: {result.rows[0].get('total', len(result.rows))}")
    else:
        print(f"   Total posts: 0 (SQL not fully implemented yet)")


def update_operations(db: Database) -> None:
    """Update data using SQL."""
    print("\n✏️  UPDATE Operations")
    print("=" * 60)
    
    # Update single row
    print("\n1. Update Alice's age:")
    db.execute("UPDATE users SET age = 31 WHERE name = 'Alice'")
    result = db.execute("SELECT name, age FROM users WHERE name = 'Alice'")
    if result.rows:
        print(f"   Alice's new age: {result.rows[0]['age']}")
    else:
        print("   (No data - SQL stub implementation)")
    
    # Update multiple rows
    print("\n2. Increment likes on all posts by user_id = 1:")
    db.execute("UPDATE posts SET likes = likes + 5 WHERE user_id = 1")
    result = db.execute("SELECT title, likes FROM posts WHERE user_id = 1")
    if result.rows:
        for row in result.rows:
            print(f"   - {row['title']}: {row['likes']} likes")
    else:
        print("   (No data - SQL stub implementation)")


def delete_operations(db: Database) -> None:
    """Delete data using SQL."""
    print("\n🗑️  DELETE Operations")
    print("=" * 60)
    
    # Count before delete
    result = db.execute("SELECT COUNT(*) as total FROM posts")
    before_count = len(result.rows)
    print(f"Posts before delete: {before_count}")
    
    # Delete specific post
    db.execute("DELETE FROM posts WHERE id = 5")
    print("✓ Deleted post with id = 5")
    
    # Count after delete
    result = db.execute("SELECT COUNT(*) as total FROM posts")
    after_count = len(result.rows)
    print(f"Posts after delete: {after_count}")


def transactions_with_sql(db: Database) -> None:
    """Use SQL within transactions."""
    print("\n💳 SQL in Transactions")
    print("=" * 60)
    
    try:
        # Start transaction
        txn = db.transaction()
        
        # Execute SQL within transaction (note: stub implementation)
        db.execute("INSERT INTO users (id, name, email, age) VALUES (5, 'Eve', 'eve@example.com', 26)")
        db.execute("INSERT INTO posts (id, user_id, title, content) VALUES (6, 5, 'Eve Post', 'My first post')")
        
        # Commit transaction
        txn.commit()
        print("✓ Transaction committed successfully")
        print("  (Note: SQL stub - actual data not persisted)")
        
        # Verify data
        result = db.execute("SELECT name FROM users WHERE id = 5")
        if result.rows:
            print(f"  New user: {result.rows[0]['name']}")
        
    except Exception as e:
        print(f"✗ Transaction failed: {e}")


def complex_queries(db: Database) -> None:
    """More complex SQL queries."""
    print("\n🎯 Complex Queries")
    print("=" * 60)
    
    # SELECT with multiple conditions
    print("\n1. Users aged 25-30:")
    result = db.execute("""
        SELECT name, age, email 
        FROM users 
        WHERE age >= 25 AND age <= 30
        ORDER BY age
    """)
    for row in result.rows:
        print(f"   - {row['name']}: {row['age']} years ({row['email']})")
    
    # SELECT with LIKE (if supported)
    print("\n2. Posts with 'Post' in title:")
    result = db.execute("""
        SELECT title, likes 
        FROM posts 
        WHERE title LIKE '%Post%'
    """)
    for row in result.rows:
        print(f"   - {row['title']}: {row['likes']} likes")


def main():
    """Main demonstration."""
    print("=" * 60)
    print("SochDB SQL Query Examples")
    print("=" * 60)
    print("\n⚠️  NOTE: SQL support is currently a stub implementation.")
    print("Full SQL functionality requires backend query engine integration.")
    print("These examples demonstrate the API, but return empty results.\n")
    
    # Open database
    db_path = "./demo_sql_db"
    print(f"\n📂 Opening database: {db_path}")
    
    # Clean up existing database
    import shutil
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    
    db = Database.open(db_path)
    print("✓ Database opened successfully")
    
    try:
        # Run demonstrations
        create_tables(db)
        insert_data(db)
        select_queries(db)
        update_operations(db)
        delete_operations(db)
        transactions_with_sql(db)
        complex_queries(db)
        
        print("\n" + "=" * 60)
        print("✓ All SQL examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close database
        db.close()
        print("\n✓ Database closed")


if __name__ == "__main__":
    main()
