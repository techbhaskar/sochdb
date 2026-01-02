// Copyright 2025 ToonDB Authors
//
// Licensed under the Apache License, Version 2.0

//! SQL Query Examples
//!
//! Demonstrates SQL support in ToonDB:
//! - CREATE TABLE, INSERT, UPDATE, DELETE
//! - SELECT with WHERE, ORDER BY, LIMIT
//! - Prepared statements
//! - Schema management

use toondb_client::{Client, ClientConfig, QueryResult};
use toondb_query::sql::{Parser, Statement, ExecutionResult};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=".repeat(60));
    println!("ToonDB SQL Query Examples");
    println!("=".repeat(60));

    // Open database
    let db_path = "./demo_sql_db_rust";
    println!("\n📂 Opening database: {}", db_path);
    
    // Clean up existing database
    let _ = std::fs::remove_dir_all(db_path);
    
    let mut client = Client::open(db_path)?;
    println!("✓ Database opened successfully");

    // Run demonstrations
    create_tables(&mut client)?;
    insert_data(&mut client)?;
    select_queries(&mut client)?;
    update_operations(&mut client)?;
    delete_operations(&mut client)?;
    complex_queries(&mut client)?;
    sql_parser_examples()?;

    println!("\n{}", "=".repeat(60));
    println!("✓ All SQL examples completed successfully!");
    println!("{}", "=".repeat(60));

    Ok(())
}

fn create_tables(client: &mut Client) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📝 Creating Tables with SQL");
    println!("{}", "=".repeat(60));

    // Create users table
    client.execute(
        r#"
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            age INTEGER,
            created_at TEXT
        )
        "#,
    )?;
    println!("✓ Created 'users' table");

    // Create posts table
    client.execute(
        r#"
        CREATE TABLE posts (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            title TEXT NOT NULL,
            content TEXT,
            likes INTEGER DEFAULT 0,
            published_at TEXT
        )
        "#,
    )?;
    println!("✓ Created 'posts' table");

    Ok(())
}

fn insert_data(client: &mut Client) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📥 Inserting Data with SQL");
    println!("{}", "=".repeat(60));

    // Insert users
    let users = vec![
        (1, "Alice", "alice@example.com", 30, "2024-01-01"),
        (2, "Bob", "bob@example.com", 25, "2024-01-02"),
        (3, "Charlie", "charlie@example.com", 35, "2024-01-03"),
        (4, "Diana", "diana@example.com", 28, "2024-01-04"),
    ];

    for (id, name, email, age, created_at) in users {
        client.execute(&format!(
            "INSERT INTO users (id, name, email, age, created_at) VALUES ({}, '{}', '{}', {}, '{}')",
            id, name, email, age, created_at
        ))?;
        println!("  ✓ Inserted user: {}", name);
    }

    // Insert posts
    let posts = vec![
        (1, 1, "First Post", "Hello World!", 10, "2024-01-05"),
        (2, 1, "Second Post", "ToonDB is awesome", 25, "2024-01-06"),
        (3, 2, "Bob's Thoughts", "SQL queries are easy", 15, "2024-01-07"),
        (4, 3, "Charlie's Guide", "Database tips", 30, "2024-01-08"),
        (5, 3, "Advanced Topics", "Performance tuning", 50, "2024-01-09"),
    ];

    for (id, user_id, title, content, likes, published_at) in posts {
        client.execute(&format!(
            "INSERT INTO posts (id, user_id, title, content, likes, published_at) VALUES ({}, {}, '{}', '{}', {}, '{}')",
            id, user_id, title, content, likes, published_at
        ))?;
        println!("  ✓ Inserted post: {}", title);
    }

    Ok(())
}

fn select_queries(client: &mut Client) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Running SELECT Queries");
    println!("{}", "=".repeat(60));

    // Simple SELECT
    println!("\n1. Select all users:");
    let result = client.execute("SELECT * FROM users")?;
    println!("   Found {} users", result.rows.len());
    for row in &result.rows {
        println!("   - {} ({})", 
                 row.get("name").map(|v| format!("{:?}", v)).unwrap_or_default(),
                 row.get("email").map(|v| format!("{:?}", v)).unwrap_or_default());
    }

    // SELECT with WHERE clause
    println!("\n2. Users older than 28:");
    let result = client.execute("SELECT name, age FROM users WHERE age > 28")?;
    for row in &result.rows {
        println!("   - {}: {} years old",
                 row.get("name").map(|v| format!("{:?}", v)).unwrap_or_default(),
                 row.get("age").map(|v| format!("{:?}", v)).unwrap_or_default());
    }

    // SELECT with ORDER BY
    println!("\n3. Posts ordered by likes (descending):");
    let result = client.execute("SELECT title, likes FROM posts ORDER BY likes DESC")?;
    for row in &result.rows {
        println!("   - {}: {} likes",
                 row.get("title").map(|v| format!("{:?}", v)).unwrap_or_default(),
                 row.get("likes").map(|v| format!("{:?}", v)).unwrap_or_default());
    }

    // SELECT with LIMIT
    println!("\n4. Top 3 most liked posts:");
    let result = client.execute("SELECT title, likes FROM posts ORDER BY likes DESC LIMIT 3")?;
    for row in &result.rows {
        println!("   - {}: {} likes",
                 row.get("title").map(|v| format!("{:?}", v)).unwrap_or_default(),
                 row.get("likes").map(|v| format!("{:?}", v)).unwrap_or_default());
    }

    Ok(())
}

fn update_operations(client: &mut Client) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n✏️  UPDATE Operations");
    println!("{}", "=".repeat(60));

    // Update single row
    println!("\n1. Update Alice's age:");
    client.execute("UPDATE users SET age = 31 WHERE name = 'Alice'")?;
    let result = client.execute("SELECT name, age FROM users WHERE name = 'Alice'")?;
    if let Some(row) = result.rows.first() {
        println!("   Alice's new age: {:?}", row.get("age"));
    }

    // Update multiple rows
    println!("\n2. Increment likes on all posts by user_id = 1:");
    client.execute("UPDATE posts SET likes = likes + 5 WHERE user_id = 1")?;
    let result = client.execute("SELECT title, likes FROM posts WHERE user_id = 1")?;
    for row in &result.rows {
        println!("   - {}: {} likes",
                 row.get("title").map(|v| format!("{:?}", v)).unwrap_or_default(),
                 row.get("likes").map(|v| format!("{:?}", v)).unwrap_or_default());
    }

    Ok(())
}

fn delete_operations(client: &mut Client) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🗑️  DELETE Operations");
    println!("{}", "=".repeat(60));

    // Count before delete
    let result = client.execute("SELECT COUNT(*) as total FROM posts")?;
    let before_count = result.rows.len();
    println!("Posts before delete: {}", before_count);

    // Delete specific post
    client.execute("DELETE FROM posts WHERE id = 5")?;
    println!("✓ Deleted post with id = 5");

    // Count after delete
    let result = client.execute("SELECT COUNT(*) as total FROM posts")?;
    let after_count = result.rows.len();
    println!("Posts after delete: {}", after_count);

    Ok(())
}

fn complex_queries(client: &mut Client) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 Complex Queries");
    println!("{}", "=".repeat(60));

    // SELECT with multiple conditions
    println!("\n1. Users aged 25-30:");
    let result = client.execute(
        "SELECT name, age, email FROM users WHERE age >= 25 AND age <= 30 ORDER BY age"
    )?;
    for row in &result.rows {
        println!("   - {}: {} years ({})",
                 row.get("name").map(|v| format!("{:?}", v)).unwrap_or_default(),
                 row.get("age").map(|v| format!("{:?}", v)).unwrap_or_default(),
                 row.get("email").map(|v| format!("{:?}", v)).unwrap_or_default());
    }

    // SELECT with pattern matching
    println!("\n2. Posts with 'Post' in title:");
    let result = client.execute("SELECT title, likes FROM posts WHERE title LIKE '%Post%'")?;
    for row in &result.rows {
        println!("   - {}: {} likes",
                 row.get("title").map(|v| format!("{:?}", v)).unwrap_or_default(),
                 row.get("likes").map(|v| format!("{:?}", v)).unwrap_or_default());
    }

    Ok(())
}

fn sql_parser_examples() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔧 SQL Parser Examples");
    println!("{}", "=".repeat(60));

    // Parse a SELECT statement
    let sql = "SELECT id, name FROM users WHERE age > 25 ORDER BY name LIMIT 10";
    match Parser::parse(sql) {
        Ok(stmt) => {
            println!("\n✓ Successfully parsed SELECT:");
            println!("  SQL: {}", sql);
            println!("  AST: {:?}", stmt);
        }
        Err(errors) => {
            println!("✗ Parse errors: {:?}", errors);
        }
    }

    // Parse an INSERT statement
    let sql = "INSERT INTO users (id, name, email) VALUES (1, 'Alice', 'alice@example.com')";
    match Parser::parse(sql) {
        Ok(stmt) => {
            println!("\n✓ Successfully parsed INSERT:");
            println!("  SQL: {}", sql);
        }
        Err(errors) => {
            println!("✗ Parse errors: {:?}", errors);
        }
    }

    // Parse a CREATE TABLE statement
    let sql = "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT NOT NULL, price REAL)";
    match Parser::parse(sql) {
        Ok(stmt) => {
            println!("\n✓ Successfully parsed CREATE TABLE:");
            println!("  SQL: {}", sql);
        }
        Err(errors) => {
            println!("✗ Parse errors: {:?}", errors);
        }
    }

    Ok(())
}
