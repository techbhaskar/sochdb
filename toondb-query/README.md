# toondb-query

Query planning and execution engine for ToonDB.

## Overview

`toondb-query` provides the query layer for ToonDB, featuring:

- **Query Builder**: Fluent API for constructing queries
- **Query Optimizer**: Cost-based optimization for efficient execution
- **Projection Pushdown**: Only read columns you need
- **Filter Pushdown**: Apply filters at storage layer
- **TOON Output**: Token-optimized format (40-66% fewer tokens than JSON)

## Features

- **SQL-like Queries**: Familiar query patterns without SQL parsing overhead
- **Path-based Access**: O(|path|) resolution independent of data size
- **Aggregations**: COUNT, SUM, AVG, MIN, MAX with efficient execution
- **Sorting & Pagination**: ORDER BY, LIMIT, OFFSET support
- **Join Support**: Nested loop and hash joins

## Installation

```toml
[dependencies]
toondb-query = "0.2.5"
```

## Usage

Most users should use the high-level [`toondb`](https://crates.io/crates/toondb) crate:

```rust
use toondb::{Database, Query};

let db = Database::open("./my_data")?;

// Query with builder pattern
let results = db.query("users")
    .filter("age", ">", 21)
    .select(&["name", "email"])
    .order_by("name")
    .limit(10)
    .execute()?;

// Results in token-efficient TOON format
println!("{}", results.to_toon()); 
// users[3]{name,email}: Alice,alice@...|Bob,bob@...|Carol,carol@...
```

## Query Execution Pipeline

```
┌──────────────┐
│    Query     │  User query (builder or path)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Parse     │  Validate and normalize
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Optimize   │  Pushdown filters, projections
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Execute    │  Scan, filter, project, sort
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Format     │  TOON or JSON output
└──────────────┘
```

## Crate Structure

| Crate | Purpose |
|-------|---------|
| [`toondb`](https://crates.io/crates/toondb) | High-level client API (start here) |
| [`toondb-core`](https://crates.io/crates/toondb-core) | Core types and traits |
| [`toondb-storage`](https://crates.io/crates/toondb-storage) | Storage engine with WAL |
| [`toondb-index`](https://crates.io/crates/toondb-index) | HNSW vector indexing |
| `toondb-query` | Query execution (this crate) |

## License

Apache-2.0 - see [LICENSE](../LICENSE) for details.
