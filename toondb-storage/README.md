# toondb-storage

High-performance storage engine for ToonDB with ACID transactions and crash recovery.

## Overview

`toondb-storage` provides the persistence layer for ToonDB, featuring:

- **Write-Ahead Logging (WAL)**: Durable writes with crash recovery
- **MVCC Transactions**: Snapshot isolation for concurrent access
- **Columnar Storage**: 80% I/O reduction via projection pushdown
- **Memory-Mapped I/O**: Zero-copy reads for maximum throughput

## Features

- **ACID Transactions**: Full transactional guarantees with rollback support
- **Crash Recovery**: Automatic recovery from WAL after unexpected shutdown
- **Batch Writes**: Group commit for high write throughput
- **Compaction**: Background compaction to reclaim space
- **FFI Layer**: C-compatible FFI for Python/Node.js bindings

## Installation

```toml
[dependencies]
toondb-storage = "0.2.5"
```

## Usage

Most users should use the high-level [`toondb`](https://crates.io/crates/toondb) crate:

```rust
use toondb::Database;

let db = Database::open("./my_data")?;

// Transactional writes
let txn = db.transaction()?;
txn.put("users/1", b"Alice")?;
txn.commit()?;
```

## Architecture

```
┌─────────────────────────────────────────┐
│           toondb-storage                │
├─────────────────────────────────────────┤
│  Transaction Layer (MVCC, Isolation)    │
├─────────────────────────────────────────┤
│  WAL (Write-Ahead Log, Durability)      │
├─────────────────────────────────────────┤
│  Page Cache (Buffer Pool, LRU)          │
├─────────────────────────────────────────┤
│  Storage Engine (LSM-tree, Compaction)  │
└─────────────────────────────────────────┘
```

## Performance

| Operation | Throughput |
|-----------|------------|
| Point reads | ~500K ops/sec |
| Sequential writes | ~200K ops/sec |
| Batch writes (1000) | ~2M ops/sec |
| Recovery time | <100ms for 1GB WAL |

## Crate Structure

| Crate | Purpose |
|-------|---------|
| [`toondb`](https://crates.io/crates/toondb) | High-level client API (start here) |
| [`toondb-core`](https://crates.io/crates/toondb-core) | Core types and traits |
| `toondb-storage` | Storage engine (this crate) |
| [`toondb-index`](https://crates.io/crates/toondb-index) | HNSW vector indexing |
| [`toondb-query`](https://crates.io/crates/toondb-query) | Query planning and execution |

## License

Apache-2.0 - see [LICENSE](../LICENSE) for details.
