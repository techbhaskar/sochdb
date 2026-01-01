# toondb-core

Core types and traits for ToonDB - the LLM-native database.

## Overview

`toondb-core` provides the foundational types, error handling, and trait definitions used throughout the ToonDB ecosystem. This crate is typically not used directly - instead, use the [`toondb`](https://crates.io/crates/toondb) client crate.

## Features

- **Core Types**: Value types, keys, and serialization primitives
- **Error Handling**: Unified error types across all ToonDB crates
- **Trait Definitions**: Common traits for storage backends and query execution
- **TOON Format**: Token-optimized notation for LLM efficiency (40-66% fewer tokens than JSON)

## Installation

```toml
[dependencies]
toondb-core = "0.2.5"
```

## Usage

Most users should use the high-level [`toondb`](https://crates.io/crates/toondb) crate instead:

```rust
use toondb::Database;

let db = Database::open("./my_data")?;
```

## Crate Structure

ToonDB is organized into several crates:

| Crate | Purpose |
|-------|---------|
| [`toondb`](https://crates.io/crates/toondb) | High-level client API (start here) |
| `toondb-core` | Core types and traits (this crate) |
| [`toondb-storage`](https://crates.io/crates/toondb-storage) | Storage engine with WAL |
| [`toondb-index`](https://crates.io/crates/toondb-index) | HNSW vector indexing |
| [`toondb-query`](https://crates.io/crates/toondb-query) | Query planning and execution |

## License

Apache-2.0 - see [LICENSE](../LICENSE) for details.
