//! SochDB-based catalog for segment metadata.
//!
//! Uses SochDB storage for unified transaction semantics instead of SQLite.
//! Vector data is stored in mmap-able segment files.

mod sochdb_catalog;

pub use sochdb_catalog::{Catalog, CollectionInfo, SegmentInfo};
