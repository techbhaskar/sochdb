// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! SochDB IPC Server
//!
//! Standalone server process that provides IPC access to a SochDB database.
//! This is used by the JavaScript and Python SDKs in embedded mode.
//!
//! # Usage
//!
//! ```bash
//! # Start server for a database directory
//! sochdb-server --db ./my_database
//!
//! # Specify custom socket path
//! sochdb-server --db ./my_database --socket /tmp/custom.sock
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use tracing::{Level, info, error};
use tracing_subscriber::FmtSubscriber;

use sochdb_storage::database::Database;
use sochdb_storage::ipc_server::{IpcServer, IpcServerConfig};

/// SochDB IPC Server - provides multi-process access to SochDB databases
#[derive(Parser, Debug)]
#[command(name = "sochdb-server")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the database directory
    #[arg(short, long, default_value = "./sochdb_data")]
    db: PathBuf,

    /// Path to the Unix socket (default: <db>/sochdb.sock)
    #[arg(short, long)]
    socket: Option<PathBuf>,

    /// Maximum number of client connections
    #[arg(long, default_value = "100")]
    max_clients: usize,

    /// Connection timeout in milliseconds
    #[arg(long, default_value = "30000")]
    timeout_ms: u64,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    log_level: String,
}

fn main() {
    let args = Args::parse();

    // Initialize tracing
    let level = match args.log_level.to_lowercase().as_str() {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("setting default subscriber");

    // Determine socket path
    let socket_path = args.socket.unwrap_or_else(|| {
        args.db.join("sochdb.sock")
    });

    // Ensure database directory exists
    if !args.db.exists() {
        std::fs::create_dir_all(&args.db).expect("Failed to create database directory");
    }

    info!(
        "Starting SochDB server: db={:?}, socket={:?}",
        args.db, socket_path
    );

    // Open database
    let db = match Database::open(&args.db) {
        Ok(db) => Arc::new(db),
        Err(e) => {
            error!("Failed to open database: {}", e);
            std::process::exit(1);
        }
    };

    // Configure and start IPC server
    let config = IpcServerConfig {
        socket_path: socket_path.clone(),
        max_connections: args.max_clients,
        ..Default::default()
    };

    let server = IpcServer::new(Arc::clone(&db), config);

    // Handle shutdown signals
    ctrlc::set_handler(move || {
        info!("Received shutdown signal, cleaning up...");
        // Clean up socket file
        if socket_path.exists() {
            let _ = std::fs::remove_file(&socket_path);
        }
        std::process::exit(0);
    }).expect("Error setting Ctrl-C handler");

    // Start the server (blocks) - use run() not start() which is non-blocking
    info!("SochDB server ready, accepting connections");
    if let Err(e) = server.run() {
        error!("Server error: {}", e);
        std::process::exit(1);
    }
}
