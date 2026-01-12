# How to Integrate with MCP (Model Context Protocol)

> Use SochDB as persistent memory for Claude, Cursor, and other MCP clients.

---

## Problem

You want to give your LLM agent (Claude Desktop, Cursor, Goose, etc.) persistent memory using SochDB.

---

## Solution

### 1. Install SochDB MCP Server

```bash
# Build from source
cd sochdb
cargo build --release -p sochdb-mcp

# Or install binary
cp target/release/sochdb-mcp /usr/local/bin/
```

### 2. Configure Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or equivalent:

```json
{
  "mcpServers": {
    "sochdb": {
      "command": "sochdb-mcp",
      "args": ["--db", "~/.sochdb/claude_memory"],
      "env": {
        "RUST_LOG": "sochdb=info"
      }
    }
  }
}
```

### 3. Configure Cursor

Add to Cursor MCP settings:

```json
{
  "mcpServers": {
    "sochdb": {
      "command": "/usr/local/bin/sochdb-mcp",
      "args": ["--db", "~/.sochdb/cursor_memory"]
    }
  }
}
```

### 4. Configure Goose

Add to `~/.config/goose/mcp.json`:

```json
{
  "servers": {
    "sochdb": {
      "command": "sochdb-mcp",
      "args": ["--db", "~/.sochdb/goose_memory"]
    }
  }
}
```

---

## Available MCP Tools

Once configured, your LLM has access to these tools:

### `sochdb_put`

Store a value at a path.

```
Tool: sochdb_put
Arguments:
  path: "users/alice/preferences"
  value: '{"theme": "dark", "language": "en"}'
```

### `sochdb_get`

Retrieve a value by path.

```
Tool: sochdb_get
Arguments:
  path: "users/alice/preferences"
```

### `sochdb_delete`

Delete a key.

```
Tool: sochdb_delete
Arguments:
  path: "users/alice/old_data"
```

### `sochdb_query`

Execute a SochQL query. Returns results in TOON format.

```
Tool: sochdb_query
Arguments:
  query: "SELECT id,name FROM users WHERE score > 80"
  format: "toon"
  limit: 100
```

### `sochdb_context_query`

Assemble LLM context with token budgeting.

```
Tool: sochdb_context_query
Arguments:
  sections:
    - name: "user_prefs"
      kind: "get"
      path: "users/alice/preferences"
      priority: 1
    - name: "history"
      kind: "last"
      table: "messages"
      top_k: 10
      priority: 2
  token_budget: 4096
  format: "toon"
  truncation: "tail_drop"
```

### `sochdb_list_tables`

List all tables with semantic metadata.

```
Tool: sochdb_list_tables
Arguments:
  include_metadata: true
```

### `sochdb_describe`

Get detailed schema for a table.

```
Tool: sochdb_describe
Arguments:
  table: "users"
```

### Memory Tools

The MCP server also provides specialized memory tools:

- `memory_search_episodes` - Search for similar past episodes by semantic similarity
- `memory_get_episode_timeline` - Get the event timeline for an episode

---

## Example: Agent with Persistent Memory

Here's how an agent might use SochDB for memory:

### Storing Conversation

```
User: Remember that I prefer dark mode

Claude thinks: I should store this preference
Claude uses tool: sochdb_put
  path: "user/preferences/theme"
  value: "dark"

Claude: I've noted your preference for dark mode. I'll remember this for future sessions.
```

### Recalling Information

```
User: What theme do I prefer?

Claude thinks: Let me check stored preferences
Claude uses tool: sochdb_get
  path: "user/preferences/theme"
  result: "dark"

Claude: You prefer dark mode.
```

### Building Context

```
User: Summarize what we discussed yesterday

Claude thinks: I need to retrieve conversation history
Claude uses tool: sochdb_context_query
  sections:
    - name: "history"
      kind: "last"
      table: "conversations"
      top_k: 20
      priority: 1
  token_budget: 2000

Claude: Based on our conversation yesterday, we discussed...
```

---

## Advanced Configuration

### Custom Tool Definitions

Create `mcp-tools.json` for custom tool schemas:

```json
{
  "tools": [
    {
      "name": "remember",
      "description": "Store information for later recall",
      "parameters": {
        "key": { "type": "string", "description": "What to remember" },
        "value": { "type": "string", "description": "The information" }
      }
    }
  ],
  "mappings": {
    "remember": {
      "tool": "sochdb_put",
      "path_template": "memory/{key}"
    }
  }
}
```

### Server Options

```bash
sochdb-mcp --help

Options:
  --db <PATH>           Database path [default: ~/.sochdb/mcp]
  --socket <PATH>       Unix socket path for IPC mode
  --read-only           Disable write operations
  --max-value-size <N>  Maximum value size in bytes [default: 1048576]
  --log-level <LEVEL>   Logging level [default: info]
```

---

## Troubleshooting

### MCP Server Not Starting

Check logs:
```bash
RUST_LOG=debug sochdb-mcp --db ./test_db
```

### Permission Denied

Ensure the database directory is writable:
```bash
mkdir -p ~/.sochdb
chmod 755 ~/.sochdb
```

### Claude Not Seeing Tools

1. Restart Claude Desktop
2. Check config JSON syntax
3. Verify binary path is correct

### Debug Mode

Add to MCP config:
```json
{
  "mcpServers": {
    "sochdb": {
      "command": "sochdb-mcp",
      "args": ["--db", "~/.sochdb/debug", "--log-level", "debug"],
      "env": {
        "RUST_LOG": "sochdb=trace",
        "RUST_BACKTRACE": "1"
      }
    }
  }
}
```

---

## Discussion

### Security Considerations

- MCP servers run locally with your user permissions
- Data is stored unencrypted by default
- Consider `--read-only` for shared databases

### Performance Tips

- Use `context_query` instead of multiple `get` calls
- Limit scan results with `limit` parameter
- Use vector search for semantic queries

---

## See Also

- [Vector Search Guide](/cookbook/vector-indexing) — Embedding workflows
- [Logging Guide](/cookbook/logging) — Observability setup
- [Getting Started](/getting-started/quickstart) — Quick start guide

