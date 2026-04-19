# search-semantically

![logo](./assets/hero.png)

Embeddable semantic code search with multi-signal POEM ranking.

A Rust library crate that provides local, incremental code search combining BM25 full-text search, vector similarity via ONNX embeddings, path matching, symbol matching, import graph propagation, and git recency — ranked using **Pareto-optimal Election Method (POEM)**.

## Features

- **Semantic search** — ONNX-powered `all-MiniLM-L6-v2` embeddings with cosine similarity
- **Full-text search** — BM25 scoring over chunk content
- **Tree-sitter chunking** — language-aware splitting into functions, structs, impls, etc.
- **Multi-signal ranking** — six independent signals fused via POEM
- **Incremental indexing** — only re-indexes files changed since last run (mtime diffing)
- **Git-aware** — recency signal based on commit history
- **Import graph propagation** — follows import/usage relationships to boost relevant code
- **Zero external services** — runs entirely locally, model downloaded and cached on first use
- **Embeddable** — designed as a library crate, not a CLI tool

## Supported Languages

| Language | Feature Flag |
|---|---|
| Rust | `tree-sitter-rust` (default) |
| TypeScript | `ts-typescript` |
| Python | `ts-python` |
| Go | `ts-go` |
| Java | `ts-java` |
| C | `ts-c` |
| C++ | `ts-cpp` |
| Markdown | built-in (text chunker) |

Plain text and other formats fall back to line/paragraph chunking automatically.

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
search-semantically = "0.1"
```

Then use it:

```rust
use search_semantically::SearchEngine;
use std::path::PathBuf;

let engine = SearchEngine::new(PathBuf::from("/path/to/project"))?;
engine.index()?; // incremental — only processes new/changed files
let results = engine.search("function that parses HTTP headers")?;
for result in results {
    println!("{}", result);
}
```

On first run, the ONNX model (`all-MiniLM-L6-v2`, 384-dim) is downloaded and cached at `$XDG_CACHE_DIR/search-semantically/models/Xenova/all-MiniLM-L6-v2/`.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   SearchEngine                       │
│                  (top-level API)                      │
├──────────┬──────────┬──────────┬───────────┬────────┤
│ scanner  │ chunker  │ embedder │  metrics  │ ranker │
│  (walk)  │ (ts/txt) │  (ONNX)  │ (6 signals)│ (POEM) │
├──────────┴──────────┴──────────┴───────────┴────────┤
│                     db (SQLite)                      │
│         files · chunks · symbols · imports           │
└─────────────────────────────────────────────────────┘
```

### Data Flow

1. `SearchEngine::search()` opens/creates `.search-index/search.db` in the project root
2. **Scanner** walks the project, diffing against indexed files by mtime
3. New/changed files are **chunked**, **embedded**, and stored in SQLite
4. Query is **classified** (`Identifier` / `NaturalLanguage` / `PathLike`)
5. Six metric signals are **computed** per candidate (up to 1000 candidates)
6. Results are **ranked** via POEM and returned as formatted output

### The Six Signals

| Signal | Description |
|---|---|
| **BM25** | Full-text relevance over chunk content |
| **Cosine** | Vector similarity between query and chunk embeddings |
| **Path** | Match strength between query and file path |
| **Symbol** | Match against defined symbol names (functions, structs, etc.) |
| **Import Graph** | Propagation through import/usage relationships |
| **Git Recency** | How recently the file was modified in git history |

## Key Types

| Type | Purpose |
|---|---|
| `SearchEngine` | Main entry point, constructed with a project root `PathBuf` |
| `StoredChunk` | A chunk row from the DB (id, file_id, path, lines, kind, content) |
| `TextChunk` | In-memory chunk produced by chunkers (content, line range, kind, optional name) |
| `MetricScores` | Six `f64` scores per candidate |
| `QueryType` | `Identifier` / `NaturalLanguage` / `PathLike` |
| `FileType` | Enum of supported languages and formats |

## Building & Testing

```bash
cargo build                        # debug build (downloads ONNX model on first embed)
cargo test                         # run all tests (uses tempfile, no external deps needed)
cargo test -- --nocapture          # run tests with stdout visible
```

All tests use `tempfile::TempDir` for full isolation — no setup required.

## Index Storage

- **Index database**: `<project_root>/.search-index/search.db`
- **ONNX model cache**: `$XDG_CACHE_DIR/search-semantically/models/Xenova/all-MiniLM-L6-v2/`
