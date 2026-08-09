# search-semantically

Embeddable semantic code search with multi-signal POEM ranking. A Rust library crate that provides local, incremental code search combining BM25 full-text search, vector similarity via ONNX embeddings, path matching, symbol matching, import graph propagation, and git recency — ranked using Pareto-optimal Election Method (POEM).

## Engineering Policies

These are _strict_ policies that must be followed by all engineers and developers in this project. MRs will be rejected if these policies are violated.

### Dependency Management

- All dependencies _must_ be added, removed and updated using `cargo` on the command line.
- Under no circumstances should the Cargo.toml be manually edited with regard to dependencies.

### Coding

- The use of `.unwrap()` is forbidden under _all_ circumstances. The program should _never_ panic.
- In a case where something needs to be unwrapped and it is _logically impossible_ for a panic to occur, the use of `.expect()` with an informative message is permitted.
- Always run `cargo fmt` before committing code.
- Always run `cargo clippy` before committing code.
- Keep code comments to a minimum. Only comment in cases where something is unable to be gleaned from the code itself.

## Build & Test

```bash
cargo build          # debug build (downloads ONNX model on first embed)
cargo test           # run all tests (uses tempfile, no external deps needed)
cargo test -- --nocapture  # run tests with stdout visible
```

The crate uses edition 2024. Default features enable `tree-sitter-rust`. Additional language support via feature flags: `ts-typescript`, `ts-python`, `ts-go`, `ts-java`, `ts-c`, `ts-cpp`.

## Architecture

| Module | Purpose |
|---|---|
| `engine` | `SearchEngine` — top-level API: search (embeds + indexes + ranks in one call) |
| `scanner` | Walks project dir, classifies files by `FileType`, records mtime |
| `chunker` | Dispatches to `text_chunker` or `ts_chunker` based on file type |
| `text_chunker` | Line/paragraph chunking for plain text formats |
| `ts_chunker` | Tree-sitter–aware chunking (functions, structs, impls, etc.) |
| `db` | SQLite via `rusqlite` — stores files, chunks, symbols, imports, embeddings |
| `embedder` | Downloads & runs ONNX `all-MiniLM-L6-v2` via `ort` for vector embeddings |
| `vector_store` | Cosine similarity + top-k retrieval over stored embeddings |
| `metrics` | Computes six per-chunk signals: BM25, cosine, path, symbol, import-graph, git-recency |
| `query_classifier` | Classifies query as `Identifier`, `NaturalLanguage`, or `PathLike` |
| `ranker` | POEM (Pareto-optimal Election Method) multi-signal ranking |
| `format` | Formats ranked results as human-readable text |

### Data flow

1. `SearchEngine::search()` opens/creates `.search-index/search.db` in the project root
2. `scanner` walks the project, diffing against indexed files by mtime
3. New/changed files are chunked, embedded, and stored in SQLite; orphaned embeddings (from interrupted indexing) are repaired
4. Query is classified → six metric signals computed → POEM ranking → formatted output

`SearchEngine::prepare()` eagerly resolves the ONNX model without building the index; `search()` remains lazy and hard-errors on embedder failure.

### Key types

- `SearchEngine` — main entry point, constructed with a project root `PathBuf`
- `StoredChunk` — a chunk row from the DB (id, file_id, path, lines, kind, content)
- `TextChunk` — in-memory chunk produced by chunkers (content, line range, kind, optional name)
- `MetricScores` — six f64 scores per candidate
- `MetricAvailability` — per-metric active/inactive mask for `poem_rank` (use `all_active()` for default)
- `QueryType` — `Identifier` / `NaturalLanguage` / `PathLike`
- `FileType` — enum of supported languages (Rust, TypeScript, Python, Go, Java, C, C++, Markdown, etc.)
- `DownloadEvent` — enum emitted to `DownloadCallback` (`Started`, `Completed`, `Failed`)

## Conventions

- Index is stored at `<project_root>/.search-index/search.db`
- ONNX model cached at `$XDG_CACHE_DIR/search-semantically/models/Xenova/all-MiniLM-L6-v2/`
- Embedding dimension: 384 (all-MiniLM-L6-v2)
- Candidate limit for metrics: 1000
- The `db` module uses integer IDs (i64) for files and chunks; foreign keys enforce referential integrity
- Error handling: `anyhow::Result` throughout; embedder download/init failures propagate as `Err` from `search()` (no silent degradation)
- All tests use `tempfile::TempDir` for isolation

## Dependencies of note

- `ort` (ONNX Runtime) — dynamic loading via `load-dynamic` feature
- `libloading` — pre-flight dylib verification before `ort` session creation
- `rusqlite` — bundled SQLite
- `tree-sitter` + per-language grammars (feature-gated)
- `tokenizers` (HuggingFace) — for embedding tokenization
- `reqwest` (blocking) — for model download
- `ignore` — for `.gitignore`-aware directory walking
