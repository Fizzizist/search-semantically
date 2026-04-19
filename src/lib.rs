#![allow(dead_code)]

//! # search-semantically
//!
//! Embeddable semantic code search with multi-signal POEM ranking.
//!
//! Provides local, incremental code search combining BM25 full-text search,
//! vector similarity via ONNX embeddings, path matching, symbol matching,
//! import graph propagation, and git recency — ranked using Pareto-optimal
//! Election Method (POEM).
//!
//! # Quick start
//!
//! ```no_run
//! use search_semantically::SearchEngine;
//! use std::path::PathBuf;
//!
//! let engine = SearchEngine::new(PathBuf::from("/path/to/project"));
//! let results = engine.search("database connection handler", 20, None).unwrap();
//! println!("{results}");
//! ```

mod chunker;
mod db;
mod embedder;
mod engine;
mod format;
mod metrics;
mod query_classifier;
mod ranker;
mod scanner;
mod text_chunker;
mod ts_chunker;
mod vector_store;

pub use db::StoredChunk;
pub use engine::SearchEngine;
pub use format::{SearchResult, format_results};
pub use query_classifier::QueryType;
pub use ranker::MetricScores;
pub use scanner::FileType;
pub use text_chunker::TextChunk;
