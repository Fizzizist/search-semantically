use std::path::Path;

use anyhow::{Context, Result};
use rusqlite::params;

const SCHEMA_VERSION: i64 = 1;

#[derive(Debug, Clone)]
pub struct IndexedFile {
    pub id: i64,
    pub file_path: String,
    pub mtime: f64,
    pub file_type: String,
}

#[derive(Debug, Clone)]
pub struct StoredChunk {
    pub id: i64,
    pub file_id: i64,
    pub file_path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub kind: String,
    pub name: Option<String>,
    pub content: String,
    pub file_type: String,
}

pub struct SearchDb {
    conn: rusqlite::Connection,
}

impl SearchDb {
    pub fn open(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("Creating search index directory: {}", parent.display())
            })?;
        }

        let conn = rusqlite::Connection::open(db_path)
            .with_context(|| format!("Opening search DB at {}", db_path.display()))?;

        let mut db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    fn init_schema(&mut self) -> Result<()> {
        self.conn
            .execute_batch(
                "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON;",
            )
            .context("Setting pragmas")?;

        self.conn
            .execute_batch("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            .context("Creating meta table")?;

        let current_version: i64 = self
            .conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'schema_version'",
                [],
                |row| row.get::<_, String>(0),
            )
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        if current_version >= SCHEMA_VERSION {
            return Ok(());
        }

        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL UNIQUE,
                    mtime REAL NOT NULL,
                    file_type TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                    file_path TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    name TEXT,
                    content TEXT NOT NULL,
                    file_type TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path);

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    content,
                    name,
                    file_path,
                    content='chunks',
                    content_rowid='id',
                    tokenize='porter unicode61'
                );

                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, content, name, file_path)
                    VALUES (new.id, new.content, new.name, new.file_path);
                END;

                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, content, name, file_path)
                    VALUES ('delete', old.id, old.content, old.name, old.file_path);
                END;

                CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, content, name, file_path)
                    VALUES ('delete', old.id, old.content, old.name, old.file_path);
                    INSERT INTO chunks_fts(rowid, content, name, file_path)
                    VALUES (new.id, new.content, new.name, new.file_path);
                END;

                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    model_name TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY (chunk_id, model_name)
                );

                CREATE TABLE IF NOT EXISTS imports (
                    source_file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                    target_file_path TEXT NOT NULL,
                    PRIMARY KEY (source_file_id, target_file_path)
                );

                CREATE INDEX IF NOT EXISTS idx_imports_target ON imports(target_file_path);

                CREATE TABLE IF NOT EXISTS symbols (
                    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
                ",
            )
            .context("Creating schema")?;

        self.conn
            .execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?1)",
                params![SCHEMA_VERSION.to_string()],
            )
            .context("Setting schema version")?;

        Ok(())
    }

    pub fn upsert_file(&mut self, file_path: &str, mtime: f64, file_type: &str) -> Result<i64> {
        let existing: Option<i64> = self
            .conn
            .query_row(
                "SELECT id FROM files WHERE file_path = ?1",
                params![file_path],
                |row| row.get(0),
            )
            .ok();

        if let Some(id) = existing {
            self.conn.execute(
                "UPDATE files SET mtime = ?1, file_type = ?2 WHERE id = ?3",
                params![mtime, file_type, id],
            )?;
            return Ok(id);
        }

        self.conn.execute(
            "INSERT INTO files (file_path, mtime, file_type) VALUES (?1, ?2, ?3)",
            params![file_path, mtime, file_type],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn get_file(&mut self, file_path: &str) -> Result<Option<IndexedFile>> {
        let result = self.conn.query_row(
            "SELECT id, file_path, mtime, file_type FROM files WHERE file_path = ?1",
            params![file_path],
            |row| {
                Ok(IndexedFile {
                    id: row.get(0)?,
                    file_path: row.get(1)?,
                    mtime: row.get(2)?,
                    file_type: row.get(3)?,
                })
            },
        );
        match result {
            Ok(f) => Ok(Some(f)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn get_all_files(&mut self) -> Result<Vec<IndexedFile>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, file_path, mtime, file_type FROM files")?;
        let rows = stmt.query_map([], |row| {
            Ok(IndexedFile {
                id: row.get(0)?,
                file_path: row.get(1)?,
                mtime: row.get(2)?,
                file_type: row.get(3)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn delete_file(&mut self, file_id: i64) -> Result<()> {
        self.conn
            .execute("DELETE FROM files WHERE id = ?1", params![file_id])?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn insert_chunk(
        &mut self,
        file_id: i64,
        file_path: &str,
        start_line: i64,
        end_line: i64,
        kind: &str,
        name: Option<&str>,
        content: &str,
        file_type: &str,
    ) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO chunks (file_id, file_path, start_line, end_line, kind, name, content, file_type) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![file_id, file_path, start_line, end_line, kind, name, content, file_type],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn delete_chunks_for_file(&mut self, file_id: i64) -> Result<()> {
        self.conn
            .execute("DELETE FROM chunks WHERE file_id = ?1", params![file_id])?;
        Ok(())
    }

    pub fn get_all_chunks(&mut self) -> Result<Vec<StoredChunk>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, file_id, file_path, start_line, end_line, kind, name, content, file_type FROM chunks",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(StoredChunk {
                id: row.get(0)?,
                file_id: row.get(1)?,
                file_path: row.get(2)?,
                start_line: row.get(3)?,
                end_line: row.get(4)?,
                kind: row.get(5)?,
                name: row.get(6)?,
                content: row.get(7)?,
                file_type: row.get(8)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn get_chunks_by_ids(&mut self, chunk_ids: &[i64]) -> Result<Vec<StoredChunk>> {
        if chunk_ids.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = 500;
        let mut results = Vec::new();

        for chunk in chunk_ids.chunks(batch_size) {
            let placeholders: Vec<String> = chunk
                .iter()
                .enumerate()
                .map(|(i, _)| format!("?{}", i + 1))
                .collect();
            let sql = format!(
                "SELECT id, file_id, file_path, start_line, end_line, kind, name, content, file_type FROM chunks WHERE id IN ({})",
                placeholders.join(",")
            );
            let params: Vec<&dyn rusqlite::ToSql> =
                chunk.iter().map(|id| id as &dyn rusqlite::ToSql).collect();
            let mut stmt = self.conn.prepare(&sql)?;
            let rows = stmt.query_map(params.as_slice(), |row| {
                Ok(StoredChunk {
                    id: row.get(0)?,
                    file_id: row.get(1)?,
                    file_path: row.get(2)?,
                    start_line: row.get(3)?,
                    end_line: row.get(4)?,
                    kind: row.get(5)?,
                    name: row.get(6)?,
                    content: row.get(7)?,
                    file_type: row.get(8)?,
                })
            })?;
            results.extend(rows.collect::<Result<Vec<_>, _>>()?);
        }

        Ok(results)
    }

    pub fn upsert_embedding(
        &mut self,
        chunk_id: i64,
        model_name: &str,
        vector_blob: &[u8],
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings (chunk_id, model_name, vector) VALUES (?1, ?2, ?3)",
            params![chunk_id, model_name, vector_blob],
        )?;
        Ok(())
    }

    pub fn batch_upsert_embeddings(&mut self, items: &[(i64, String, Vec<u8>)]) -> Result<()> {
        let tx = self.conn.transaction()?;
        for (chunk_id, model_name, vector_blob) in items {
            tx.execute(
                "INSERT OR REPLACE INTO embeddings (chunk_id, model_name, vector) VALUES (?1, ?2, ?3)",
                params![chunk_id, model_name, vector_blob],
            )
            .with_context(|| "batch upsert embedding")?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn get_all_embeddings(&mut self, model_name: &str) -> Result<Vec<(i64, Vec<u8>)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT chunk_id, vector FROM embeddings WHERE model_name = ?1")?;
        let rows = stmt.query_map(params![model_name], |row| {
            let chunk_id: i64 = row.get(0)?;
            let vector: Vec<u8> = row.get(1)?;
            Ok((chunk_id, vector))
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn get_chunk_ids_without_embedding(&mut self, model_name: &str) -> Result<Vec<i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id FROM chunks c LEFT JOIN embeddings e ON c.id = e.chunk_id AND e.model_name = ?1 WHERE e.chunk_id IS NULL",
        )?;
        let rows = stmt.query_map(params![model_name], |row| row.get::<_, i64>(0))?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn fts_search(&mut self, query: &str, limit: i64) -> Result<Vec<(i64, f64)>> {
        let result = self
            .conn
            .prepare(
                "SELECT chunks_fts.rowid as chunk_id, -bm25(chunks_fts, 1.0, 10.0, 5.0) as score FROM chunks_fts WHERE chunks_fts MATCH ?1 ORDER BY score DESC LIMIT ?2",
            );
        let mut stmt = match result {
            Ok(s) => s,
            Err(_) => return Ok(Vec::new()),
        };

        let rows = stmt.query_map(params![query, limit], |row| {
            let id: i64 = row.get(0)?;
            let score: f64 = row.get(1)?;
            Ok((id, score))
        });
        match rows {
            Ok(mapped) => {
                let mut results = Vec::new();
                for row in mapped {
                    match row {
                        Ok(r) => results.push(r),
                        Err(_) => return Ok(Vec::new()),
                    }
                }
                Ok(results)
            }
            Err(_) => Ok(Vec::new()),
        }
    }

    pub fn rebuild_fts(&mut self) -> Result<()> {
        self.conn
            .execute_batch("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")?;
        Ok(())
    }

    pub fn insert_import(&mut self, source_file_id: i64, target_file_path: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO imports (source_file_id, target_file_path) VALUES (?1, ?2)",
            params![source_file_id, target_file_path],
        )?;
        Ok(())
    }

    pub fn delete_imports_for_file(&mut self, source_file_id: i64) -> Result<()> {
        self.conn.execute(
            "DELETE FROM imports WHERE source_file_id = ?1",
            params![source_file_id],
        )?;
        Ok(())
    }

    pub fn get_imports_from(&mut self, source_file_id: i64) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT target_file_path FROM imports WHERE source_file_id = ?1")?;
        let rows = stmt.query_map(params![source_file_id], |row| row.get::<_, String>(0))?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn get_importers_of(&mut self, target_file_path: &str) -> Result<Vec<i64>> {
        let mut stmt = self
            .conn
            .prepare("SELECT source_file_id FROM imports WHERE target_file_path = ?1")?;
        let rows = stmt.query_map(params![target_file_path], |row| row.get::<_, i64>(0))?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn insert_symbol(&mut self, chunk_id: i64, name: &str, kind: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO symbols (chunk_id, name, kind) VALUES (?1, ?2, ?3)",
            params![chunk_id, name, kind],
        )?;
        Ok(())
    }

    pub fn get_all_symbols(&mut self) -> Result<Vec<(i64, String)>> {
        let mut stmt = self.conn.prepare("SELECT chunk_id, name FROM symbols")?;
        let rows = stmt.query_map([], |row| {
            let chunk_id: i64 = row.get(0)?;
            let name: String = row.get(1)?;
            Ok((chunk_id, name))
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn get_chunk_count(&mut self) -> Result<i64> {
        self.conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
            .map_err(Into::into)
    }

    pub fn get_file_count(&mut self) -> Result<i64> {
        self.conn
            .query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))
            .map_err(Into::into)
    }

    pub fn get_files_with_zero_chunks(&mut self) -> Result<Vec<IndexedFile>> {
        let mut stmt = self.conn.prepare(
            "SELECT f.id, f.file_path, f.mtime, f.file_type
             FROM files f
             LEFT JOIN chunks c ON c.file_id = f.id
             WHERE c.id IS NULL",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(IndexedFile {
                id: row.get(0)?,
                file_path: row.get(1)?,
                mtime: row.get(2)?,
                file_type: row.get(3)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn get_orphaned_chunk_ids(&mut self) -> Result<Vec<i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id FROM chunks c
             LEFT JOIN files f ON f.id = c.file_id
             WHERE f.id IS NULL",
        )?;
        let rows = stmt.query_map([], |row| row.get::<_, i64>(0))?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn delete_chunks_by_ids(&mut self, chunk_ids: &[i64]) -> Result<()> {
        if chunk_ids.is_empty() {
            return Ok(());
        }
        let placeholders: Vec<String> = chunk_ids
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect();
        let sql = format!(
            "DELETE FROM chunks WHERE id IN ({})",
            placeholders.join(",")
        );
        let params: Vec<&dyn rusqlite::ToSql> = chunk_ids
            .iter()
            .map(|id| id as &dyn rusqlite::ToSql)
            .collect();
        self.conn.execute(&sql, params.as_slice())?;
        Ok(())
    }

    #[cfg(test)]
    pub fn inject_orphaned_chunk(&mut self) -> Result<()> {
        self.conn.execute("PRAGMA foreign_keys=OFF", [])?;
        self.conn.execute(
            "INSERT INTO chunks (file_id, file_path, start_line, end_line, kind, content, file_type)
             VALUES (99999, 'orphan.rs', 1, 1, 'file', 'orphan code', 'rust')",
            [],
        )?;
        self.conn.execute("PRAGMA foreign_keys=ON", [])?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn index_file_transactional(
        &mut self,
        file_path: &str,
        mtime: f64,
        file_type: &str,
        existing_file_id: Option<i64>,
        chunks: &[super::text_chunker::TextChunk],
        imports: &[String],
    ) -> Result<(i64, Vec<i64>)> {
        let tx = self.conn.transaction()?;

        let file_id = match existing_file_id {
            Some(id) => {
                tx.execute(
                    "UPDATE files SET mtime = ?1, file_type = ?2 WHERE id = ?3",
                    params![mtime, file_type, id],
                )?;
                id
            }
            None => {
                tx.execute(
                    "INSERT INTO files (file_path, mtime, file_type) VALUES (?1, ?2, ?3)",
                    params![file_path, mtime, file_type],
                )?;
                tx.last_insert_rowid()
            }
        };

        tx.execute("DELETE FROM chunks WHERE file_id = ?1", params![file_id])?;
        tx.execute(
            "DELETE FROM imports WHERE source_file_id = ?1",
            params![file_id],
        )?;

        let mut chunk_ids = Vec::with_capacity(chunks.len());
        for text_chunk in chunks {
            tx.execute(
                "INSERT INTO chunks (file_id, file_path, start_line, end_line, kind, name, content, file_type)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    file_id,
                    text_chunk.file_path,
                    text_chunk.start_line as i64,
                    text_chunk.end_line as i64,
                    text_chunk.kind.to_string(),
                    text_chunk.name,
                    text_chunk.content,
                    file_type,
                ],
            )?;
            let chunk_id = tx.last_insert_rowid();
            chunk_ids.push(chunk_id);

            if let Some(ref name) = text_chunk.name {
                tx.execute(
                    "INSERT INTO symbols (chunk_id, name, kind) VALUES (?1, ?2, ?3)",
                    params![chunk_id, name, text_chunk.kind.to_string()],
                )?;
            }
        }

        for target_path in imports {
            tx.execute(
                "INSERT OR IGNORE INTO imports (source_file_id, target_file_path) VALUES (?1, ?2)",
                params![file_id, target_path],
            )?;
        }

        tx.commit()?;
        Ok((file_id, chunk_ids))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_db() -> SearchDb {
        let temp = TempDir::new().expect("temp dir");
        let db_path = temp.path().join("test.db");
        SearchDb::open(&db_path).expect("should open db")
    }

    #[test]
    fn schema_initializes_without_error() {
        let _db = test_db();
    }

    #[test]
    fn upsert_and_get_file() {
        let mut db = test_db();
        let id = db
            .upsert_file("src/main.rs", 1234.5, "rust")
            .expect("upsert");
        assert!(id > 0);

        let file = db.get_file("src/main.rs").expect("get");
        assert!(file.is_some());
        let f = file.expect("file");
        assert_eq!(f.file_path, "src/main.rs");
        assert_eq!(f.file_type, "rust");
    }

    #[test]
    fn upsert_file_idempotent() {
        let mut db = test_db();
        let id1 = db.upsert_file("test.rs", 1.0, "rust").expect("upsert1");
        let id2 = db.upsert_file("test.rs", 2.0, "rust").expect("upsert2");
        assert_eq!(id1, id2);
    }

    #[test]
    fn insert_and_get_chunks() {
        let mut db = test_db();
        let file_id = db.upsert_file("main.rs", 1.0, "rust").expect("file");

        let chunk_id = db
            .insert_chunk(
                file_id,
                "main.rs",
                1,
                5,
                "function",
                Some("main"),
                "fn main() {}",
                "rust",
            )
            .expect("insert chunk");

        assert!(chunk_id > 0);

        let chunks = db.get_all_chunks().expect("get all");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].name, Some("main".to_string()));
    }

    #[test]
    fn delete_file_cascades_to_chunks() {
        let mut db = test_db();
        let file_id = db.upsert_file("temp.rs", 1.0, "rust").expect("file");
        db.insert_chunk(file_id, "temp.rs", 1, 1, "file", None, "code", "rust")
            .expect("chunk");

        db.delete_file(file_id).expect("delete");

        let chunks = db.get_all_chunks().expect("chunks");
        assert!(chunks.is_empty());
    }

    #[test]
    fn embedding_crud() {
        let mut db = test_db();
        let file_id = db.upsert_file("test.rs", 1.0, "rust").expect("file");
        let chunk_id = db
            .insert_chunk(file_id, "test.rs", 1, 1, "file", None, "code", "rust")
            .expect("chunk");

        let vector = vec![0.1_f32, 0.2, 0.3];
        let blob = crate::vector_store::pack_vector(&vector);
        db.upsert_embedding(chunk_id, "test-model", &blob)
            .expect("upsert embedding");

        let embeddings = db.get_all_embeddings("test-model").expect("get embeddings");
        assert_eq!(embeddings.len(), 1);
    }

    #[test]
    fn get_chunk_ids_without_embedding() {
        let mut db = test_db();
        let file_id = db.upsert_file("test.rs", 1.0, "rust").expect("file");
        let c1 = db
            .insert_chunk(file_id, "test.rs", 1, 1, "file", None, "a", "rust")
            .expect("chunk");
        let c2 = db
            .insert_chunk(file_id, "test.rs", 2, 2, "file", None, "b", "rust")
            .expect("chunk");

        let vector = vec![0.1_f32];
        let blob = crate::vector_store::pack_vector(&vector);
        db.upsert_embedding(c1, "model", &blob).expect("embed");

        let missing = db
            .get_chunk_ids_without_embedding("model")
            .expect("missing");
        assert_eq!(missing, vec![c2]);
    }

    #[test]
    fn import_crud() {
        let mut db = test_db();
        let f1 = db.upsert_file("main.rs", 1.0, "rust").expect("file");
        db.upsert_file("lib.rs", 1.0, "rust").expect("file2");
        db.insert_import(f1, "lib.rs").expect("insert import");

        let imports = db.get_imports_from(f1).expect("imports");
        assert_eq!(imports, vec!["lib.rs"]);

        let importers = db.get_importers_of("lib.rs").expect("importers");
        assert_eq!(importers, vec![f1]);
    }

    #[test]
    fn symbol_crud() {
        let mut db = test_db();
        let file_id = db.upsert_file("test.rs", 1.0, "rust").expect("file");
        let chunk_id = db
            .insert_chunk(
                file_id,
                "test.rs",
                1,
                5,
                "function",
                Some("main"),
                "fn main() {}",
                "rust",
            )
            .expect("chunk");

        db.insert_symbol(chunk_id, "main", "function")
            .expect("symbol");

        let symbols = db.get_all_symbols().expect("symbols");
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0], (chunk_id, "main".to_string()));
    }

    #[test]
    fn get_chunks_by_ids() {
        let mut db = test_db();
        let file_id = db.upsert_file("test.rs", 1.0, "rust").expect("file");
        let c1 = db
            .insert_chunk(file_id, "test.rs", 1, 1, "file", None, "a", "rust")
            .expect("chunk");
        let c2 = db
            .insert_chunk(file_id, "test.rs", 2, 2, "file", None, "b", "rust")
            .expect("chunk");

        let chunks = db.get_chunks_by_ids(&[c1, c2]).expect("get by ids");
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn get_all_files_returns_inserted() {
        let mut db = test_db();
        db.upsert_file("a.rs", 1.0, "rust").expect("file");
        db.upsert_file("b.rs", 2.0, "rust").expect("file");

        let files = db.get_all_files().expect("get all");
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn fts_search_after_insert() {
        let mut db = test_db();
        let file_id = db.upsert_file("test.rs", 1.0, "rust").expect("file");
        db.insert_chunk(
            file_id,
            "test.rs",
            1,
            5,
            "function",
            Some("search_engine"),
            "fn search_engine() { /* search implementation */ }",
            "rust",
        )
        .expect("chunk");

        let results = db.fts_search("search_engine", 10).expect("fts");
        assert_eq!(results.len(), 1);
        assert!(results[0].1 > 0.0);
    }

    #[test]
    fn fts_search_malformed_query_returns_empty() {
        let mut db = test_db();
        let file_id = db.upsert_file("test.rs", 1.0, "rust").expect("file");
        db.insert_chunk(file_id, "test.rs", 1, 1, "file", None, "content", "rust")
            .expect("chunk");

        let results = db.fts_search("??? OR AND NOT", 10).expect("fts");
        assert!(results.is_empty());
    }

    #[test]
    fn batch_upsert_embeddings_transactional() {
        let mut db = test_db();
        let file_id = db.upsert_file("test.rs", 1.0, "rust").expect("file");
        let c1 = db
            .insert_chunk(file_id, "test.rs", 1, 1, "file", None, "a", "rust")
            .expect("chunk");
        let c2 = db
            .insert_chunk(file_id, "test.rs", 2, 2, "file", None, "b", "rust")
            .expect("chunk");

        let blob1 = crate::vector_store::pack_vector(&[0.1_f32]);
        let blob2 = crate::vector_store::pack_vector(&[0.2_f32]);

        db.batch_upsert_embeddings(&[
            (c1, "model".to_string(), blob1),
            (c2, "model".to_string(), blob2),
        ])
        .expect("batch upsert");

        let embeddings = db.get_all_embeddings("model").expect("get");
        assert_eq!(embeddings.len(), 2);
    }

    #[test]
    fn get_files_with_zero_chunks_returns_files_without_chunks() {
        let mut db = test_db();
        let id_with_chunks = db.upsert_file("with.rs", 1.0, "rust").expect("file");
        db.insert_chunk(
            id_with_chunks,
            "with.rs",
            1,
            1,
            "file",
            None,
            "code",
            "rust",
        )
        .expect("chunk");

        let _id_without = db.upsert_file("without.rs", 1.0, "rust").expect("file");

        let zero_chunk_files = db.get_files_with_zero_chunks().expect("query");
        assert_eq!(zero_chunk_files.len(), 1);
        assert_eq!(zero_chunk_files[0].file_path, "without.rs");
    }

    #[test]
    fn get_files_with_zero_chunks_empty_db_returns_empty() {
        let mut db = test_db();
        let result = db.get_files_with_zero_chunks().expect("query");
        assert!(result.is_empty());
    }

    #[test]
    fn get_orphaned_chunk_ids_detects_orphans() {
        let mut db = test_db();
        let file_id = db.upsert_file("real.rs", 1.0, "rust").expect("file");
        db.insert_chunk(file_id, "real.rs", 1, 1, "file", None, "code", "rust")
            .expect("chunk");

        db.conn
            .execute("PRAGMA foreign_keys=OFF", [])
            .expect("disable FK");
        db.conn
            .execute(
                "INSERT INTO chunks (file_id, file_path, start_line, end_line, kind, content, file_type)
                 VALUES (99999, 'orphan.rs', 1, 1, 'file', 'code', 'rust')",
                [],
            )
            .expect("insert orphan");
        db.conn
            .execute("PRAGMA foreign_keys=ON", [])
            .expect("re-enable FK");

        let orphans = db.get_orphaned_chunk_ids().expect("query");
        assert_eq!(orphans.len(), 1);
    }

    #[test]
    fn index_file_transactional_inserts_chunks_atomically() {
        let mut db = test_db();
        let chunks = vec![crate::text_chunker::TextChunk {
            file_path: "test.rs".to_string(),
            start_line: 1,
            end_line: 3,
            kind: crate::text_chunker::ChunkKind::Function,
            name: Some("my_func".to_string()),
            content: "fn my_func() {}".to_string(),
        }];

        let (file_id, chunk_ids) = db
            .index_file_transactional("test.rs", 1.0, "rust", None, &chunks, &[])
            .expect("transactional");

        assert!(file_id > 0);
        assert_eq!(chunk_ids.len(), 1);

        let all_chunks = db.get_all_chunks().expect("chunks");
        assert_eq!(all_chunks.len(), 1);
        assert_eq!(all_chunks[0].name, Some("my_func".to_string()));

        let symbols = db.get_all_symbols().expect("symbols");
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].1, "my_func");
    }

    #[test]
    fn index_file_transactional_replaces_old_chunks() {
        let mut db = test_db();
        let old_chunks = vec![crate::text_chunker::TextChunk {
            file_path: "test.rs".to_string(),
            start_line: 1,
            end_line: 1,
            kind: crate::text_chunker::ChunkKind::Function,
            name: Some("old".to_string()),
            content: "fn old() {}".to_string(),
        }];

        let (file_id, old_ids) = db
            .index_file_transactional("test.rs", 1.0, "rust", None, &old_chunks, &[])
            .expect("first insert");
        assert_eq!(old_ids.len(), 1);

        let new_chunks = vec![crate::text_chunker::TextChunk {
            file_path: "test.rs".to_string(),
            start_line: 1,
            end_line: 2,
            kind: crate::text_chunker::ChunkKind::Function,
            name: Some("new".to_string()),
            content: "fn new() {}".to_string(),
        }];

        let (_, new_ids) = db
            .index_file_transactional("test.rs", 2.0, "rust", Some(file_id), &new_chunks, &[])
            .expect("second insert");
        assert_eq!(new_ids.len(), 1);

        let all_chunks = db.get_all_chunks().expect("chunks");
        assert_eq!(all_chunks.len(), 1);
        assert_eq!(all_chunks[0].name, Some("new".to_string()));

        let symbols = db.get_all_symbols().expect("symbols");
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].1, "new");
    }

    #[test]
    fn index_file_transactional_inserts_imports() {
        let mut db = test_db();
        let chunks = vec![crate::text_chunker::TextChunk {
            file_path: "main.rs".to_string(),
            start_line: 1,
            end_line: 1,
            kind: crate::text_chunker::ChunkKind::File,
            name: None,
            content: "fn main() {}".to_string(),
        }];

        let imports = vec!["lib.rs".to_string()];
        let (file_id, _) = db
            .index_file_transactional("main.rs", 1.0, "rust", None, &chunks, &imports)
            .expect("insert");

        let from_imports = db.get_imports_from(file_id).expect("imports");
        assert_eq!(from_imports, vec!["lib.rs"]);
    }

    #[test]
    fn delete_chunks_by_ids_removes_chunks() {
        let mut db = test_db();
        let file_id = db.upsert_file("test.rs", 1.0, "rust").expect("file");
        let c1 = db
            .insert_chunk(file_id, "test.rs", 1, 1, "file", None, "a", "rust")
            .expect("chunk");
        db.insert_chunk(file_id, "test.rs", 2, 2, "file", None, "b", "rust")
            .expect("chunk");

        db.delete_chunks_by_ids(&[c1]).expect("delete");

        let chunks = db.get_all_chunks().expect("chunks");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "b");
    }

    #[test]
    fn delete_chunks_by_ids_empty_is_noop() {
        let mut db = test_db();
        db.delete_chunks_by_ids(&[]).expect("delete empty");
        let chunks = db.get_all_chunks().expect("chunks");
        assert!(chunks.is_empty());
    }
}
