use anyhow::{Context, Result};
use ort::session::Session;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum DownloadEvent {
    Started { url: String },
    Completed { url: String },
    Failed { url: String, error: String },
}

pub type DownloadCallback = Arc<dyn Fn(DownloadEvent) + Send + Sync>;

pub(crate) const DEFAULT_MODEL_NAME: &str = "Xenova/all-MiniLM-L6-v2";
const DEFAULT_DIMENSION: usize = 384;
const MIN_MODEL_SIZE_BYTES: u64 = 1_000_000;

pub struct Embedder {
    model_cache_dir: PathBuf,
    model_name: String,
    session: Option<Session>,
    tokenizer: Option<tokenizers::Tokenizer>,
    dimension: usize,
    download_callback: Option<DownloadCallback>,
}

impl Embedder {
    pub fn new(model_cache_dir: PathBuf) -> Self {
        Self {
            model_cache_dir,
            model_name: DEFAULT_MODEL_NAME.to_string(),
            session: None,
            tokenizer: None,
            dimension: DEFAULT_DIMENSION,
            download_callback: None,
        }
    }

    pub fn with_download_callback(mut self, callback: DownloadCallback) -> Self {
        self.download_callback = Some(callback);
        self
    }

    pub fn set_download_callback(&mut self, callback: DownloadCallback) {
        self.download_callback = Some(callback);
    }

    pub fn is_initialized(&self) -> bool {
        self.session.is_some()
    }

    pub fn initialize(&mut self) -> Result<()> {
        if self.session.is_some() {
            return Ok(());
        }

        let model_dir = self.model_cache_dir.join(&self.model_name);
        std::fs::create_dir_all(&model_dir)
            .with_context(|| format!("Creating model cache dir: {}", model_dir.display()))?;

        if !cache_is_valid(&model_dir) {
            download_model(&self.model_name, &model_dir, self.download_callback.clone())?;
        }

        let onnx_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        let dylib_path = resolve_onnx_runtime()?;
        configure_onnx_runtime(&dylib_path);

        let session = Session::builder()
            .context("Creating ONNX session builder")?
            .commit_from_file(&onnx_path)
            .with_context(|| format!("Loading ONNX model from {}", onnx_path.display()))?;

        let tokenizer_data = std::fs::read_to_string(&tokenizer_path)
            .with_context(|| format!("Reading tokenizer from {}", tokenizer_path.display()))?;
        let tokenizer = tokenizers::Tokenizer::from_str(&tokenizer_data)
            .map_err(|e| anyhow::anyhow!("Parsing tokenizer JSON: {e}"))?;

        self.dimension = detect_dimension(&session);
        self.session = Some(session);
        self.tokenizer = Some(tokenizer);

        Ok(())
    }

    pub fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let tokenizer = self.tokenizer.as_ref().expect("Embedder not initialized");

        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            let encoding = tokenizer
                .encode(*text, true)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

            let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
            let attention_mask: Vec<i64> = encoding
                .get_attention_mask()
                .iter()
                .map(|&m| m as i64)
                .collect();
            let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();

            let len = ids.len();
            let input_ids = ndarray::Array2::from_shape_vec((1, len), ids)
                .context("Creating input_ids array")?;
            let attn_mask = ndarray::Array2::from_shape_vec((1, len), attention_mask)
                .context("Creating attention_mask array")?;
            let token_types = ndarray::Array2::from_shape_vec((1, len), type_ids)
                .context("Creating token_type_ids array")?;

            let session = self.session.as_mut().expect("Embedder not initialized");

            let input_ids_val =
                ort::value::Tensor::from_array(input_ids).context("Creating input_ids tensor")?;
            let attn_mask_val = ort::value::Tensor::from_array(attn_mask)
                .context("Creating attention_mask tensor")?;
            let token_types_val = ort::value::Tensor::from_array(token_types)
                .context("Creating token_type_ids tensor")?;

            let outputs = session
                .run(ort::inputs! {
                    "input_ids" => input_ids_val,
                    "attention_mask" => attn_mask_val,
                    "token_type_ids" => token_types_val,
                })
                .context("Running ONNX inference")?;

            let output = outputs.iter().next().context("No output from model")?.1;

            let (_, data) = output
                .try_extract_tensor::<f32>()
                .context("Extracting tensor")?;

            let mask_f32: Vec<f32> = encoding
                .get_attention_mask()
                .iter()
                .map(|&m| m as f32)
                .collect();
            let embedding = mean_pool_normalize(data, len, self.dimension, &mask_f32);

            results.push(embedding);
        }

        Ok(results)
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

pub(crate) fn cache_is_valid(model_dir: &Path) -> bool {
    let model_path = model_dir.join("model.onnx");
    let tokenizer_path = model_dir.join("tokenizer.json");

    if !model_dir.is_dir() {
        return false;
    }

    let model_ok = std::fs::metadata(&model_path)
        .map(|m| m.len() > MIN_MODEL_SIZE_BYTES)
        .unwrap_or(false);

    let tokenizer_ok = std::fs::metadata(&tokenizer_path)
        .map(|m| m.len() > 0)
        .unwrap_or(false);

    model_ok && tokenizer_ok
}

fn onnx_runtime_dylib_name() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "onnxruntime.dll"
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        "libonnxruntime.so"
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        "libonnxruntime.dylib"
    }
    #[cfg(not(any(
        target_os = "windows",
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios"
    )))]
    {
        "libonnxruntime.so"
    }
}

static ORT_DYLIB_INIT: std::sync::OnceLock<()> = std::sync::OnceLock::new();

fn resolve_onnx_runtime() -> Result<PathBuf> {
    let dylib_name = onnx_runtime_dylib_name();
    let path = resolve_dylib_path(dylib_name);

    let lib = unsafe { libloading::Library::new(&path) }.map_err(|e| {
        anyhow::anyhow!(
            "ONNX Runtime dynamic library not found at '{}': {e}\n\
             Install libonnxruntime or set ORT_DYLIB_PATH to its location.\n\
             See: https://ort.pyke.io/setup/linking",
            path.display()
        )
    })?;

    let _symbol: libloading::Symbol<unsafe extern "C" fn() -> *const std::ffi::c_void> =
        unsafe { lib.get(b"OrtGetApiBase") }.map_err(|_| {
            anyhow::anyhow!(
                "Loaded '{}' but 'OrtGetApiBase' symbol is missing — \
                 the library may be an incompatible ONNX Runtime version",
                path.display()
            )
        })?;

    std::mem::forget(lib);

    Ok(path)
}

fn configure_onnx_runtime(dylib_path: &Path) {
    if !dylib_path.is_absolute() || !dylib_path.exists() {
        return;
    }

    ORT_DYLIB_INIT.get_or_init(|| {
        if std::env::var("ORT_DYLIB_PATH").is_err() {
            // SAFETY: guarded by OnceLock so the write happens at most once
            // process-wide, before any ort session is created. Callers that
            // need to set ORT_DYLIB_PATH themselves should do so before
            // constructing a SearchEngine.
            unsafe {
                std::env::set_var("ORT_DYLIB_PATH", dylib_path);
            }
        }
    });
}

fn resolve_dylib_path(dylib_name: &str) -> PathBuf {
    if let Ok(s) = std::env::var("ORT_DYLIB_PATH")
        && !s.is_empty()
    {
        return PathBuf::from(s);
    }

    let exe_relative = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .map(|dir| dir.join(dylib_name));
    if let Some(ref p) = exe_relative
        && p.exists()
    {
        return p.clone();
    }

    #[cfg(unix)]
    {
        if let Some(resolved) = resolve_via_ldconfig(dylib_name) {
            return resolved;
        }
    }

    PathBuf::from(dylib_name)
}

#[cfg(unix)]
fn resolve_via_ldconfig(dylib_name: &str) -> Option<PathBuf> {
    let output = std::process::Command::new("ldconfig")
        .arg("-p")
        .output()
        .ok()?;
    let stdout = String::from_utf8(output.stdout).ok()?;

    let soname = format!("\t{dylib_name}.");
    for line in stdout.lines() {
        if (line.contains(&soname) || line.contains(&format!("\t{dylib_name} ")))
            && let Some(path_part) = line.rsplit(" => ").next()
        {
            let path = path_part.trim();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }
    None
}

fn detect_dimension(session: &Session) -> usize {
    session
        .outputs()
        .first()
        .and_then(|outlet| outlet.dtype().tensor_shape())
        .and_then(|shape| shape.last().copied())
        .filter(|&d| d > 0)
        .map(|d| d as usize)
        .unwrap_or(DEFAULT_DIMENSION)
}

fn mean_pool_normalize(data: &[f32], seq_len: usize, dim: usize, mask: &[f32]) -> Vec<f32> {
    let mut pooled = vec![0.0_f32; dim];
    let mut mask_sum = 0.0_f32;

    for i in 0..seq_len {
        let weight = mask[i];
        mask_sum += weight;
        for j in 0..dim {
            pooled[j] += data[i * dim + j] * weight;
        }
    }

    if mask_sum > 0.0 {
        for val in pooled.iter_mut() {
            *val /= mask_sum;
        }
    }

    let norm: f32 = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in pooled.iter_mut() {
            *val /= norm;
        }
    }

    pooled
}

fn download_model(
    model_name: &str,
    target_dir: &Path,
    callback: Option<DownloadCallback>,
) -> Result<()> {
    let files = [
        ("onnx/model.onnx", "model.onnx"),
        ("tokenizer.json", "tokenizer.json"),
    ];

    let model_name_owned = model_name.to_string();
    let target_dir_owned = target_dir.to_path_buf();

    let handle = std::thread::spawn(move || -> Result<()> {
        for (remote_file, local_file) in &files {
            let url =
                format!("https://huggingface.co/{model_name_owned}/resolve/main/{remote_file}");
            let dest = target_dir_owned.join(local_file);
            let tmp_dest = target_dir_owned.join(format!("{local_file}.tmp"));

            if let Err(e) = download_single_file(&url, &dest, &tmp_dest, callback.as_ref()) {
                if let Some(ref cb) = callback {
                    cb(DownloadEvent::Failed {
                        url: url.clone(),
                        error: e.to_string(),
                    });
                }
                return Err(e);
            }
        }
        Ok(())
    });

    let result = handle
        .join()
        .map_err(|e| anyhow::anyhow!("Model download thread panicked: {e:?}"))?;

    if result.is_err() {
        cleanup_tmp_files(target_dir);
    }

    result
}

fn download_single_file(
    url: &str,
    dest: &Path,
    tmp_dest: &Path,
    callback: Option<&DownloadCallback>,
) -> Result<()> {
    if let Some(cb) = callback {
        cb(DownloadEvent::Started {
            url: url.to_string(),
        });
    }

    let response = reqwest::blocking::get(url)
        .with_context(|| format!("HTTP request to {url}"))?
        .error_for_status()
        .context("HTTP request failed")?;
    let buf = response.bytes().context("Reading response body")?;

    std::fs::write(tmp_dest, &buf)
        .with_context(|| format!("Writing temp file {}", tmp_dest.display()))?;

    std::fs::rename(tmp_dest, dest)
        .with_context(|| format!("Renaming {} to {}", tmp_dest.display(), dest.display()))?;

    if let Some(cb) = callback {
        cb(DownloadEvent::Completed {
            url: url.to_string(),
        });
    }

    Ok(())
}

fn cleanup_tmp_files(dir: &Path) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "tmp") {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn embedder_new_has_no_session() {
        let embedder = Embedder::new(std::env::temp_dir());
        assert!(embedder.session.is_none());
        assert!(embedder.tokenizer.is_none());
    }

    #[test]
    fn embedder_default_dimension() {
        let embedder = Embedder::new(std::env::temp_dir());
        assert_eq!(embedder.dimension(), 384);
    }

    #[test]
    fn mean_pool_normalize_produces_unit_vector() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = vec![1.0_f32, 1.0];
        let result = mean_pool_normalize(&data, 2, 3, &mask);

        let norm: f32 = result.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "Should be unit vector, got norm {norm}"
        );
    }

    #[test]
    fn mean_pool_normalize_with_zero_mask() {
        let data = vec![1.0_f32, 2.0, 3.0];
        let mask = vec![0.0_f32];
        let result = mean_pool_normalize(&data, 1, 3, &mask);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn cache_is_valid_cold_dir_returns_false() {
        let temp_dir = TempDir::new().expect("temp dir");
        let non_existent = temp_dir.path().join("does_not_exist");
        assert!(!cache_is_valid(&non_existent));
    }

    #[test]
    fn cache_is_valid_only_tokenizer_returns_false() {
        let temp_dir = TempDir::new().expect("temp dir");
        fs::write(temp_dir.path().join("tokenizer.json"), b"{}").expect("write");
        assert!(!cache_is_valid(temp_dir.path()));
    }

    #[test]
    fn cache_is_valid_zero_byte_model_returns_false() {
        let temp_dir = TempDir::new().expect("temp dir");
        fs::write(temp_dir.path().join("model.onnx"), b"").expect("write");
        fs::write(temp_dir.path().join("tokenizer.json"), b"{}").expect("write");
        assert!(!cache_is_valid(temp_dir.path()));
    }

    #[test]
    fn cache_is_valid_truncated_model_returns_false() {
        let temp_dir = TempDir::new().expect("temp dir");
        let small_data = vec![0u8; 500];
        fs::write(temp_dir.path().join("model.onnx"), small_data).expect("write");
        fs::write(temp_dir.path().join("tokenizer.json"), b"{}").expect("write");
        assert!(!cache_is_valid(temp_dir.path()));
    }

    #[test]
    fn cache_is_valid_empty_tokenizer_returns_false() {
        let temp_dir = TempDir::new().expect("temp dir");
        let big_data = vec![0u8; 1_000_001];
        fs::write(temp_dir.path().join("model.onnx"), big_data).expect("write");
        fs::write(temp_dir.path().join("tokenizer.json"), b"").expect("write");
        assert!(!cache_is_valid(temp_dir.path()));
    }

    #[test]
    fn cache_is_valid_valid_cache_returns_true() {
        let temp_dir = TempDir::new().expect("temp dir");
        let big_data = vec![0u8; 1_000_001];
        fs::write(temp_dir.path().join("model.onnx"), big_data).expect("write");
        fs::write(temp_dir.path().join("tokenizer.json"), b"{}").expect("write");
        assert!(cache_is_valid(temp_dir.path()));
    }

    #[test]
    fn download_model_unroutable_url_returns_error_and_no_partials() {
        let temp_dir = TempDir::new().expect("temp dir");
        let result = download_model("nonexistent/nonexistent-model-xyz", temp_dir.path(), None);
        assert!(result.is_err());

        let entries = fs::read_dir(temp_dir.path()).expect("read dir");
        for entry in entries {
            let entry = entry.expect("dir entry");
            let path = entry.path();
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .expect("file name");
            assert!(
                !file_name.ends_with(".tmp"),
                "Found leftover .tmp file: {}",
                file_name
            );
            assert!(
                file_name != "model.onnx" && file_name != "tokenizer.json",
                "Found unexpected file: {}",
                file_name
            );
        }
    }

    #[test]
    fn download_callback_receives_started_event() {
        let temp_dir = TempDir::new().expect("temp dir");
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let events_clone = events.clone();
        let callback: DownloadCallback = std::sync::Arc::new(move |event: DownloadEvent| {
            events_clone.lock().expect("lock").push(event);
        });
        let result = download_model(
            "nonexistent/nonexistent-model-xyz",
            temp_dir.path(),
            Some(callback),
        );
        assert!(result.is_err());
        let events = events.lock().expect("lock");
        assert!(
            events
                .iter()
                .any(|e| matches!(e, DownloadEvent::Started { .. })),
            "Should receive Started event"
        );
    }

    #[test]
    fn download_callback_receives_failed_event() {
        let temp_dir = TempDir::new().expect("temp dir");
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let events_clone = events.clone();
        let callback: DownloadCallback = std::sync::Arc::new(move |event: DownloadEvent| {
            events_clone.lock().expect("lock").push(event);
        });
        let result = download_model(
            "nonexistent/nonexistent-model-xyz",
            temp_dir.path(),
            Some(callback),
        );
        assert!(result.is_err());
        let events = events.lock().expect("lock");
        assert!(
            events
                .iter()
                .any(|e| matches!(e, DownloadEvent::Failed { .. })),
            "Should receive Failed event"
        );
    }

    #[test]
    fn resolve_onnx_runtime_missing_dylib_returns_error() {
        let orig = std::env::var("ORT_DYLIB_PATH").ok();
        // SAFETY: isolated test — temporarily clearing ORT_DYLIB_PATH to force
        // the probe to search the default dlopen path.
        unsafe {
            std::env::remove_var("ORT_DYLIB_PATH");
        }
        let result = resolve_onnx_runtime();
        if let Some(v) = orig {
            // SAFETY: restoring the original env var after the probe.
            unsafe {
                std::env::set_var("ORT_DYLIB_PATH", v);
            }
        }
        match result {
            Ok(_) => {
                // The dylib is installed on this machine — the "missing dylib"
                // path cannot be exercised. Skip without failing.
            }
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("ONNX Runtime") || msg.contains("ORT_DYLIB_PATH"),
                    "Error should guide user to install or set ORT_DYLIB_PATH, got: {msg}"
                );
            }
        }
    }

    #[test]
    fn resolve_onnx_runtime_bad_dylib_path_returns_error() {
        let temp_dir = TempDir::new().expect("temp dir");
        let fake_dylib = temp_dir.path().join("libonnxruntime.so");
        fs::write(&fake_dylib, b"not a shared library").expect("write");

        let orig = std::env::var("ORT_DYLIB_PATH").ok();
        // SAFETY: isolated test — setting ORT_DYLIB_PATH to a file that is not
        // a valid shared library, to exercise the probe's error path.
        unsafe {
            std::env::set_var("ORT_DYLIB_PATH", &fake_dylib);
        }
        let result = resolve_onnx_runtime();
        if let Some(v) = orig {
            // SAFETY: restoring the original env var.
            unsafe {
                std::env::set_var("ORT_DYLIB_PATH", v);
            }
        } else {
            // SAFETY: removing the env var we just set.
            unsafe {
                std::env::remove_var("ORT_DYLIB_PATH");
            }
        }
        assert!(result.is_err());
    }

    #[test]
    fn configure_onnx_runtime_guards_against_relative_path() {
        let orig = std::env::var("ORT_DYLIB_PATH").ok();
        // SAFETY: isolated test.
        unsafe {
            std::env::remove_var("ORT_DYLIB_PATH");
        }
        configure_onnx_runtime(std::path::Path::new("relative/path/libonnxruntime.so"));
        assert!(
            std::env::var("ORT_DYLIB_PATH").is_err(),
            "Relative path must not be written to env"
        );
        if let Some(v) = orig {
            // SAFETY: restoring original env var.
            unsafe {
                std::env::set_var("ORT_DYLIB_PATH", v);
            }
        }
    }
}
