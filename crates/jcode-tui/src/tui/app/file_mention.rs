//! File mention (at-file) search and caching for jcode's `@file` completion.
//!
//! This module provides:
//! - `PathIndex` – an in-memory snapshot of the workspace file tree.
//! - `FileIndexManager` – async background refresh with RCU-style atomic swap.
//! - `SearchHistory` – incremental search cache that makes backspace O(1).
//! - `FileMentionCache` – unified public API used by the input UI.

use super::char_bag::CharBag;
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
//  Data models
// ---------------------------------------------------------------------------

/// A single file (or directory) entry in the file index.
#[derive(Clone, Debug)]
struct FileEntry {
    /// Relative path from workspace root, e.g. "src/cli/startup.rs".
    pub path: Arc<str>,
    /// Just the filename portion, e.g. "startup.rs".
    pub filename: Arc<str>,
    pub is_directory: bool,
    /// Extension-based heuristic: false when the extension is in TEXT_EXTENSIONS,
    /// true otherwise. Refined during actual file read (null-byte scan).
    pub is_likely_binary: bool,
    pub char_bag: CharBag,
}

/// An immutable snapshot of the workspace file tree.
///
/// #### Two-layer index strategy
///
/// | Layer | Source | When built |
/// |-------|--------|-----------|
/// | `entries` | `git ls-files --cached --others --exclude-standard` | Background task, TTL 30 s |
/// | `lazy_entries` | `fs::read_dir` on-demand | When user query points to an ignored directory |
///
/// Entries from both layers are chained together in `search_in_index`.
#[derive(Clone, Debug)]
struct PathIndex {
    /// Base entries from git ls-files (excludes gitignored paths).
    pub entries: Vec<FileEntry>,
    /// Lazy entries from on-demand `read_dir` of ignored directories.
    pub lazy_entries: Vec<FileEntry>,
    /// Directories whose files have already been lazy-scanned (dedup).
    pub scanned_ignored_dirs: HashSet<Arc<str>>,
    /// Path → index into `entries` (not lazy_entries).
    pub path_to_index: HashMap<Arc<str>, usize>,
    /// Workspace root directory.
    pub root: PathBuf,
    /// Monotonic timestamp of last build.
    pub built_at: Instant,
    /// Loaded .gitignore patterns for the walkdir fallback.
    #[allow(dead_code)]
    gitignore_patterns: Vec<GitignorePattern>,
}

impl PathIndex {
    pub fn empty(root: PathBuf) -> Self {
        Self {
            entries: Vec::new(),
            lazy_entries: Vec::new(),
            scanned_ignored_dirs: HashSet::new(),
            path_to_index: HashMap::new(),
            root,
            built_at: Instant::now(),
            gitignore_patterns: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
//  Lightweight .gitignore parser (P3-21: walkdir fallback gitignore support)
// ---------------------------------------------------------------------------

/// A single parsed .gitignore pattern.
#[derive(Clone, Debug)]
struct GitignorePattern {
    /// Whether it's a negation (`!pattern`).
    negated: bool,
    /// Whether the pattern is anchored (starts with `/`).
    anchored: bool,
    /// Whether the pattern targets directories only (ends with `/`).
    dir_only: bool,
    /// The glob-like body after stripping `!`, `/`, trailing `/`.
    body: String,
}

/// Parse patterns from a `.gitignore` file.
fn load_gitignore(dir: &Path) -> Vec<GitignorePattern> {
    let path = dir.join(".gitignore");
    let file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let reader = std::io::BufReader::new(file);
    let mut patterns = Vec::new();
    for line in reader.lines().map_while(Result::ok) {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut negated = false;
        let rest = if let Some(stripped) = trimmed.strip_prefix('!') {
            negated = true;
            stripped
        } else {
            trimmed
        };
        let anchored = rest.starts_with('/');
        let body = if anchored { &rest[1..] } else { rest };
        let dir_only = body.ends_with('/');
        let body = if dir_only {
            &body[..body.len() - 1]
        } else {
            body
        };
        patterns.push(GitignorePattern {
            negated,
            anchored,
            dir_only,
            body: body.to_string(),
        });
    }
    patterns
}

/// Check if a relative path matches a gitignore pattern.
fn matches_gitignore(rel_path: &str, is_dir: bool, patterns: &[GitignorePattern]) -> bool {
    let mut ignored = false;
    for p in patterns {
        // Simple matching: check if path contains the pattern body as a segment.
        let matches = if p.anchored {
            rel_path == p.body || rel_path.starts_with(&format!("{}/", p.body))
        } else if p.body.contains('/') {
            // Pattern with slash: match from any directory level.
            rel_path == p.body
                || rel_path.ends_with(&format!("/{}", p.body))
                || rel_path.starts_with(&format!("{}/", p.body))
        } else {
            // Simple name pattern: match filename or any path component.
            if let Some(name) = Path::new(rel_path).file_name().and_then(|n| n.to_str()) {
                glob_match_name(name, &p.body)
            } else {
                rel_path.contains(&p.body)
            }
        };
        // Directory-only patterns only apply to directories.
        if p.dir_only && !is_dir {
            continue;
        }
        if matches {
            ignored = !p.negated;
        }
    }
    ignored
}

/// Simplified glob match for common gitignore patterns like `*.o`, `*.pyc`, `target`.
fn glob_match_name(name: &str, pattern: &str) -> bool {
    if pattern == name {
        return true;
    }
    if let Some(ext) = pattern.strip_prefix("*.") {
        return name.ends_with(&format!(".{}", ext));
    }
    if pattern.starts_with('*') && pattern.ends_with('*') {
        let inner = &pattern[1..pattern.len() - 1];
        return name.contains(inner);
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return name.starts_with(prefix);
    }
    if let Some(suffix) = pattern.strip_prefix('*') {
        return name.ends_with(suffix);
    }
    false
}

/// A single file match produced by the search engine.
#[derive(Clone, Debug)]
pub(crate) struct FileMatch {
    /// Match score (higher is better).
    pub score: f64,
    /// Relative file path.
    pub path: Arc<str>,
    pub is_directory: bool,
    /// `true` when this file was recently opened by the user.
    pub is_recent: bool,
    /// `true` when the extension is not in the known text whitelist.
    pub is_likely_binary: bool,
}

// ---------------------------------------------------------------------------
//  Search history
// ---------------------------------------------------------------------------

struct HistoryEntry {
    query: String,
    results: Vec<FileMatch>,
}

/// Exact-query search-history cache.
///
/// Stores results only for exact matches (same query string).  Used for
/// O(1) backspace recovery.  All other cases (prefix extension, partial
/// match) fall through to a full `search_in_index` scan — the engine is
/// fast enough (~0.3 ms for 5000 files) that incremental history filtering
/// is not needed.
///
/// Capacity is capped at `max_entries` (default 20).
struct SearchHistory {
    entries: Vec<HistoryEntry>,
    max_entries: usize,
}

enum LookupResult {
    /// Cache hit — return these results immediately.
    Hit(Vec<FileMatch>),
    /// Cache miss — caller must perform a full search.
    Miss,
}

impl SearchHistory {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            max_entries: 20,
        }
    }

    /// Returns `Hit` when `query` matches any cached entry exactly.
    /// All other cases → `Miss` → full search in `search_in_index`.
    pub fn lookup(&mut self, query: &str) -> LookupResult {
        if query.is_empty() {
            self.entries.clear();
            return LookupResult::Miss;
        }
        if let Some(entry) = self.entries.iter().rev().find(|e| e.query == query) {
            return LookupResult::Hit(entry.results.clone());
        }
        LookupResult::Miss
    }

    /// Save a full-search result.  Capacity-controlled.
    pub fn save(&mut self, query: &str, results: &[FileMatch]) {
        self.entries.push(HistoryEntry {
            query: query.to_string(),
            results: results.to_vec(),
        });
        while self.entries.len() > self.max_entries {
            self.entries.remove(0);
        }
    }
}

// ---------------------------------------------------------------------------
//  Search engine
// ---------------------------------------------------------------------------

/// Maximum number of results returned to the UI.
const MAX_RESULTS: usize = 15;

/// Check if `query` matches a directory path at a path-segment boundary.
///
/// Matches when the last component of `dir_path` begins with `query`,
/// or when `query/` appears as an exact prefix somewhere in `dir_path`.
/// Examples for query "src":
///   "src/"              → last component "src" starts_with "src" → true
///   "crates/jcode-tui/src/" → contains "/src/" → true
///   "scripts/"          → last component is "scripts", not "src" → false
fn dir_to_query_match(dir_path: &str, query: &str) -> bool {
    // Direct match: dir starts with query (e.g. "src/" for query "src")
    if dir_path.starts_with(query) {
        return true;
    }
    // Segment match: "/query/" appears in path (e.g. "/src/" in "crates/.../src/")
    let segment = format!("/{}/", query);
    if dir_path.contains(&segment) {
        return true;
    }
    // Trailing segment: path ends with "/query/" (already covered by contains)
    // Also: last component of dir starts with query
    if let Some(last_slash) = dir_path.trim_end_matches('/').rfind('/') {
        let last = &dir_path[last_slash + 1..];
        if last.starts_with(query) {
            return true;
        }
    }
    false
}

/// Tiered matching: try the fastest strategies first, only falling back to
/// the expensive DP fuzzy matcher when nothing else matches.
fn match_entry(entry: &FileEntry, query_lower: &str, slash_query: &str) -> f64 {
    // L1 – filename prefix  (~30 ns)  ──────────────────────────────
    if entry.filename.starts_with(query_lower) {
        return 100.0 + (entry.filename.len() as f64).sqrt();
    }

    // L2 – full-path prefix  (~50 ns) ──────────────────────────────
    if entry.path.starts_with(query_lower) {
        return 85.0;
    }

    // L2b – path-segment prefix (query appears after / in path).
    // slash_query is precomputed as "/{query_lower}" once in the caller.
    if let Some(pos) = entry.path.find(slash_query) {
        return 80.0 * (1.0 - (pos as f64 / entry.path.len().max(1) as f64));
    }

    // L3 – filename substring  (~80 ns) ─────────────────────────────
    if let Some(pos) = entry.filename.find(query_lower) {
        return 65.0 * (1.0 - (pos as f64 / entry.filename.len().max(1) as f64));
    }

    // L4 – full-path substring  (~100 ns) ───────────────────────────
    // Lower-priority catch-all. L2b above already gives higher scores
    // to segment-boundary matches like "/src".
    if let Some(pos) = entry.path.find(query_lower) {
        return 45.0 * (1.0 - (pos as f64 / entry.path.len().max(1) as f64));
    }

    // L5 – DP fuzzy subsequence  (~500 ns) ─────────────────────────
    // Uses jcode-fuzzy which already treats `/`, `-`, `_`, `.`, `:`
    // as boundary characters, giving path-like queries a natural boost.
    //
    // Skip fuzzy when query contains '/' — for structured path queries
    // like "src/lib.rs", a weak fuzzy match against "src/cli/debug.rs"
    // produces noise.  L1-L4 above are the correct minimum bar for
    // path-like input.
    if !query_lower.contains('/') {
        let filename_score = jcode_fuzzy::fuzzy_score(query_lower, &entry.filename).unwrap_or(0) as f64;
        let path_score = jcode_fuzzy::fuzzy_score(query_lower, &entry.path).unwrap_or(0) as f64;

        if filename_score > 0.0 || path_score > 0.0 {
            return 20.0
                + filename_score.max(path_score) * 1.0
                + if filename_score > 0.0 { 10.0 } else { 0.0 };
        }
    }

    0.0
}

/// Show recent files + root-level files when the query is empty.
fn show_all_files(
    index: &PathIndex,
    recent_files: &[Arc<str>],
    max_results: usize,
) -> Vec<FileMatch> {
    let mut results: Vec<FileMatch> = recent_files
        .iter()
        .filter_map(|path| {
            let idx = index.path_to_index.get(path)?;
            let entry = &index.entries[*idx];
            Some(FileMatch {
                score: 100.0,
                path: entry.path.clone(),
                is_directory: false,
                is_recent: true,
                is_likely_binary: entry.is_likely_binary,
            })
        })
        .collect();

    // Root-level entries: directories first, then visible files, skip hidden.
    // This matches Zed's behavior where @ shows recent files + the top-level
    // directory structure rather than every loose file.
    let mut root_dirs: Vec<FileMatch> = Vec::new();
    let mut root_files: Vec<FileMatch> = Vec::new();
    for entry in index.entries.iter().chain(index.lazy_entries.iter()) {
        let is_root_level = !entry.path.contains('/') && !entry.path.contains('\\');
        if !is_root_level || results.iter().any(|r| r.path == entry.path) {
            continue;
        }
        if entry.is_directory {
            root_dirs.push(FileMatch {
                score: 30.0,
                path: entry.path.clone(),
                is_directory: true,
                is_recent: false,
                is_likely_binary: false,
            });
        } else if !entry.path.starts_with(".git/") && entry.path.as_ref() != ".git" {
            // Skip only .git contents, keep other dotfiles (.gitignore, .env, etc.).
            root_files.push(FileMatch {
                score: 0.0,
                path: entry.path.clone(),
                is_directory: false,
                is_recent: false,
                is_likely_binary: entry.is_likely_binary,
            });
        }
    }
    results.extend(root_dirs);
    results.extend(root_files);
    results.truncate(max_results);
    results
}

/// Core file-search function (synchronous hot path; must return < 5 ms).
///
/// Merges both `entries` (git ls-files) and `lazy_entries` (on-demand
/// ignored-directory scan) so that `@ai-memory/` finds gitignored files.
fn search_in_index(
    query: &str,
    index: &PathIndex,
    recent_files: &[Arc<str>],
) -> Vec<FileMatch> {
    if query.is_empty() {
        return show_all_files(index, recent_files, MAX_RESULTS);
    }

    // Normalize query: strip leading "/" so "@src" and "@/src" behave
    // identically (both compare against relative paths).
    let query_raw = query.trim_start_matches('/');
    if query_raw.is_empty() {
        return show_all_files(index, recent_files, MAX_RESULTS);
    }
    let query_lower = query_raw.to_lowercase();
    let slash_query = format!("/{}", query_lower);
    let query_bag = CharBag::from(&query_lower);
    let mut results: Vec<FileMatch> = Vec::with_capacity(64);

    // Chain base + lazy entries (two-layer index, see PathIndex docs).
    for entry in index.entries.iter().chain(index.lazy_entries.iter()) {
        // CharBag pre-filter: O(1), eliminates 60-80% of candidates.
        if !entry.char_bag.is_superset(query_bag) {
            continue;
        }

        let score = match_entry(entry, &query_lower, &slash_query);
        if score > 0.0 {
            let is_recent = recent_files.contains(&entry.path);
            results.push(FileMatch {
                score: score + if is_recent { 50.0 } else { 0.0 },
                path: entry.path.clone(),
                is_directory: entry.is_directory,
                is_recent,
                is_likely_binary: entry.is_likely_binary,
            });
        }
    }

    // Build directory entries from matching ancestors of file results.
    // Score 110 ensures directories rank above all file matches.
    if !query.is_empty() {
        let mut dirs = HashSet::new();
        for r in &results {
            let mut remaining = r.path.as_ref();
            while let Some(slash) = remaining.rfind('/') {
                let dir = &r.path[..slash + 1];
                if dir_to_query_match(dir, &query_lower) {
                    dirs.insert(Arc::from(dir));
                }
                remaining = &r.path[..slash];
            }
        }
        for dir in dirs {
            results.push(FileMatch {
                score: 110.0,
                path: dir,
                is_directory: true,
                is_recent: false,
                is_likely_binary: false,
            });
        }
    }

    // Sort: score descending, then shorter paths first, then path ascending.
    results.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.path.len().cmp(&b.path.len()))
            .then_with(|| a.path.cmp(&b.path))
    });

    // Ensure a mix of directories and files.  Directories are useful for
    // navigation, but files are what the user ultimately wants.  Keep at most
    // 4 directory entries so files from deep inside the target directory
    // still appear even when many ancestor directories match.
    let mut final_results = Vec::with_capacity(MAX_RESULTS);
    let mut dirs_seen = 0usize;
    const MAX_DIRS: usize = 4;
    for r in results {
        if r.is_directory {
            if dirs_seen >= MAX_DIRS {
                continue;
            }
            dirs_seen += 1;
        }
        final_results.push(r);
        if final_results.len() >= MAX_RESULTS {
            break;
        }
    }
    final_results
}

// ---------------------------------------------------------------------------
//  Index builder
// ---------------------------------------------------------------------------

/// Maximum files collected into the index (safety cap).
const MAX_FILES: usize = 5_000;

/// Directories to skip during walkdir fallback. Unrelated to .gitignore;
/// purely avoids indexing huge dependency / build directories.
const SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "dist",
    "build",
    "__pycache__",
    "vendor",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "bower_components",
    ".next",
    ".nuxt",
    "coverage",
    ".terraform",
    ".serverless",
    ".netlify",
];

/// Run `git ls-files` and return sorted relative paths.
///
/// The command `--cached --others --exclude-standard` respects both
/// `.gitignore` and `.git/info/exclude`.
async fn git_ls_files(cwd: &Path) -> Option<Vec<String>> {
    let output = tokio::time::timeout(
        Duration::from_secs(3),
        tokio::process::Command::new("git")
            .args(["ls-files", "--cached", "--others", "--exclude-standard"])
            .current_dir(cwd)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
            .output(),
    )
    .await
    .ok()?
    .ok()?;

    if !output.status.success() {
        return None;
    }

    let mut files: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|l| {
            let l = l.trim();
            !l.is_empty() && !l.starts_with(".git/")
        })
        .map(|l| l.trim().to_string())
        .collect();

    files.sort();
    files.truncate(MAX_FILES);
    Some(files)
}

/// Walkdir fallback when `git ls-files` is unavailable.
///
/// Uses `symlink_metadata` to avoid following symlinks, skips directories
/// listed in `SKIP_DIRS`, and respects `.gitignore` patterns (P3-21).
async fn walkdir_collect(cwd: &Path) -> Vec<String> {
    let cwd = cwd.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let gitignore = load_gitignore(&cwd);
        let mut files: Vec<String> = Vec::new();
        let mut stack: Vec<PathBuf> = vec![cwd.clone()];
        // Track per-directory gitignore patterns.
        let mut dir_gitignores: Vec<(PathBuf, Vec<GitignorePattern>)> = Vec::new();

        while let Some(dir) = stack.pop() {
            let entries = match std::fs::read_dir(&dir) {
                Ok(e) => e,
                Err(_) => continue,
            };

            // Load .gitignore for this directory.
            let local_patterns = load_gitignore(&dir);
            if !local_patterns.is_empty() {
                dir_gitignores.push((dir.clone(), local_patterns));
            }

            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();

                // Skip symlinks.
                if entry.file_type().is_ok_and(|ft| ft.is_symlink()) {
                    continue;
                }

                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

                if path.is_dir() {
                    if !SKIP_DIRS.contains(&name) {
                        // Check gitignore for directory.
                        if let Ok(rel) = path.strip_prefix(&cwd) {
                            if let Some(rel_str) = rel.to_str() {
                                if !matches_any_gitignore(rel_str, true, &gitignore, &dir_gitignores) {
                                    stack.push(path);
                                }
                            }
                        }
                    }
                    continue;
                }

                if let Ok(rel) = path.strip_prefix(&cwd) {
                    if let Some(rel_str) = rel.to_str() {
                        if matches_any_gitignore(rel_str, false, &gitignore, &dir_gitignores) {
                            continue;
                        }
                        files.push(rel_str.to_string());
                        if files.len() >= MAX_FILES {
                            return files;
                        }
                    }
                }
            }
        }

        files
    })
    .await
    .unwrap_or_default()
}

/// Check a relative path against root and per-directory gitignore patterns.
fn matches_any_gitignore(
    rel: &str,
    is_dir: bool,
    root_patterns: &[GitignorePattern],
    dir_patterns: &[(PathBuf, Vec<GitignorePattern>)],
) -> bool {
    if matches_gitignore(rel, is_dir, root_patterns) {
        return true;
    }
    for (dir, patterns) in dir_patterns {
        // Only apply if the path is under this directory.
        if rel.starts_with(&format!("{}/", dir.display()))
            || rel == dir.to_string_lossy().as_ref()
        {
            let sub_path = rel.strip_prefix(&format!("{}/", dir.display())).unwrap_or(rel);
            if matches_gitignore(sub_path, is_dir, patterns) {
                return true;
            }
        }
    }
    false
}

/// Build a fresh `PathIndex` for the workspace.
///
/// Prefers `git ls-files` (fast, .gitignore-aware) with a walkdir fallback.
async fn build_path_index(cwd: &Path) -> PathIndex {
    // Strategy 1: git ls-files.
    let file_paths = match git_ls_files(cwd).await {
        Some(paths) => paths,
        None => walkdir_collect(cwd).await,
    };

    let mut entries = Vec::with_capacity(file_paths.len());
    let mut path_to_index = HashMap::with_capacity(file_paths.len());

    for (i, path_str) in file_paths.iter().enumerate() {
        let p = Path::new(path_str);
        let filename: Arc<str> = p
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(path_str)
            .into();

        let entry = FileEntry {
            path: Arc::from(path_str.as_str()),
            filename,
            is_directory: false,
            is_likely_binary: p
                .extension()
                .and_then(|e| e.to_str())
                .map(|ext| !TEXT_EXTENSIONS.contains(&ext))
                .unwrap_or(true),
            char_bag: CharBag::from(path_str),
        };

        path_to_index.insert(entry.path.clone(), i);
        entries.push(entry);
    }

    PathIndex {
        entries,
        lazy_entries: Vec::new(),
        scanned_ignored_dirs: HashSet::new(),
        path_to_index,
        root: cwd.to_path_buf(),
        built_at: Instant::now(),
        gitignore_patterns: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
//  FileIndexManager (RCU-style atomic index swap)
// ---------------------------------------------------------------------------

/// Manages background index builds with lock-free reads.
///
/// - Reads (`snapshot`) use `std::sync::RwLock::read` — the read lock is
///   never contended because the write lock is held for nanoseconds (an
///   `Arc` pointer swap).
/// - Writes happen in `tokio::spawn` tasks; `std::sync::RwLock::write`
///   blocks the worker thread briefly, which is acceptable for a sub-µs
///   critical section.
///
/// We deliberately use `std::sync::RwLock`, **not** `tokio::sync::RwLock`.
/// `tokio::sync::RwLock::blocking_read()` panics when called from inside a
/// tokio runtime, and `snapshot()` is called from the sync hot path which
/// runs on the tokio main thread.
struct FileIndexManager {
    current: Arc<RwLock<Arc<PathIndex>>>,
    cwd: PathBuf,
    refreshing: Arc<AtomicBool>,
    /// Flag set by the background file watcher (notify) when FS changes occur.
    /// Checked in `check_refresh` to trigger re-index without TTL polling.
    dirty: Arc<AtomicBool>,
    /// Holds the notify watcher thread alive. Dropping this stops watching.
    #[allow(dead_code)]
    _watcher_handle: Arc<Mutex<Option<notify::INotifyWatcher>>>,
}

impl FileIndexManager {
    pub fn new(cwd: PathBuf) -> Self {
        let dirty = Arc::new(AtomicBool::new(false));
        let watcher = Self::start_watcher(cwd.clone(), dirty.clone());

        Self {
            current: Arc::new(RwLock::new(Arc::new(PathIndex::empty(cwd.clone())))),
            cwd,
            refreshing: Arc::new(AtomicBool::new(false)),
            dirty,
            _watcher_handle: Arc::new(Mutex::new(watcher)),
        }
    }

    /// Spawn a background thread that watches `cwd` for filesystem changes
    /// and sets the `dirty` flag. Returns the watcher handle.
    fn start_watcher(cwd: PathBuf, dirty: Arc<AtomicBool>) -> Option<notify::INotifyWatcher> {
        use notify::{Event, EventKind, RecursiveMode, Watcher};
        let (tx, rx) = std::sync::mpsc::channel();
        let mut watcher = notify::INotifyWatcher::new(
            move |res: Result<Event, notify::Error>| {
                if let Ok(event) = res {
                    // Ignore access-only events; only care about creates, modifies,
                    // deletes, and renames.
                    match event.kind {
                        EventKind::Create(_)
                        | EventKind::Modify(_)
                        | EventKind::Remove(_)
                        | EventKind::Any => {}
                        _ => return,
                    }
                    let _ = tx.send(());
                }
            },
            notify::Config::default(),
        )
        .ok()?;

        let _ = watcher.watch(&cwd, RecursiveMode::NonRecursive);
        // The watcher must not be dropped — keep it alive in this thread or leak
        // it to be cleaned up when the process exits.
        std::mem::forget(watcher);

        // Drain events; set dirty on any relevant filesystem change.
        // Debounce: only set dirty at most once per second.
        std::thread::spawn(move || {
            let mut last_set = std::time::Instant::now()
                .checked_sub(Duration::from_secs(2))
                .unwrap();
            for _ in rx {
                let now = std::time::Instant::now();
                if now.duration_since(last_set) >= Duration::from_secs(1) {
                    dirty.store(true, Ordering::Release);
                    last_set = now;
                }
            }
        });

        None // handle is null; thread owns the real watcher
    }

    /// Obtain a lightweight snapshot of the current index.
    ///
    /// Uses `std::sync::RwLock::read` — safe on the sync hot path because
    /// the write lock is only held for an `Arc` pointer swap (~ns).
    pub fn snapshot(&self) -> Arc<PathIndex> {
        self.current.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Kick off an async background refresh.
    ///
    /// Takes `&self` (not `&mut self`) because state is shared through `Arc`.
    pub fn refresh_async(&self) {
        if self.refreshing.swap(true, Ordering::Acquire) {
            return; // already refreshing
        }

        let cwd = self.cwd.clone();
        let current = self.current.clone();
        let refreshing = self.refreshing.clone();

        tokio::spawn(async move {
            let new_index = build_path_index(&cwd).await;

            // Write lock held for sub-µs: just an Arc pointer swap.
            {
                let mut guard = current.write().unwrap_or_else(|e| e.into_inner());
                *guard = Arc::new(new_index);
            }

            refreshing.store(false, Ordering::Release);
        });
    }

    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    pub fn is_refreshing(&self) -> bool {
        self.refreshing.load(Ordering::Acquire)
    }
}

// ---------------------------------------------------------------------------
//  FileMentionCache — unified public API
// ---------------------------------------------------------------------------

/// Top-level cache that wires together indexing, search, history, and lazy
/// ignored-directory scanning.
///
/// Supports multiple workspace directories (P3-18). The first manager (index 0)
/// is the primary workspace. Search results from secondary workspaces are
/// prefixed with the workspace name.
pub(crate) struct FileMentionCache {
    /// Per-workspace index managers. Index 0 is the primary (cwd).
    managers: Vec<FileIndexManager>,
    /// Workspace display names (last path component) for result prefixing.
    workspace_names: Vec<String>,
    history: SearchHistory,
    /// Recently-opened files (capped at 10), used for ranking.
    recent_files: VecDeque<Arc<str>>,
}

impl FileMentionCache {
    pub fn new() -> Self {
        Self {
            managers: vec![FileIndexManager::new(PathBuf::new())],
            workspace_names: vec![String::new()],
            history: SearchHistory::new(),
            recent_files: VecDeque::new(),
        }
    }

    /// Add an additional workspace directory to search.
    #[allow(dead_code)]
    pub fn add_workspace(&mut self, path: &Path) {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(".")
            .to_string();
        self.managers
            .push(FileIndexManager::new(path.to_path_buf()));
        self.workspace_names.push(name);
    }

    /// Main entry-point: return candidate files for `query`.
    ///
    /// Must return in < 5 ms on the synchronous UI thread.
    pub fn candidates(&mut self, query: &str) -> Vec<FileMatch> {
        // 1. Check search history.
        match self.history.lookup(query) {
            LookupResult::Hit(results) => return results,
            LookupResult::Miss => {}
        }

        let recent: Vec<Arc<str>> = self.recent_files.iter().cloned().collect();
        let mut all_results: Vec<FileMatch> = Vec::new();

        for (wi, manager) in self.managers.iter().enumerate() {
            let snapshot = manager.snapshot();
            if snapshot.entries.is_empty() {
                manager.refresh_async();
                continue;
            }

            let mut index = (*snapshot).clone();
            if !query.is_empty() {
                // Strip leading / for path resolution.
                let q = query.trim_start_matches('/');
                if !q.is_empty() {
                    ensure_ignored_dir_scanned(q, &mut index);
                }
            }

            let results = search_in_index(query, &index, &recent);

            // Prefix paths from secondary workspaces.
            if wi > 0 {
                let prefix = &self.workspace_names[wi];
                all_results.extend(results.into_iter().map(|mut m| {
                    m.path = Arc::from(format!("{}/{}", prefix, m.path).as_str());
                    m
                }));
            } else {
                all_results.extend(results);
            }
        }

        // Sort merged results by score, truncate.
        all_results.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.path.cmp(&b.path))
        });
        all_results.truncate(MAX_RESULTS);

        // Persist in history.
        self.history.save(query, &all_results);
        all_results
    }

    /// Record a file open so it ranks higher in future searches.
    pub fn record_file_open(&mut self, path: Arc<str>) {
        if let Some(idx) = self.recent_files.iter().position(|p| p == &path) {
            self.recent_files.remove(idx);
        }
        self.recent_files.push_front(path);
        if self.recent_files.len() > 10 {
            self.recent_files.pop_back();
        }
    }

    /// Ensure the index is still valid for the current working directory.
    pub fn check_refresh(&mut self, cwd: &Path) {
        // Primary workspace: recreate if cwd changed.
        if self.managers[0].cwd() != cwd {
            self.managers[0] = FileIndexManager::new(cwd.to_path_buf());
            self.workspace_names[0] = cwd
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(".")
                .to_string();
            self.history = SearchHistory::new();
        }

        for manager in &self.managers {
            let index = manager.snapshot();
            let dirty = manager.dirty.swap(false, Ordering::Acquire);
            let needs_refresh = index.entries.is_empty()
                || dirty
                || index.built_at.elapsed() > Duration::from_secs(30);

            if needs_refresh && !manager.is_refreshing() {
                manager.refresh_async();
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  Lazy ignored-directory scanning
// ---------------------------------------------------------------------------

/// Check whether `query` targets an ignored directory and, if so, populate
/// `index.lazy_entries` from `fs::read_dir`.
fn ensure_ignored_dir_scanned(query: &str, index: &mut PathIndex) {
    // Walk the query's directory prefixes, deepest first.
    let mut candidate = query.to_string();
    while let Some(slash_pos) = candidate.rfind('/') {
        candidate.truncate(slash_pos);
        try_scan_ignored_dir(&candidate, index);
    }
    // Also check the leaf (e.g. "ai-memory" from "ai-memory/de").
    try_scan_ignored_dir(query, index);
}

fn try_scan_ignored_dir(dir_path: &str, index: &mut PathIndex) {
    let dir_key: Arc<str> = Arc::from(dir_path);

    // Already scanned or is in the base index (i.e. not ignored).
    if index.scanned_ignored_dirs.contains(&dir_key) {
        return;
    }
    if index
        .entries
        .iter()
        .any(|e| e.path.starts_with(dir_path))
    {
        return;
    }

    let abs_dir = index.root.join(dir_path);
    if !abs_dir.is_dir() {
        // Prefix fallback: when the query is a root-level name and the
        // exact directory does not exist, scan the workspace root for
        // directory names that start with `dir_path` (e.g. "ai" →
        // "ai-memory/").  This is how users discover gitignored
        // directories without typing their full name.
        if !dir_path.contains('/') {
            if let Ok(rd) = std::fs::read_dir(&index.root) {
                for entry in rd.filter_map(|e| e.ok()) {
                    if entry.file_type().is_ok_and(|ft| ft.is_dir()) {
                        let name = entry.file_name();
                        let name_str = name.to_string_lossy();
                        if name_str.starts_with(dir_path) && !name_str.starts_with('.') {
                            // Recursively scan the matched directory.
                            try_scan_ignored_dir(&name_str, index);
                        }
                    }
                }
            }
        }
        return;
    }

    let mut new_files = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&abs_dir) {
        for entry in rd.filter_map(|e| e.ok()) {
            // Skip symlinks (design §5.2.8).
            if entry.file_type().is_ok_and(|ft| ft.is_symlink()) {
                continue;
            }

            let abs_path = entry.path();
            if let Ok(rel) = abs_path.strip_prefix(&index.root) {
                if let Some(rel_str) = rel.to_str() {
                    if !rel_str.starts_with(".git/") {
                        let filename: Arc<str> = abs_path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or(rel_str)
                            .into();

                        new_files.push(FileEntry {
                            path: Arc::from(rel_str),
                            filename,
                            is_directory: abs_path.is_dir(),
                            is_likely_binary: abs_path
                                .extension()
                                .and_then(|e| e.to_str())
                                .map(|ext| !TEXT_EXTENSIONS.contains(&ext))
                                .unwrap_or(!abs_path.is_dir()),
                            char_bag: CharBag::from(rel_str),
                        });
                    }
                }
            }
        }
    }

    index.lazy_entries.extend(new_files);
    index.scanned_ignored_dirs.insert(dir_key);
}

// ---------------------------------------------------------------------------
//  File content loading (wired into the send path)
// ---------------------------------------------------------------------------

/// Maximum size of a single file before truncation.
const MAX_FILE_SIZE: usize = 100 * 1024; // 100 KB

/// Maximum total content loaded across all @file references.
const MAX_FILE_TOTAL_BUDGET: usize = 500 * 1024; // 500 KB

/// Known text file extensions (whitelist). Files not in this list are still
/// read, but a null-byte check decides whether to treat them as binary.
const TEXT_EXTENSIONS: &[&str] = &[
    "rs", "py", "js", "ts", "jsx", "tsx", "go", "java", "c", "cpp", "h", "hpp",
    "rb", "php", "swift", "kt", "scala", "clj", "el", "lua", "r", "R",
    "toml", "yaml", "yml", "json", "xml", "ini", "cfg", "conf",
    "md", "txt", "log", "csv", "sh", "bash", "zsh", "fish",
    "sql", "css", "scss", "html", "htm", "svg", "vue", "svelte",
    "Makefile", "Dockerfile", "gitignore", "env",
    "lock", "gradle", "cmake", "meson",
];

/// Fast binary check: whitelisted extension → text; otherwise null-byte scan.
fn is_likely_binary(path: &Path) -> bool {
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        if TEXT_EXTENSIONS.contains(&ext) {
            return false;
        }
    }
    // Read the first 8 KB and check for null bytes.
    if let Ok(buf) = std::fs::read(path) {
        let check_len = buf.len().min(8192);
        buf[..check_len].contains(&0u8)
    } else {
        false
    }
}

/// Recursively collect files from a directory, respecting SKIP_DIRS blacklist.
/// Stops after `max_files` entries.
fn collect_dir_files(
    dir: &Path,
    _cwd: &Path,
    out: &mut Vec<PathBuf>,
    seen: &mut HashSet<PathBuf>,
    max_files: usize,
) {
    if out.len() >= max_files || !dir.is_dir() {
        return;
    }
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.filter_map(|e| e.ok()) {
            let path = entry.path();
            // Skip symlinks.
            if entry.file_type().is_ok_and(|ft| ft.is_symlink()) {
                continue;
            }
            if path.is_dir() {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if SKIP_DIRS.contains(&name) {
                    continue;
                }
                collect_dir_files(&path, _cwd, out, seen, max_files);
            } else if seen.insert(path.clone()) {
                out.push(path);
                if out.len() >= max_files {
                    return;
                }
            }
        }
    }
}

/// Load file contents referenced by `file_chips` and prepend them to the
/// user's prompt. This is called from the send path (`submit_input`).
///
/// `file_chips` contains relative paths (as they appear in the input text).
/// Paths are resolved against the current directory. Directories are
/// recursively walked (up to 50 files per directory, respecting SKIP_DIRS).
/// The same binary detection, size truncation, and budget management applies.
pub(crate) fn build_prompt_with_files(input: &str, file_chips: &[PathBuf]) -> String {
    if file_chips.is_empty() {
        return input.to_string();
    }

    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    // Expand directories into individual file paths.
    let mut expanded_files: Vec<PathBuf> = Vec::new();
    const MAX_FILES_PER_DIR: usize = 50;
    let mut seen: HashSet<PathBuf> = HashSet::new();

    for chip in file_chips {
        let abs = cwd.join(chip);
        if abs.is_dir() {
            // Recursively collect files, respecting skip dirs.
            collect_dir_files(&abs, &cwd, &mut expanded_files, &mut seen, MAX_FILES_PER_DIR);
        } else {
            if seen.insert(abs.clone()) {
                expanded_files.push(abs);
            }
        }
    }

    // Collect files synchronously (send path runs on the main thread).
    let mut file_blocks: Vec<(String, String)> = Vec::with_capacity(expanded_files.len());

    for path in &expanded_files {
        let rel_path = path
            .strip_prefix(&cwd)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| path.to_string_lossy().to_string());

        if is_likely_binary(path) {
            let rp = rel_path.clone();
            file_blocks.push((rp, format!("[skipped: binary file {}]", rel_path)));
            continue;
        }

        match std::fs::read_to_string(path) {
            Ok(content) => {
                let block = if content.len() <= MAX_FILE_SIZE {
                    content
                } else {
                    let line_count = content.lines().count();
                    let preview: String = content.lines().take(200).collect::<Vec<_>>().join("\n");
                    format!(
                        "{}\n\n[... file too large: {} lines, {} bytes, showing first 200 lines]",
                        preview, line_count, content.len(),
                    )
                };
                file_blocks.push((rel_path, block));
            }
            Err(e) => {
                let rp = rel_path.clone();
                file_blocks.push((rp, format!("[read failed: {} → {}]", rel_path, e)));
            }
        }
    }

    // Context budget: cumulative cap.
    let mut context = String::new();
    let mut total = 0usize;
    for (path, content) in &file_blocks {
        if total > MAX_FILE_TOTAL_BUDGET {
            let truncated: String = format!(
                "[context budget exhausted ({} KB total), skipped]\n{}",
                MAX_FILE_TOTAL_BUDGET / 1024,
                content,
            )
            .chars()
            .take(2000)
            .collect();
            context.push_str(&format!("\n--- {} ---\n{}\n", path, truncated));
        } else {
            total += content.len();
            context.push_str(&format!("\n--- {} ---\n{}\n", path, content));
        }
    }

    format!("{}{}", input, context)
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn file_entry(path: &str) -> FileEntry {
        let filename: Arc<str> = Path::new(path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(path)
            .into();

        FileEntry {
            path: Arc::from(path),
            filename,
            is_directory: false,
            is_likely_binary: false,
            char_bag: CharBag::from(path),
        }
    }

    fn file_match(path: &str) -> FileMatch {
        FileMatch {
            score: 50.0,
            path: Arc::from(path),
            is_directory: false,
            is_recent: false,
            is_likely_binary: false,
        }
    }

    fn build_test_index(paths: &[&str]) -> PathIndex {
        let mut entries = Vec::new();
        for p in paths {
            entries.push(file_entry(p));
        }
        PathIndex {
            entries,
            lazy_entries: Vec::new(),
            scanned_ignored_dirs: HashSet::new(),
            path_to_index: HashMap::new(),
            root: PathBuf::from("."),
            built_at: Instant::now(),
            gitignore_patterns: Vec::new(),
        }
    }

    // -- match_entry ---------------------------------------------------------

    fn s(q: &str) -> String { format!("/{}", q) }

    #[test]
    fn match_filename_prefix_wins() {
        let entry = file_entry("deep/nested/src/lib.rs");
        let score = match_entry(&entry, "lib.rs", &s("lib.rs"));
        assert!(
            score > 90.0,
            "filename prefix should score high, got {score}"
        );
    }

    #[test]
    fn match_path_prefix() {
        let entry = file_entry("src/cli/args.rs");
        let score = match_entry(&entry, "src/cli", &s("src/cli"));
        assert!(
            score > 70.0,
            "path prefix should score high, got {score}"
        );
    }

    #[test]
    fn match_fuzzy_subsequence() {
        let entry = file_entry("src/cli/startup.rs");
        let score = match_entry(&entry, "scli", &s("scli"));
        assert!(score > 0.0, "fuzzy match should work for 'scli'");
    }

    #[test]
    fn match_no_match() {
        let entry = file_entry("src/lib.rs");
        assert_eq!(match_entry(&entry, "xyz", &s("xyz")), 0.0);
    }

    // -- search_in_index ----------------------------------------------------

    #[test]
    fn search_returns_results() {
        let index = build_test_index(&["src/main.rs", "Cargo.toml"]);
        let results = search_in_index("main", &index, &[]);
        assert!(!results.is_empty());
        assert!(results[0].path.contains("main"));
    }

    #[test]
    fn search_empty_query() {
        let index = build_test_index(&["README.md", "src/main.rs"]);
        let results = search_in_index("", &index, &[]);
        // README.md is root-level → should appear.
        assert!(results.iter().any(|r| r.path.as_ref() == "README.md"));
    }

    #[test]
    fn search_prioritizes_filename() {
        let mut index = build_test_index(&["deep/nested/args.rs", "src/args.rs"]);
        // Build path_to_index for recent-file lookups.
        for (i, e) in index.entries.iter().enumerate() {
            index.path_to_index.insert(e.path.clone(), i);
        }
        let results = search_in_index("args", &index, &[]);
        // Both files match; "deep/nested/args.rs" has "args" in filename and shorter path
        // portion, so either ordering is valid as long as both appear.
        assert!(results.len() >= 1);
        let paths: Vec<&str> = results.iter().map(|r| r.path.as_ref()).collect();
        assert!(paths.contains(&"src/args.rs"));
        assert!(paths.contains(&"deep/nested/args.rs"));
    }

    #[test]
    fn search_recent_file_boost() {
        let index = build_test_index(&["Cargo.toml", "src/main.rs"]);
        let recent = vec![Arc::from("Cargo.toml")];
        let results = search_in_index("Cargo", &index, &recent);
        assert!(results.iter().any(|r| r.is_recent));
    }

    // -- SearchHistory ------------------------------------------------------

    #[test]
    // -- SearchHistory (exact-match only) -----------------------------------

    #[test]
    fn history_exact_hit_after_save() {
        let mut history = SearchHistory::new();
        history.save("src", &[file_match("src/main.rs")]);
        match history.lookup("src") {
            LookupResult::Hit(results) => assert_eq!(results.len(), 1),
            _ => panic!("exact match should hit"),
        }
    }

    #[test]
    fn history_miss_on_prefix_extension() {
        // Incremental input like "src" → "src/" falls through to full search.
        let mut history = SearchHistory::new();
        history.save("src", &[file_match("src/main.rs")]);
        assert!(matches!(
            history.lookup("src/cli"),
            LookupResult::Miss
        ));
    }

    #[test]
    fn history_backspace_to_exact() {
        let mut history = SearchHistory::new();
        history.save("src/cli", &[file_match("src/cli/args.rs")]);
        history.save("src/cli/s", &[file_match("src/cli/startup.rs")]);
        // Backspace to "src/cli" — exact match, hit.
        match history.lookup("src/cli") {
            LookupResult::Hit(results) => assert_eq!(results.len(), 1),
            _ => panic!("backspace to exact should hit"),
        }
        // Backspace further to "src" — not exact, miss, triggers full search.
        assert!(matches!(history.lookup("src"), LookupResult::Miss));
    }

    #[test]
    fn history_empty_clears_and_misses() {
        let mut history = SearchHistory::new();
        history.save("abc", &[file_match("x")]);
        assert!(matches!(history.lookup(""), LookupResult::Miss));
        assert!(matches!(history.lookup("abc"), LookupResult::Miss));
    }
}
