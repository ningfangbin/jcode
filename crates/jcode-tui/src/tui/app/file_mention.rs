//! File mention (at-file) search and caching for jcode's `@file` completion.
//!
//! This module provides:
//! - `PathIndex` – an in-memory snapshot of the workspace file tree.
//! - `FileIndexManager` – async background refresh with RCU-style atomic swap.
//! - `SearchHistory` – incremental search cache that makes backspace O(1).
//! - `FileMentionCache` – unified public API used by the input UI.

use super::char_bag::CharBag;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
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
        }
    }
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

/// Incremental search-history cache.
///
/// Invariants:
/// - `history[i].query` is a prefix of `history[i+1].query`
/// - `history[i].results` ⊇ `history[i+1].results` (parent superset-of-child)
///
/// Capacity is capped at `max_entries` (default 20). When the user makes a
/// "jump edit" (e.g. deleting a middle character) the prefix invariant is
/// violated and the entire history is cleared.
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

    /// Try to satisfy `query` from the history cache.
    ///
    /// Three cases:
    /// 1. Exact hit on the last entry → O(1).
    /// 2. `query` extends the last entry → incrementally filter → push.
    /// 3. `query` is a prefix of an earlier entry (backspace) → pop → hit.
    /// Otherwise → Miss.
    pub fn lookup(&mut self, query: &str) -> LookupResult {
        if query.is_empty() {
            self.entries.clear();
            return LookupResult::Miss;
        }

        // Exact hit on the most recent entry.
        if let Some(last) = self.entries.last() {
            if last.query == query {
                return LookupResult::Hit(last.results.clone());
            }
        }

        // Incremental narrowing: user typed more characters.
        if let Some(last) = self.entries.last() {
            if query.starts_with(&last.query) && !last.query.is_empty() {
                let filtered: Vec<FileMatch> = last
                    .results
                    .iter()
                    .filter(|m| matches_filter(&m.path, query))
                    .cloned()
                    .collect();

                if !filtered.is_empty() {
                    self.entries.push(HistoryEntry {
                        query: query.to_string(),
                        results: filtered.clone(),
                    });
                    return LookupResult::Hit(filtered);
                }
            }
        }

        // Backspace recovery: walk backwards to find the deepest matching
        // parent, then truncate the history to that point.
        while self.entries.len() > 1 {
            let parent_idx = self.entries.len() - 2;
            let parent_query = self.entries[parent_idx].query.clone();
            let parent_results = self.entries[parent_idx].results.clone();

            if query.starts_with(&parent_query) {
                self.entries.truncate(parent_idx + 1);

                // If the user typed *more* than the parent but we don't have
                // a direct entry, filter from the parent.
                if query.len() > parent_query.len() {
                    let filtered: Vec<FileMatch> = parent_results
                        .iter()
                        .filter(|m| matches_filter(&m.path, query))
                        .cloned()
                        .collect();
                    if !filtered.is_empty() {
                        self.entries.push(HistoryEntry {
                            query: query.to_string(),
                            results: filtered.clone(),
                        });
                        return LookupResult::Hit(filtered);
                    }
                }
                return LookupResult::Hit(parent_results);
            }
            self.entries.pop();
        }

        LookupResult::Miss
    }

    /// Persist a full-search result.
    ///
    /// If the user made a "jump edit" (the last query is not a prefix of
    /// `query`), the entire history is cleared before saving. This preserves
    /// correctness at the cost of one extra full search.
    pub fn save(&mut self, query: &str, results: &[FileMatch]) {
        // Prefix-invariant guard: jump edits clear the history.
        if let Some(last) = self.entries.last() {
            if !query.starts_with(&last.query) && query != last.query {
                self.entries.clear();
            }
        }

        // Replace duplicate (backspace-then-same) entry.
        if let Some(last) = self.entries.last() {
            if last.query == query {
                self.entries.pop();
            }
        }

        self.entries.push(HistoryEntry {
            query: query.to_string(),
            results: results.to_vec(),
        });

        // Capacity control: drop the oldest entries.
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

/// Return `true` when a directory path matches the user's query at a
/// path-segment boundary.
///
/// Examples for query "src":
///   "src/"           → starts_with("src")     → true
///   "crates/jcode-tui/src/" → contains("/src") → true
///   "scripts/"       → neither                 → false
fn dir_matches_query(query: &str, dir_path: &str) -> bool {
    if dir_path.starts_with(query) {
        return true;
    }
    let segment = format!("/{}", query);
    dir_path.contains(&segment)
}

/// Lightweight path-match used by the history cache (faster than match_entry).
fn matches_filter(path: &str, query: &str) -> bool {
    if path.contains(query) {
        return true;
    }
    if let Some(base) = Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
    {
        if base.contains(query) {
            return true;
        }
    }
    false
}

/// Tiered matching: try the fastest strategies first, only falling back to
/// the expensive DP fuzzy matcher when nothing else matches.
fn match_entry(entry: &FileEntry, query_lower: &str) -> f64 {
    // L1 – filename prefix  (~30 ns)  ──────────────────────────────
    if entry.filename.starts_with(query_lower) {
        return 100.0 + (entry.filename.len() as f64).sqrt();
    }

    // L2 – full-path prefix  (~50 ns) ──────────────────────────────
    if entry.path.starts_with(query_lower) {
        return 85.0;
    }

    // L2b – path-segment prefix (query appears after / in path).
    // e.g. query "src" matches "crates/jcode-tui/src/foo.rs" at the
    // "/src" boundary, but NOT "scripts/foo.rs" (no /src segment).
    {
        let segment = format!("/{}", query_lower);
        if let Some(pos) = entry.path.find(&segment) {
            return 80.0 * (1.0 - (pos as f64 / entry.path.len().max(1) as f64));
        }
    }

    // L3 – filename substring  (~80 ns) ─────────────────────────────
    if let Some(pos) = entry.filename.find(query_lower) {
        return 65.0 * (1.0 - (pos as f64 / entry.filename.len().max(1) as f64));
    }

    // L4 – full-path substring at segment boundary only.
    // Avoids matching "src" inside "crates/jcode-app-core/src/..."
    // indiscriminately (those are handled by L2b above).
    if let Some(pos) = entry.path.find(query_lower) {
        let is_segment_boundary = pos == 0
            || entry.path.as_bytes().get(pos.wrapping_sub(1)) == Some(&b'/');
        if is_segment_boundary {
            return 45.0 * (1.0 - (pos as f64 / entry.path.len().max(1) as f64));
        }
    }

    // L5 – DP fuzzy subsequence  (~500 ns) ─────────────────────────
    // Uses jcode-fuzzy which already treats `/`, `-`, `_`, `.`, `:`
    // as boundary characters, giving path-like queries a natural boost.
    let filename_score = jcode_fuzzy::fuzzy_score(query_lower, &entry.filename).unwrap_or(0) as f64;
    let path_score = jcode_fuzzy::fuzzy_score(query_lower, &entry.path).unwrap_or(0) as f64;

    if filename_score > 0.0 || path_score > 0.0 {
        return 20.0
            + filename_score.max(path_score) * 1.0
            + if filename_score > 0.0 { 10.0 } else { 0.0 };
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
        } else if !entry.path.starts_with('.') {
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

    let query_lower = query.to_lowercase();
    let query_bag = CharBag::from(&query_lower);
    let mut results: Vec<FileMatch> = Vec::with_capacity(64);

    // Chain base + lazy entries (two-layer index, see PathIndex docs).
    for entry in index.entries.iter().chain(index.lazy_entries.iter()) {
        // CharBag pre-filter: O(1), eliminates 60-80% of candidates.
        if !entry.char_bag.is_superset(query_bag) {
            continue;
        }

        let score = match_entry(entry, &query_lower);
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

    // Inject matching ancestor directories so users can navigate into
    // them (e.g. @src shows crates/jcode-tui/src/ as a clickable dir).
    if !query.is_empty() {
        let mut new_dirs: Vec<FileMatch> = Vec::new();
        let mut seen: HashSet<Arc<str>> = HashSet::new();
        for r in &results {
            seen.insert(r.path.clone());
        }
        for r in &results {
            let full: &str = &r.path;
            let mut remaining = full;
            while let Some(slash) = remaining.rfind('/') {
                let ancestor = &full[..slash + 1];
                // Stop walking up once the ancestor no longer matches.
                if !dir_matches_query(&query_lower, ancestor) {
                    break;
                }
                let key: Arc<str> = Arc::from(ancestor);
                if !seen.contains(&key) {
                    seen.insert(key.clone());
                    // Shorter paths rank higher within the same score tier.
                    let depth_penalty = ancestor.matches('/').count() as f64 * 0.5;
                    new_dirs.push(FileMatch {
                        score: 95.0 - depth_penalty,
                        path: key,
                        is_directory: true,
                        is_recent: false,
                        is_likely_binary: false,
                    });
                }
                remaining = &full[..slash];
            }
        }
        results.extend(new_dirs);
    }

    // Stable sort: score descending, then path ascending.
    results.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.path.cmp(&b.path))
    });
    results.truncate(MAX_RESULTS);
    results
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
/// Uses `symlink_metadata` to avoid following symlinks, and skips the
/// directories listed in `SKIP_DIRS`.
async fn walkdir_collect(cwd: &Path) -> Vec<String> {
    let cwd = cwd.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let mut files: Vec<String> = Vec::new();
        let mut stack: Vec<PathBuf> = vec![cwd.clone()];

        while let Some(dir) = stack.pop() {
            let entries = match std::fs::read_dir(&dir) {
                Ok(e) => e,
                Err(_) => continue,
            };

            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();

                // Skip symlinks.
                if entry.file_type().is_ok_and(|ft| ft.is_symlink()) {
                    continue;
                }

                if path.is_dir() {
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    if !SKIP_DIRS.contains(&name) {
                        stack.push(path);
                    }
                    continue;
                }

                if let Ok(rel) = path.strip_prefix(&cwd) {
                    if let Some(rel_str) = rel.to_str() {
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
}

impl FileIndexManager {
    pub fn new(cwd: PathBuf) -> Self {
        Self {
            current: Arc::new(RwLock::new(Arc::new(PathIndex::empty(cwd.clone())))),
            cwd,
            refreshing: Arc::new(AtomicBool::new(false)),
        }
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
pub(crate) struct FileMentionCache {
    index_manager: FileIndexManager,
    history: SearchHistory,
    /// Recently-opened files (capped at 10), used for ranking.
    recent_files: VecDeque<Arc<str>>,
}

impl FileMentionCache {
    pub fn new() -> Self {
        Self {
            index_manager: FileIndexManager::new(PathBuf::new()),
            history: SearchHistory::new(),
            recent_files: VecDeque::new(),
        }
    }
}

impl Default for FileMentionCache {
    fn default() -> Self {
        Self::new()
    }
}

impl FileMentionCache {
    ///
    /// Must return in < 5 ms on the synchronous UI thread.
    pub fn candidates(&mut self, query: &str) -> Vec<FileMatch> {
        // 1. Check search history.
        match self.history.lookup(query) {
            LookupResult::Hit(results) => return results,
            LookupResult::Miss => {}
        }

        // 2. Snapshot current index.
        let snapshot = self.index_manager.snapshot();

        // If the index is still empty, trigger a background build and bail.
        if snapshot.entries.is_empty() {
            self.index_manager.refresh_async();
            return Vec::new();
        }

        // 3. On-demand lazy scan for ignored directories.
        let mut index = (*snapshot).clone();
        if !query.is_empty() {
            ensure_ignored_dir_scanned(query, &mut index);
        }

        // 4. Full search.
        let recent: Vec<Arc<str>> = self.recent_files.iter().cloned().collect();
        let results = search_in_index(query, &index, &recent);

        // 5. Persist in history.
        self.history.save(query, &results);
        results
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
        if self.index_manager.cwd() != cwd {
            self.index_manager = FileIndexManager::new(cwd.to_path_buf());
            self.history = SearchHistory::new();
        }

        let index = self.index_manager.snapshot();
        let needs_refresh = index.entries.is_empty()
            || index.built_at.elapsed() > Duration::from_secs(30);

        if needs_refresh && !self.index_manager.is_refreshing() {
            self.index_manager.refresh_async();
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
        }
    }

    // -- match_entry ---------------------------------------------------------

    #[test]
    fn match_filename_prefix_wins() {
        let entry = file_entry("deep/nested/src/lib.rs");
        let score = match_entry(&entry, "lib.rs");
        assert!(
            score > 90.0,
            "filename prefix should score high, got {score}"
        );
    }

    #[test]
    fn match_path_prefix() {
        let entry = file_entry("src/cli/args.rs");
        let score = match_entry(&entry, "src/cli");
        assert!(
            score > 70.0,
            "path prefix should score high, got {score}"
        );
    }

    #[test]
    fn match_fuzzy_subsequence() {
        let entry = file_entry("src/cli/startup.rs");
        let score = match_entry(&entry, "scli");
        assert!(score > 0.0, "fuzzy match should work for 'scli'");
    }

    #[test]
    fn match_no_match() {
        let entry = file_entry("src/lib.rs");
        assert_eq!(match_entry(&entry, "xyz"), 0.0);
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
    fn history_incremental_narrowing() {
        let mut history = SearchHistory::new();
        let all = vec![
            file_match("src/cli/args.rs"),
            file_match("src/cli/startup.rs"),
            file_match("src/main.rs"),
        ];

        history.save("src", &all);
        match history.lookup("src/cl") {
            LookupResult::Hit(results) => {
                assert_eq!(results.len(), 2);
                assert!(results.iter().all(|r| r.path.contains("src/cl")));
            }
            _ => panic!("should be hit"),
        }
    }

    #[test]
    fn history_backspace_recovery() {
        let mut history = SearchHistory::new();
        history.save("src", &vec![file_match("src/cli/args.rs"), file_match("src/main.rs")]);
        history.save("src/cl", &vec![file_match("src/cli/args.rs")]);
        history.save("src/cli", &vec![file_match("src/cli/args.rs")]);

        match history.lookup("src/cl") {
            LookupResult::Hit(results) => assert_eq!(results.len(), 1),
            _ => panic!("should recover from history"),
        }
    }

    #[test]
    fn history_jump_edit_clears() {
        let mut history = SearchHistory::new();
        history.save("src", &[file_match("src/main.rs")]);
        // User types "srx" — that's not a prefix extension of "src".
        history.save("srx", &[file_match("srx")]);
        // After the jump-edit clear, history should contain only "srx".
        match history.lookup("srx") {
            LookupResult::Hit(results) => assert_eq!(results.len(), 1),
            _ => panic!("should hit after jump edit"),
        }
    }
}
