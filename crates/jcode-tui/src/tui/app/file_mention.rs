use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FileMentionCandidate {
    pub path: String,
    pub is_directory: bool,
}

impl FileMentionCandidate {
    pub fn suffix(&self) -> &'static str {
        if self.is_directory { "/" } else { "" }
    }
}

pub(crate) struct FileMentionCache {
    /// Shared cache updated by background async task.
    inner: Arc<Mutex<FileMentionCacheInner>>,
    /// Multi-level prefix cache: each entry is (query, candidates) for a
    /// successively longer query. Used for instant backspace restore.
    history: Vec<(String, Vec<FileMentionCandidate>)>,
}

struct FileMentionCacheInner {
    files: Vec<String>,
    dirs: HashSet<String>,
    cwd: PathBuf,
    refreshed_at: Instant,
    /// True when a background refresh is in flight.
    refreshing: bool,
}

impl FileMentionCache {
    const MAX_FILES: usize = 5000;
    const TTL_MS: u64 = 1000;
    const GIT_TIMEOUT: Duration = Duration::from_secs(2);

    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(FileMentionCacheInner {
                files: Vec::new(),
                dirs: HashSet::new(),
                cwd: PathBuf::new(),
                refreshed_at: Instant::now(),
                refreshing: false,
            })),
            history: Vec::new(),
        }
    }

    /// Kick off an async refresh if cache is stale. Returns immediately;
    /// candidates() will use whichever data is currently in cache.
    pub fn refresh_if_needed(&mut self, cwd: &Path) {
        let (should_refresh, refresh_cwd, is_empty) = {
            let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if inner.refreshing {
                return;
            }
            let expired = inner.refreshed_at.elapsed().as_millis() as u64 > Self::TTL_MS;
            let dir_changed = inner.cwd != cwd;
            let empty = inner.files.is_empty();
            if !expired && !dir_changed {
                return;
            }
            (true, cwd.to_path_buf(), empty)
        };

        if !should_refresh {
            return;
        }

        // First-ever load: synchronously so @ shows results immediately.
        // Subsequent refreshes are async to avoid blocking the render loop.
        if is_empty {
            if let Some((files, dirs)) = git_ls_files_sync(&refresh_cwd) {
                let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                inner.files = files;
                inner.dirs = dirs;
                inner.cwd = refresh_cwd;
                inner.refreshed_at = Instant::now();
                inner.refreshing = false;
                self.history.clear();
                return;
            }
            // git failed - fall through to async which uses walkdir
        }

        // Mark refreshing so concurrent calls don't double-spawn
        {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.refreshing = true;
        }

        let inner_clone = Arc::clone(&self.inner);
        tokio::task::spawn(async move {
            let (files, dirs) = collect_workspace_entries_async(&refresh_cwd).await;
            let mut inner = inner_clone.lock().unwrap_or_else(|e| e.into_inner());
            inner.files = files;
            inner.dirs = dirs;
            inner.cwd = refresh_cwd;
            inner.refreshed_at = Instant::now();
            inner.refreshing = false;
        });
    }

    pub fn candidates(&mut self, query: &str) -> Vec<FileMentionCandidate> {
        // Exact match — last entry in history
        if let Some((last_query, last_candidates)) = self.history.last() {
            if query == last_query && !last_candidates.is_empty() {
                return last_candidates.clone();
            }
        }

        // Narrowing: user typed more chars → filter from last history entry
        if let Some((last_query, last_candidates)) = self.history.last() {
            if query.starts_with(last_query.as_str())
                && !last_query.is_empty()
                && !last_candidates.is_empty()
            {
                let filtered: Vec<FileMentionCandidate> = last_candidates
                    .iter()
                    .filter(|c| path_matches(&c.path, query))
                    .cloned()
                    .collect();
                if !filtered.is_empty() {
                    self.history.push((query.to_string(), filtered.clone()));
                    return filtered;
                }
                // Narrowing to empty — fall through to full scan
            }
        }

        // Broadening (backspace/delete): user removed chars — pop history to
        // the deepest entry whose query IS a prefix of the current query.
        // Then narrow down from there (incremental filter from the cached set).
        while self.history.len() > 1 {
            let parent_idx = self.history.len() - 2;
            let (parent_query, _) = &self.history[parent_idx];
            if parent_query.len() <= query.len() && query.starts_with(parent_query.as_str()) {
                self.history.truncate(parent_idx + 1);
                // Narrow down from parent to current query via incremental filter
                let (base_query, base_candidates) = self.history.last().unwrap();
                if query.len() > base_query.len() {
                    let filtered: Vec<FileMentionCandidate> = base_candidates
                        .iter()
                        .filter(|c| path_matches(&c.path, query))
                        .cloned()
                        .collect();
                    if !filtered.is_empty() {
                        self.history.push((query.to_string(), filtered.clone()));
                        return filtered;
                    }
                }
                break;
            }
            self.history.truncate(parent_idx + 1);
        }

        // Full scan
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut result = Vec::new();
        let show_all = query.is_empty();

        for path in &inner.files {
            if !show_all && !path_matches(path, query) {
                continue;
            }
            result.push(FileMentionCandidate {
                path: path.clone(),
                is_directory: false,
            });
        }
        for dir in &inner.dirs {
            if !show_all && !path_matches(dir, query) {
                continue;
            }
            result.push(FileMentionCandidate {
                path: dir.clone(),
                is_directory: true,
            });
        }
        drop(inner);

        let q = query;
        result.sort_by(|a, b| {
            if show_all {
                a.is_directory.cmp(&b.is_directory).reverse()
                    .then_with(|| a.path.len().cmp(&b.path.len()))
            } else {
                match_score(b, q).cmp(&match_score(a, q))
                    .then_with(|| a.is_directory.cmp(&b.is_directory).reverse())
                    .then_with(|| a.path.len().cmp(&b.path.len()))
            }
        });
        result.truncate(15);
        self.history.clear();
        self.history.push((query.to_string(), result.clone()));
        result
    }
}

/// Collect workspace files and directories (async).
///
/// Strategy:
/// 1. `git ls-files` via tokio::process with timeout (fastest, .gitignore-aware)
/// 2. Recursive `std::fs::read_dir` via spawn_blocking as fallback
async fn collect_workspace_entries_async(cwd: &Path) -> (Vec<String>, HashSet<String>) {
    let cwd = cwd.to_path_buf();
    // Strategy 1: git ls-files with tokio timeout
    match tokio::time::timeout(
        FileMentionCache::GIT_TIMEOUT,
        git_ls_files_async(&cwd),
    )
    .await
    {
        Ok(Some(result)) => return result,
        _ => {}
    }
    // Strategy 2: walkdir fallback on blocking thread
    let cwd2 = cwd.clone();
    tokio::task::spawn_blocking(move || walkdir_fallback(&cwd2))
        .await
        .unwrap_or_default()
}

async fn git_ls_files_async(cwd: &Path) -> Option<(Vec<String>, HashSet<String>)> {
    let output = tokio::process::Command::new("git")
        .args(["ls-files", "--cached", "--others"])
        .current_dir(cwd)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .output()
        .await
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let (files, dirs) = parse_file_list(stdout.lines());
    Some((files, dirs))
}

/// Sync git ls-files with watchdog thread for 2s timeout. Used for first load.
fn git_ls_files_sync(cwd: &Path) -> Option<(Vec<String>, HashSet<String>)> {
    let child = std::process::Command::new("git")
        .args(["ls-files", "--cached", "--others"])
        .current_dir(cwd)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;

    let pid = child.id();
    std::thread::spawn(move || {
        std::thread::sleep(FileMentionCache::GIT_TIMEOUT);
        let _ = std::process::Command::new("kill")
            .arg(pid.to_string())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    });

    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let (files, dirs) = parse_file_list(stdout.lines());
    Some((files, dirs))
}

fn walkdir_fallback(cwd: &Path) -> (Vec<String>, HashSet<String>) {
    let mut files = Vec::new();
    let mut dirs = HashSet::new();

    let mut stack: Vec<PathBuf> = vec![cwd.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n,
                None => continue,
            };
            // Skip .git directory
            if name == ".git" && path.is_dir() {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if let Ok(rel) = path.strip_prefix(cwd) {
                if let Some(rel_str) = rel.to_str() {
                    files.push(rel_str.to_string());
                    // Insert all ancestor directories
                    for ancestor in rel.parent().into_iter().flat_map(|p| p.ancestors()) {
                        let a = ancestor.to_string_lossy();
                        if a.is_empty() {
                            break;
                        }
                        dirs.insert(a.to_string());
                    }
                    if files.len() >= FileMentionCache::MAX_FILES {
                        return (files, dirs);
                    }
                }
            }
        }
    }
    (files, dirs)
}

fn parse_file_list<'a>(lines: impl Iterator<Item = &'a str>) -> (Vec<String>, HashSet<String>) {
    // Collect all file paths (no limit yet)
    let mut files: Vec<String> = lines
        .filter(|line| {
            let line = line.trim();
            !line.is_empty() && !line.starts_with(".git/")
        })
        .map(|line| line.trim().to_string())
        .collect();

    // Sort alphabetically so truncation samples from across the repo
    files.sort();
    files.truncate(FileMentionCache::MAX_FILES);

    // Extract ancestor directories from the truncated file list
    let mut dirs = HashSet::new();
    for line in &files {
        if let Some(parent) = Path::new(line).parent() {
            for ancestor in parent.ancestors() {
                let a = ancestor.to_string_lossy();
                if a.is_empty() {
                    break;
                }
                dirs.insert(a.to_string());
            }
        }
    }
    (files, dirs)
}

pub(crate) fn common_prefix(strings: &[&str]) -> Option<String> {
    if strings.is_empty() {
        return None;
    }
    let first = strings[0];
    let mut end = first.len();
    for s in &strings[1..] {
        end = end.min(s.len());
        end = first[..end]
            .char_indices()
            .take_while(|(i, c)| s.chars().nth(*i) == Some(*c))
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);
        if end == 0 {
            return None;
        }
    }
    if end == 0 { None } else { Some(first[..end].to_string()) }
}

fn path_matches(path: &str, query: &str) -> bool {
    if path.contains(query) {
        return true;
    }
    let basename = Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path);
    if basename.contains(query) {
        return true;
    }
    !crate::tui::fuzzy::fuzzy_match_positions(query, path).is_empty()
}

fn match_score(candidate: &FileMentionCandidate, query: &str) -> u8 {
    let path = &candidate.path;
    let basename = Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path);

    if basename.starts_with(query) {
        3
    } else if path.starts_with(query) {
        2
    } else if basename.contains(query) {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_matches_basename() {
        assert!(path_matches("src/lib.rs", "lib.rs"));
        assert!(path_matches("crates/jcode-tui/src/main.rs", "main.rs"));
    }

    #[test]
    fn path_matches_full_path() {
        assert!(path_matches("src/cli/args.rs", "src/cli"));
        assert!(path_matches("crates/jcode-tui/src/app.rs", "jcode-tui"));
    }

    #[test]
    fn path_matches_fuzzy() {
        // fuzzy.rs matches subsequences: "s" "c" "l" "i" found in "src/cli/...up.rs"
        assert!(path_matches("src/cli/startup.rs", "scli"));
        // "strup" subsequence in "startup"
        assert!(path_matches("src/cli/startup.rs", "strup"));
    }

    #[test]
    fn path_matches_no_match() {
        assert!(!path_matches("src/lib.rs", "xyz"));
        assert!(!path_matches("src/main.rs", "lib"));
    }

    #[test]
    fn match_score_basename_prefix_wins() {
        let c = FileMentionCandidate { path: "deep/nested/src/lib.rs".into(), is_directory: false };
        assert_eq!(match_score(&c, "lib"), 3);
        assert_eq!(match_score(&c, "lib.rs"), 3);
    }

    #[test]
    fn match_score_path_prefix() {
        let c = FileMentionCandidate { path: "src/cli/args.rs".into(), is_directory: false };
        assert_eq!(match_score(&c, "src/cli"), 2);
    }

    #[test]
    fn match_score_basename_contains() {
        let c = FileMentionCandidate { path: "src/cli/startup.rs".into(), is_directory: false };
        assert_eq!(match_score(&c, "tart"), 1);
    }

    #[test]
    fn match_score_fallback() {
        let c = FileMentionCandidate { path: "src/cli/args.rs".into(), is_directory: false };
        // "rs" appears in basename "args.rs" — score 1
        assert_eq!(match_score(&c, "rs"), 1);
        // "xyz" not in basename nor path prefix — score 0
        assert_eq!(match_score(&c, "xyz"), 0);
    }

    #[test]
    fn parse_file_list_includes_dotfiles() {
        let input = ".envrc\n.env.example\nsrc/main.rs\n";
        let (files, dirs) = parse_file_list(input.lines());
        assert!(files.contains(&".envrc".to_string()));
        assert!(files.contains(&".env.example".to_string()));
        assert!(files.contains(&"src/main.rs".to_string()));
        assert!(dirs.contains("src"));
    }

    #[test]
    fn parse_file_list_skips_empty_lines() {
        let input = "src/main.rs\n\n\nsrc/lib.rs\n";
        let (files, _) = parse_file_list(input.lines());
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn parse_file_list_ancestor_dirs() {
        let input = "crates/jcode-tui/src/lib.rs\n";
        let (files, dirs) = parse_file_list(input.lines());
        assert_eq!(files.len(), 1);
        assert!(dirs.contains("crates/jcode-tui/src"));
        assert!(dirs.contains("crates/jcode-tui"));
        assert!(dirs.contains("crates"));
    }

    #[test]
    fn common_prefix_basic() {
        let s: Vec<&str> = vec!["src/cli/args.rs", "src/cli/startup.rs"];
        assert_eq!(common_prefix(&s).unwrap(), "src/cli/");
    }

    #[test]
    fn common_prefix_single_char() {
        let s: Vec<&str> = vec!["abc", "abd", "abe"];
        assert_eq!(common_prefix(&s).unwrap(), "ab");
    }

    #[test]
    fn common_prefix_no_common() {
        let s: Vec<&str> = vec!["abc", "xyz"];
        assert_eq!(common_prefix(&s), None);
    }
}
