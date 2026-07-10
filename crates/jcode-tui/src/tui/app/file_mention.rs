use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FileMentionCandidate {
    pub path: String,
    pub is_directory: bool,
}

impl FileMentionCandidate {
    pub fn suffix(&self) -> &'static str {
        if self.is_directory {
            "/"
        } else {
            ""
        }
    }
}

pub(crate) struct FileMentionCache {
    files: Vec<String>,
    dirs: HashSet<String>,
    cwd: PathBuf,
    refreshed_at: Instant,
}

impl FileMentionCache {
    const MAX_FILES: usize = 2000;
    const TTL_MS: u64 = 1000;

    pub fn new() -> Self {
        Self {
            files: Vec::new(),
            dirs: HashSet::new(),
            cwd: PathBuf::new(),
            refreshed_at: Instant::now(),
        }
    }

    pub fn refresh_if_needed(&mut self, cwd: &Path) {
        let expired = self.refreshed_at.elapsed().as_millis() as u64 > Self::TTL_MS;
        let dir_changed = self.cwd != cwd;
        if expired || dir_changed {
            self.cwd = cwd.to_path_buf();
            self.refreshed_at = Instant::now();
            let (files, dirs) = collect_workspace_entries(cwd);
            self.files = files;
            self.dirs = dirs;
        }
    }

    pub fn candidates(&self, query: &str) -> Vec<FileMentionCandidate> {
        let mut result = Vec::new();
        let show_all = query.is_empty();

        for path in &self.files {
            if !show_all && !path_matches(path, query) {
                continue;
            }
            result.push(FileMentionCandidate {
                path: path.clone(),
                is_directory: false,
            });
        }
        for dir in &self.dirs {
            if !show_all && !path_matches(dir, query) {
                continue;
            }
            result.push(FileMentionCandidate {
                path: dir.clone(),
                is_directory: true,
            });
        }
        let q = query;
        result.sort_by(|a, b| {
            if show_all {
                a.is_directory.cmp(&b.is_directory).reverse()
                    .then_with(|| a.path.len().cmp(&b.path.len()))
            } else {
                // Primary: basename exact starts-with gets top rank
                match_score(b, q).cmp(&match_score(a, q))
                    .then_with(|| a.is_directory.cmp(&b.is_directory).reverse())
                    .then_with(|| a.path.len().cmp(&b.path.len()))
            }
        });
        result.truncate(15);
        result
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty() && self.dirs.is_empty()
    }
}

fn collect_workspace_entries(cwd: &Path) -> (Vec<String>, HashSet<String>) {
    let mut files = Vec::new();
    let mut dirs = HashSet::new();

    let output = std::process::Command::new("git")
        .args(["ls-files", "--cached", "--others", "--exclude-standard"])
        .current_dir(cwd)
        .output();
    if let Ok(output) = output {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('.') {
                    continue;
                }
                files.push(line.to_string());
                // Insert all ancestor directories
                if let Some(parent) = Path::new(line).parent() {
                    for ancestor in parent.ancestors() {
                        let a = ancestor.to_string_lossy();
                        if a.is_empty() {
                            break;
                        }
                        dirs.insert(a.into_owned());
                    }
                }
                if files.len() >= FileMentionCache::MAX_FILES {
                    break;
                }
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
    if end == 0 {
        None
    } else {
        Some(first[..end].to_string())
    }
}

/// Check if `path` matches `query` using path-segment-aware matching.
///
/// Strategy (Zed-style):
/// 1. Exact contains in full path (fastest, catches most cases)
/// 2. Query contained in the basename portion
/// 3. Fuzzy subsequence match over the full path
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

/// Score how well `query` matches `path`. Higher = better.
///
/// Priority:
/// 3 = basename starts with query
/// 2 = full path starts with query
/// 1 = basename contains query
/// 0 = fallthrough (full path contains or fuzzy match)
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
