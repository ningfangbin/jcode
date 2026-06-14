//! Memory recall benchmark (Mode 1 / no-LLM).
//!
//! Faithful offline harness for measuring memory retrieval accuracy. Reuses the
//! REAL jcode retrieval primitives:
//!   - `jcode::memory_graph::MemoryGraph` deserialization (real on-disk graphs)
//!   - `jcode::embedding::embed` (real all-MiniLM-L6-v2 ONNX model)
//!   - `jcode::memory::format_context_for_relevance` (real live query window)
//!   - a faithful re-implementation of `score_and_filter` (cosine + gap filter)
//!
//! Privacy: all data lives OUTSIDE the repo (default `~/jcode-memory-bench`).
//! Nothing here writes into the repo tree.
//!
//! Subcommands:
//!   queries  - replay sessions -> emit per-turn query windows (labels/queries.jsonl)
//!   pool     - run retrievers over queries -> emit candidate pool (labels/pool.jsonl)
//!   metrics  - read cached gold labels -> emit recall@k/MRR/nDCG (results/*.json)
//!
//! Run via: cargo run --profile selfdev --features dev-bins --bin memory_recall_bench -- <subcmd> ...

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use jcode::embedding;
use jcode::memory::format_context_for_relevance;
use jcode::memory_graph::MemoryGraph;
use jcode::session::Session;
use serde::{Deserialize, Serialize};

// ---- Tunables that mirror production retrieval (memory.rs) ----
const EMBEDDING_SIMILARITY_THRESHOLD: f32 = 0.5;
const EMBEDDING_MAX_HITS: usize = 10;
const GAP_FACTOR: f32 = 0.25;
const MIN_KEEP: usize = 1;
// Memory agent context window (memory_prompt.rs constants are private; the
// production path calls format_context_for_relevance over the full message list).

fn bench_root() -> PathBuf {
    std::env::var("MEMORY_BENCH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs_home().join("jcode-memory-bench")
        })
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME").map(PathBuf::from).unwrap_or_default()
}

// ---------------- Corpus ----------------

#[derive(Clone)]
struct CorpusMemory {
    id: String,
    content: String,
    category: String,
    embedding: Option<Vec<f32>>,
    graph: String,
    source: Option<String>,
    active: bool,
    confidence: f32,
    strength: u32,
    age_days: f32,
}

struct Corpus {
    memories: Vec<CorpusMemory>,
}

impl Corpus {
    /// Load a single graph file as the search corpus.
    fn load_graph_file(path: &Path) -> Result<Corpus> {
        let graph = load_graph(path)?;
        let graph_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        let now = chrono::Utc::now();
        let memories = graph
            .memories
            .values()
            .map(|m| CorpusMemory {
                id: m.id.clone(),
                content: m.content.clone(),
                category: m.category.to_string(),
                embedding: m.embedding.clone(),
                graph: graph_name.clone(),
                source: m.source.clone(),
                active: m.active,
                confidence: m.confidence,
                strength: m.strength,
                age_days: (now - m.updated_at).num_seconds().max(0) as f32 / 86_400.0,
            })
            .collect();
        Ok(Corpus { memories })
    }

    fn active(&self) -> impl Iterator<Item = &CorpusMemory> {
        self.memories.iter().filter(|m| m.active)
    }
}

fn load_graph(path: &Path) -> Result<MemoryGraph> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let graph: MemoryGraph =
        serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))?;
    Ok(graph)
}

// ---------------- Retrievers ----------------

/// Faithful re-implementation of MemoryManager::score_and_filter for the dense
/// (embedding) path, including the score-distribution gap filter.
fn dense_retrieve(
    query_emb: &[f32],
    corpus: &Corpus,
    threshold: f32,
    limit: usize,
    apply_gap: bool,
) -> Vec<(String, f32)> {
    let entries: Vec<&CorpusMemory> = corpus
        .active()
        .filter(|m| m.embedding.is_some())
        .collect();
    let emb_refs: Vec<&[f32]> = entries
        .iter()
        .map(|m| m.embedding.as_deref().unwrap())
        .collect();
    let scores = embedding::batch_cosine_similarity(query_emb, &emb_refs);

    let mut scored: Vec<(String, f32)> = entries
        .iter()
        .zip(scores)
        .filter(|(_, s)| *s >= threshold)
        .map(|(m, s)| (m.id.clone(), s))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(limit);

    if apply_gap {
        scored = apply_gap_filter(scored, threshold);
    }
    scored
}

fn apply_gap_filter(scored: Vec<(String, f32)>, threshold: f32) -> Vec<(String, f32)> {
    if scored.len() <= 1 {
        return scored;
    }
    let top = scored[0].1;
    let range = (top - threshold).max(0.01);
    let max_gap = range * GAP_FACTOR;
    let mut keep = scored.len();
    for i in 1..scored.len() {
        let drop = scored[i - 1].1 - scored[i].1;
        if drop > max_gap && i >= MIN_KEEP {
            keep = i;
            break;
        }
    }
    scored.into_iter().take(keep).collect()
}

/// Simple BM25 lexical retriever over memory content (for hybrid experiments
/// and to widen the candidate pool so pooled gold labels are less biased).
struct Bm25 {
    docs: Vec<(String, Vec<String>)>, // (id, tokens)
    df: HashMap<String, usize>,
    avgdl: f32,
    n: usize,
}

impl Bm25 {
    fn build(corpus: &Corpus) -> Bm25 {
        let mut docs = Vec::new();
        let mut df: HashMap<String, usize> = HashMap::new();
        let mut total_len = 0usize;
        for m in corpus.active() {
            let toks = tokenize(&m.content);
            total_len += toks.len();
            let unique: HashSet<&String> = toks.iter().collect();
            for t in unique {
                *df.entry(t.clone()).or_insert(0) += 1;
            }
            docs.push((m.id.clone(), toks));
        }
        let n = docs.len().max(1);
        Bm25 {
            avgdl: total_len as f32 / n as f32,
            n,
            docs,
            df,
        }
    }

    fn search(&self, query: &str, limit: usize) -> Vec<(String, f32)> {
        const K1: f32 = 1.2;
        const B: f32 = 0.75;
        let q = tokenize(query);
        let qset: HashSet<&String> = q.iter().collect();
        let mut out: Vec<(String, f32)> = Vec::new();
        for (id, toks) in &self.docs {
            let dl = toks.len() as f32;
            let mut tf: HashMap<&String, f32> = HashMap::new();
            for t in toks {
                *tf.entry(t).or_insert(0.0) += 1.0;
            }
            let mut score = 0.0f32;
            for term in &qset {
                let Some(&f) = tf.get(*term) else { continue };
                let n_q = *self.df.get(*term).unwrap_or(&0) as f32;
                if n_q == 0.0 {
                    continue;
                }
                let idf = (((self.n as f32 - n_q + 0.5) / (n_q + 0.5)) + 1.0).ln();
                let denom = f + K1 * (1.0 - B + B * dl / self.avgdl);
                score += idf * (f * (K1 + 1.0)) / denom;
            }
            if score > 0.0 {
                out.push((id.clone(), score));
            }
        }
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        out.truncate(limit);
        out
    }
}

fn tokenize(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

/// Gentle importance prior from confidence/strength/recency. Multiplicative
/// tiebreaker on the fused relevance score; never dominates relevance.
fn memory_prior(m: &CorpusMemory) -> f32 {
    let conf = 0.9 + 0.2 * m.confidence.clamp(0.0, 1.0);
    let strength = 1.0 + 0.05 * (m.strength as f32 + 1.0).ln().min(3.0);
    let recency = 1.0 + 0.1 * (-(m.age_days.max(0.0)) / 120.0).exp();
    conf * strength * recency
}

/// Reciprocal Rank Fusion of multiple ranked lists.
fn rrf(lists: &[Vec<(String, f32)>], k: f32, limit: usize) -> Vec<(String, f32)> {
    let mut fused: HashMap<String, f32> = HashMap::new();
    for list in lists {
        for (rank, (id, _)) in list.iter().enumerate() {
            *fused.entry(id.clone()).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
        }
    }
    let mut out: Vec<(String, f32)> = fused.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out.truncate(limit);
    out
}

// ---------------- Query generation (replay sessions) ----------------

#[derive(Serialize, Deserialize, Clone)]
struct QueryRecord {
    qid: String,
    session: String,
    turn: usize,
    query: String,
    /// Memories whose `source` == this session (excluded from gold to avoid
    /// extraction leakage).
    origin_memory_ids: Vec<String>,
}

fn cmd_queries(args: &[String]) -> Result<()> {
    let opts = parse_kv(args);
    let graph_file = opts
        .get("corpus")
        .cloned()
        .unwrap_or_else(|| bench_root().join("corpus/projects/7fe469b5e6e471c1.json").display().to_string());
    let sessions_dir = opts
        .get("sessions")
        .cloned()
        .unwrap_or_else(|| format!("{}/.jcode/sessions", dirs_home().display()));
    let max_sessions: usize = opts.get("max_sessions").and_then(|s| s.parse().ok()).unwrap_or(20);
    let per_session: usize = opts.get("per_session").and_then(|s| s.parse().ok()).unwrap_or(6);
    let working_dir_filter = opts.get("working_dir").cloned();

    let corpus = Corpus::load_graph_file(Path::new(&graph_file))?;
    // Map source-session -> memory ids, for leakage exclusion.
    let mut by_source: HashMap<String, Vec<String>> = HashMap::new();
    for m in &corpus.memories {
        if let Some(src) = &m.source {
            by_source.entry(src.clone()).or_default().push(m.id.clone());
        }
    }

    // Pick recent sessions (optionally filtered by working_dir).
    let mut sessions: Vec<(PathBuf, std::time::SystemTime)> = std::fs::read_dir(&sessions_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|s| s.to_str()) == Some("json")
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .map(|n| n.starts_with("session_"))
                    .unwrap_or(false)
        })
        .filter_map(|p| {
            let mtime = std::fs::metadata(&p).ok()?.modified().ok()?;
            Some((p, mtime))
        })
        .collect();
    sessions.sort_by(|a, b| b.1.cmp(&a.1));

    let out_path = bench_root().join("labels/queries.jsonl");
    std::fs::create_dir_all(out_path.parent().unwrap())?;
    let mut out = String::new();
    let mut count = 0usize;
    let mut used_sessions = 0usize;

    for (path, _) in sessions {
        if used_sessions >= max_sessions {
            break;
        }
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let session: Session = match serde_json::from_slice(&bytes) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if let Some(filter) = &working_dir_filter {
            if session.working_dir.as_deref() != Some(filter.as_str()) {
                continue;
            }
        }
        let sid = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
        let origin_ids = by_source.get(&sid).cloned().unwrap_or_default();

        let messages: Vec<_> = session.messages.iter().map(|m| m.to_message()).collect();
        // Sample turns: user messages spread through the session.
        let user_turns: Vec<usize> = messages
            .iter()
            .enumerate()
            .filter(|(_, m)| matches!(m.role, jcode::message::Role::User))
            .map(|(i, _)| i)
            // Skip the first two user turns: they are dominated by the session
            // bootstrap (system reminder + opening ask) and carry little
            // working context to retrieve against.
            .skip(2)
            .collect();
        if user_turns.is_empty() {
            continue;
        }
        let step = (user_turns.len() / per_session).max(1);
        let mut taken = 0;
        // Start sampling from the middle so we capture turns with accumulated
        // working context rather than only the earliest turns.
        let start = step / 2;
        for &turn in user_turns.iter().skip(start).step_by(step) {
            if taken >= per_session {
                break;
            }
            // Reconstruct the live query window exactly as production would.
            let window = &messages[..=turn];
            let query = format_context_for_relevance(window);
            if query.len() < 30 {
                continue;
            }
            out.push_str(&serde_json::to_string(&QueryRecord {
                qid: format!("q{:05}", count),
                session: sid.clone(),
                turn,
                query,
                origin_memory_ids: origin_ids.clone(),
            })?);
            out.push('\n');
            count += 1;
            taken += 1;
        }
        if taken > 0 {
            used_sessions += 1;
        }
    }

    std::fs::write(&out_path, out)?;
    println!(
        "Wrote {} queries from {} sessions -> {}",
        count,
        used_sessions,
        out_path.display()
    );
    Ok(())
}

// ---------------- Pool generation ----------------

#[derive(Serialize, Deserialize)]
struct PoolRecord {
    qid: String,
    /// candidate memory id -> {content, retrievers that surfaced it}
    candidates: Vec<PoolCandidate>,
}

#[derive(Serialize, Deserialize)]
struct PoolCandidate {
    id: String,
    content: String,
    #[serde(default)]
    retrievers: Vec<String>,
}

fn cmd_pool(args: &[String]) -> Result<()> {
    let opts = parse_kv(args);
    let graph_file = opts
        .get("corpus")
        .cloned()
        .unwrap_or_else(|| bench_root().join("corpus/projects/7fe469b5e6e471c1.json").display().to_string());
    let pool_n: usize = opts.get("pool_n").and_then(|s| s.parse().ok()).unwrap_or(50);

    let corpus = Corpus::load_graph_file(Path::new(&graph_file))?;
    let content_by_id: HashMap<String, String> =
        corpus.memories.iter().map(|m| (m.id.clone(), m.content.clone())).collect();
    let bm25 = Bm25::build(&corpus);

    let queries = read_queries()?;
    let out_path = bench_root().join("labels/pool.jsonl");
    std::fs::create_dir_all(out_path.parent().unwrap())?;
    let mut out = String::new();

    for q in &queries {
        let q_emb = embedding::embed(&q.query)?;
        // Multiple diverse retrievers widen the pool (reduces pooling bias).
        let dense = dense_retrieve(&q_emb, &corpus, 0.0, pool_n, false);
        let lexical = bm25.search(&q.query, pool_n);
        let fused = rrf(&[dense.clone(), lexical.clone()], 60.0, pool_n);

        let mut retrievers_by_id: HashMap<String, Vec<String>> = HashMap::new();
        for (id, _) in &dense {
            retrievers_by_id.entry(id.clone()).or_default().push("dense".into());
        }
        for (id, _) in &lexical {
            retrievers_by_id.entry(id.clone()).or_default().push("bm25".into());
        }
        for (id, _) in &fused {
            retrievers_by_id.entry(id.clone()).or_default().push("rrf".into());
        }
        // Exclude origin-session memories to avoid extraction leakage.
        let origin: HashSet<&String> = q.origin_memory_ids.iter().collect();
        let candidates: Vec<PoolCandidate> = retrievers_by_id
            .into_iter()
            .filter(|(id, _)| !origin.contains(id))
            .map(|(id, retrievers)| PoolCandidate {
                content: content_by_id.get(&id).cloned().unwrap_or_default(),
                id,
                retrievers,
            })
            .collect();

        out.push_str(&serde_json::to_string(&PoolRecord {
            qid: q.qid.clone(),
            candidates,
        })?);
        out.push('\n');
    }

    std::fs::write(&out_path, out)?;
    println!("Wrote pool for {} queries -> {}", queries.len(), out_path.display());
    Ok(())
}

// ---------------- LLM judge (direct Anthropic via jcode Sidecar) ----------------

#[derive(Deserialize)]
struct JudgeInput {
    qid: String,
    query: String,
    candidates: Vec<PoolCandidate>,
}

const JUDGE_SYSTEM: &str = "You judge whether stored MEMORIES would be genuinely useful to surface to an AI coding agent given the CURRENT conversation context. \
Be strict and prefer precision: a memory is relevant ONLY if a competent engineer would say \"yes, knowing this specifically helps respond here.\" \
Mark relevant when the memory is a fact, user preference, correction, or procedure that applies to what is happening right now. \
Mark NOT relevant when it is off-topic, generic/obvious, only shares surface keywords, or would be noise. When unsure, exclude it. \
The context contains boilerplate (system reminders, tool output); focus on what is actually being worked on. \
Reply with ONLY a JSON array of the relevant candidate numbers, e.g. [1,4] or []. No prose.";

fn build_judge_prompt(input: &JudgeInput) -> String {
    let query = truncate_for_judge(&input.query, 6000);
    let mut p = String::new();
    p.push_str("CURRENT CONTEXT:\n");
    p.push_str(&query);
    p.push_str("\n\nCANDIDATE MEMORIES:\n");
    for (i, c) in input.candidates.iter().enumerate() {
        p.push_str(&format!("{}. {}\n", i + 1, c.content.replace('\n', " ")));
    }
    p.push_str("\nReturn the numbers of the relevant memories as a JSON array.");
    p
}

fn truncate_for_judge(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    // Keep the TAIL: the most recent context is the most informative for recall.
    let chars: Vec<char> = s.chars().collect();
    chars[chars.len() - max..].iter().collect()
}

fn parse_judge_response(resp: &str, n: usize) -> Vec<usize> {
    // Extract the first JSON array of integers from the response.
    let start = resp.find('[');
    let end = resp.rfind(']');
    let (Some(s), Some(e)) = (start, end) else {
        return Vec::new();
    };
    if e < s {
        return Vec::new();
    }
    let slice = &resp[s..=e];
    let nums: Vec<i64> = serde_json::from_str(slice).unwrap_or_default();
    nums.into_iter()
        .filter_map(|x| {
            let idx = x as usize;
            if idx >= 1 && idx <= n {
                Some(idx - 1)
            } else {
                None
            }
        })
        .collect()
}

fn cmd_judge(args: &[String]) -> Result<()> {
    let opts = parse_kv(args);
    let model = opts
        .get("model")
        .cloned()
        .unwrap_or_else(|| "claude-sonnet-4-5-20250929".to_string());
    let concurrency: usize = opts.get("concurrency").and_then(|s| s.parse().ok()).unwrap_or(8);

    let input_path = bench_root().join("labels/judge_ready.jsonl");
    let text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("reading {}", input_path.display()))?;
    let inputs: Vec<JudgeInput> = text
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    eprintln!("Judging {} queries with model {} (concurrency {})", inputs.len(), model, concurrency);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let results = rt.block_on(async {
        use futures::stream::{self, StreamExt};
        stream::iter(inputs.into_iter())
            .map(|input| {
                let model = model.clone();
                async move {
                    let sidecar = jcode::sidecar::Sidecar::with_claude_model(&model);
                    let prompt = build_judge_prompt(&input);
                    let n = input.candidates.len();
                    let mut relevant_ids = Vec::new();
                    // Retry once on transient failure.
                    for attempt in 0..2 {
                        match sidecar.complete(JUDGE_SYSTEM, &prompt).await {
                            Ok(resp) => {
                                let idxs = parse_judge_response(&resp, n);
                                relevant_ids = idxs
                                    .into_iter()
                                    .map(|i| input.candidates[i].id.clone())
                                    .collect();
                                break;
                            }
                            Err(e) => {
                                if attempt == 1 {
                                    eprintln!("judge failed for {}: {}", input.qid, e);
                                } else {
                                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                                }
                            }
                        }
                    }
                    GoldRecord {
                        qid: input.qid,
                        relevant_ids,
                    }
                }
            })
            .buffer_unordered(concurrency)
            .collect::<Vec<_>>()
            .await
    });

    let out_path = bench_root().join("labels/gold.jsonl");
    std::fs::create_dir_all(out_path.parent().unwrap())?;
    let mut out = String::new();
    let mut with_rel = 0usize;
    let mut total = 0usize;
    for g in &results {
        if !g.relevant_ids.is_empty() {
            with_rel += 1;
        }
        total += g.relevant_ids.len();
        out.push_str(&serde_json::to_string(g)?);
        out.push('\n');
    }
    std::fs::write(&out_path, out)?;
    println!(
        "Judged {} queries -> {} ({} with >=1 relevant, {} total labels)",
        results.len(),
        out_path.display(),
        with_rel,
        total
    );
    Ok(())
}

// ---------------- Metrics ----------------

#[derive(Serialize, Deserialize)]
struct GoldRecord {
    qid: String,
    relevant_ids: Vec<String>,
}

fn cmd_metrics(args: &[String]) -> Result<()> {
    let opts = parse_kv(args);
    let graph_file = opts
        .get("corpus")
        .cloned()
        .unwrap_or_else(|| bench_root().join("corpus/projects/7fe469b5e6e471c1.json").display().to_string());
    let config = opts.get("config").cloned().unwrap_or_else(|| "baseline".into());

    let corpus = Corpus::load_graph_file(Path::new(&graph_file))?;
    let bm25 = Bm25::build(&corpus);
    let queries = read_queries()?;
    let gold = read_gold()?;

    let mut recall5 = 0.0;
    let mut recall10 = 0.0;
    let mut mrr = 0.0;
    let mut ndcg = 0.0;
    let mut judged = 0usize;

    for q in &queries {
        let Some(rel) = gold.get(&q.qid) else { continue };
        if rel.is_empty() {
            continue;
        }
        judged += 1;
        let q_emb = embedding::embed(&q.query)?;
        let origin: HashSet<&String> = q.origin_memory_ids.iter().collect();

        let ranked: Vec<String> = match config.as_str() {
            "baseline" => dense_retrieve(&q_emb, &corpus, EMBEDDING_SIMILARITY_THRESHOLD, EMBEDDING_MAX_HITS, true)
                .into_iter()
                .map(|(id, _)| id)
                .collect(),
            "dense_nogap" => dense_retrieve(&q_emb, &corpus, EMBEDDING_SIMILARITY_THRESHOLD, EMBEDDING_MAX_HITS, false)
                .into_iter()
                .map(|(id, _)| id)
                .collect(),
            "dense_t0" => dense_retrieve(&q_emb, &corpus, 0.0, EMBEDDING_MAX_HITS, false)
                .into_iter()
                .map(|(id, _)| id)
                .collect(),
            "dense_t35" => dense_retrieve(&q_emb, &corpus, 0.35, EMBEDDING_MAX_HITS, false)
                .into_iter()
                .map(|(id, _)| id)
                .collect(),
            "bm25" => bm25.search(&q.query, EMBEDDING_MAX_HITS).into_iter().map(|(id, _)| id).collect(),
            "hybrid" => {
                let dense = dense_retrieve(&q_emb, &corpus, 0.0, 50, false);
                let lex = bm25.search(&q.query, 50);
                rrf(&[dense, lex], 60.0, EMBEDDING_MAX_HITS).into_iter().map(|(id, _)| id).collect()
            }
            "hybrid_priors" => {
                let dense = dense_retrieve(&q_emb, &corpus, 0.0, 50, false);
                let lex = bm25.search(&q.query, 50);
                let fused = rrf(&[dense, lex], 60.0, 50);
                // Multiply fused RRF score by a gentle prior derived from
                // confidence / strength / recency. Priors only re-order within
                // the already-retrieved set; they never add/remove candidates.
                let prior: HashMap<&String, f32> = corpus
                    .active()
                    .map(|m| (&m.id, memory_prior(m)))
                    .collect();
                let mut adj: Vec<(String, f32)> = fused
                    .into_iter()
                    .map(|(id, s)| {
                        let p = prior.get(&id).copied().unwrap_or(1.0);
                        (id, s * p)
                    })
                    .collect();
                adj.sort_by(|a, b| b.1.total_cmp(&a.1));
                adj.into_iter().take(EMBEDDING_MAX_HITS).map(|(id, _)| id).collect()
            }
            other => anyhow::bail!("unknown config: {other}"),
        };
        let ranked: Vec<String> = ranked.into_iter().filter(|id| !origin.contains(id)).collect();
        let rel_set: HashSet<&String> = rel.iter().collect();

        recall5 += recall_at(&ranked, &rel_set, 5);
        recall10 += recall_at(&ranked, &rel_set, 10);
        mrr += reciprocal_rank(&ranked, &rel_set);
        ndcg += ndcg_at(&ranked, &rel_set, 10);
    }

    let n = judged.max(1) as f32;
    let result = serde_json::json!({
        "config": config,
        "corpus": graph_file,
        "queries_judged": judged,
        "recall@5": recall5 / n,
        "recall@10": recall10 / n,
        "mrr": mrr / n,
        "ndcg@10": ndcg / n,
    });
    let out_path = bench_root().join(format!("results/{}.json", config));
    std::fs::create_dir_all(out_path.parent().unwrap())?;
    std::fs::write(&out_path, serde_json::to_string_pretty(&result)?)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn recall_at(ranked: &[String], rel: &HashSet<&String>, k: usize) -> f32 {
    if rel.is_empty() {
        return 0.0;
    }
    let hit = ranked.iter().take(k).filter(|id| rel.contains(id)).count();
    hit as f32 / rel.len() as f32
}

fn reciprocal_rank(ranked: &[String], rel: &HashSet<&String>) -> f32 {
    for (i, id) in ranked.iter().enumerate() {
        if rel.contains(id) {
            return 1.0 / (i as f32 + 1.0);
        }
    }
    0.0
}

fn ndcg_at(ranked: &[String], rel: &HashSet<&String>, k: usize) -> f32 {
    let mut dcg = 0.0;
    for (i, id) in ranked.iter().take(k).enumerate() {
        if rel.contains(id) {
            dcg += 1.0 / ((i as f32 + 2.0).ln() / 2f32.ln());
        }
    }
    let ideal_hits = rel.len().min(k);
    let mut idcg = 0.0;
    for i in 0..ideal_hits {
        idcg += 1.0 / ((i as f32 + 2.0).ln() / 2f32.ln());
    }
    if idcg == 0.0 { 0.0 } else { dcg / idcg }
}

// ---------------- helpers ----------------

fn read_queries() -> Result<Vec<QueryRecord>> {
    let path = bench_root().join("labels/queries.jsonl");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {} (run `queries` first)", path.display()))?;
    Ok(text.lines().filter_map(|l| serde_json::from_str(l).ok()).collect())
}

fn read_gold() -> Result<HashMap<String, Vec<String>>> {
    let path = bench_root().join("labels/gold.jsonl");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {} (run judge first)", path.display()))?;
    let mut map = HashMap::new();
    for line in text.lines() {
        if let Ok(g) = serde_json::from_str::<GoldRecord>(line) {
            map.insert(g.qid, g.relevant_ids);
        }
    }
    Ok(map)
}

fn parse_kv(args: &[String]) -> HashMap<String, String> {
    let mut m = HashMap::new();
    for a in args {
        if let Some(rest) = a.strip_prefix("--") {
            if let Some((k, v)) = rest.split_once('=') {
                m.insert(k.to_string(), v.to_string());
            }
        }
    }
    m
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().cloned().unwrap_or_default();
    let rest = if args.len() > 1 { &args[1..] } else { &[] };
    match cmd.as_str() {
        "queries" => cmd_queries(rest),
        "pool" => cmd_pool(rest),
        "judge" => cmd_judge(rest),
        "metrics" => cmd_metrics(rest),
        _ => {
            eprintln!(
                "usage: memory_recall_bench <queries|pool|metrics> [--key=value ...]\n\
                 \n\
                 queries  --corpus=PATH --sessions=DIR --max_sessions=N --per_session=N [--working_dir=DIR]\n\
                 pool     --corpus=PATH --pool_n=50\n\
                 metrics  --corpus=PATH --config=baseline|dense_nogap|bm25|hybrid\n\
                 \n\
                 Bench dir: {} (override with MEMORY_BENCH_DIR)",
                bench_root().display()
            );
            Ok(())
        }
    }
}
