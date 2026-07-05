//! Stress probe: build the EXACT mermaid source that
//! `jcode-tui/src/tui/swarm_plan_graph.rs::swarm_plan_mermaid` emits (logic
//! ported verbatim below; keep in sync) for realistic + hostile plan data,
//! then render through the real pipeline (`render_mermaid_untracked`).
//!
//!   scripts/dev_cargo.sh run -p jcode-tui-mermaid --features renderer \
//!       --example swarm_plan_stress
//!
//! Cases:
//!   (a) real deep-mode plan fixture (examples/swarm_plan_fixture.json,
//!       snapshot of ~/.jcode/state/swarm/_home_jeremy_jcode__git.json)
//!   (a23) first 23 items of the fixture (the original task shape)
//!   (b) 40-node plan -> truncation with the disconnected 'more' summary node
//!   (c) labels at exactly MAX_LABEL_CHARS with unicode glyphs
//!   (d) hostile label chars surviving sanitize_label
//!   (e) duplicate sanitized ids (a-1 vs a_1 -> t_a_1) + self-dependency
//!   (f) an item whose id is literally "more" alongside the summary node

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Faithful port of crates/jcode-tui/src/tui/swarm_plan_graph.rs
// ---------------------------------------------------------------------------

const MAX_GRAPH_NODES: usize = 30;
const MAX_LABEL_CHARS: usize = 42;

#[derive(Debug, Clone, Deserialize)]
struct PlanItem {
    content: String,
    status: String,
    id: String,
    #[serde(default)]
    blocked_by: Vec<String>,
    #[serde(default)]
    assigned_to: Option<String>,
}

fn swarm_plan_mermaid(items: &[PlanItem]) -> Option<String> {
    if items.is_empty() {
        return None;
    }
    let shown = &items[..items.len().min(MAX_GRAPH_NODES)];
    let mut out = String::from("flowchart TD\n");

    for item in shown {
        let id = node_id(&item.id);
        let label = node_label(item);
        let class = status_class(&item.status);
        out.push_str(&format!("    {id}[\"{label}\"]:::{class}\n"));
    }

    for item in shown {
        let to = node_id(&item.id);
        for dep in &item.blocked_by {
            if shown.iter().any(|other| &other.id == dep) {
                let from = node_id(dep);
                out.push_str(&format!("    {from} --> {to}\n"));
            }
        }
    }

    let hidden = items.len().saturating_sub(shown.len());
    if hidden > 0 {
        out.push_str(&format!(
            "    more[\"…and {hidden} more tasks\"]:::pending\n"
        ));
    }

    out.push_str("    classDef done fill:#1d3a1d,stroke:#64c864,color:#a8e0a8\n");
    out.push_str("    classDef active fill:#3a321d,stroke:#ffc864,color:#ffe0a8\n");
    out.push_str("    classDef failed fill:#3a1d1d,stroke:#ff6464,color:#ffa8a8\n");
    out.push_str("    classDef blocked fill:#3a2a1d,stroke:#ffaa50,color:#ffd0a0\n");
    out.push_str("    classDef pending fill:#26262e,stroke:#8c8c96,color:#b4b4be\n");
    Some(out)
}

fn node_id(raw: &str) -> String {
    let mut id: String = raw
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    if id.is_empty() {
        id.push('x');
    }
    format!("t_{id}")
}

fn node_label(item: &PlanItem) -> String {
    let glyph = match normalized_status(&item.status) {
        "done" => "✓",
        "active" => "▶",
        "failed" => "✗",
        "blocked" => "⏸",
        _ => "·",
    };
    let mut content = sanitize_label(&item.content);
    if content.chars().count() > MAX_LABEL_CHARS {
        content = content.chars().take(MAX_LABEL_CHARS - 1).collect();
        content.push('…');
    }
    match &item.assigned_to {
        Some(who) if !who.is_empty() => {
            format!("{glyph} {content} · @{}", sanitize_label(who))
        }
        _ => format!("{glyph} {content}"),
    }
}

fn normalized_status(status: &str) -> &'static str {
    match status {
        "completed" | "done" => "done",
        "running" | "running_stale" | "in_progress" | "active" => "active",
        "failed" | "cancelled" | "crashed" => "failed",
        "blocked" => "blocked",
        _ => "pending",
    }
}

fn status_class(status: &str) -> &'static str {
    normalized_status(status)
}

fn sanitize_label(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '"' => '\'',
            '\n' | '\r' | '\t' => ' ',
            '[' | ']' | '{' | '}' => '(',
            _ => c,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Probe harness
// ---------------------------------------------------------------------------

fn item(id: &str, content: &str, status: &str, blocked_by: &[&str]) -> PlanItem {
    PlanItem {
        content: content.to_string(),
        status: status.to_string(),
        id: id.to_string(),
        blocked_by: blocked_by.iter().map(|s| s.to_string()).collect(),
        assigned_to: None,
    }
}

fn probe(name: &str, src: &str) -> bool {
    match jcode_tui_mermaid::render_mermaid_untracked(src, Some(100)) {
        jcode_tui_mermaid::RenderResult::Image {
            width,
            height,
            path,
            ..
        } => {
            println!("OK   {name}: {width}x{height} -> {}", path.display());
            true
        }
        jcode_tui_mermaid::RenderResult::Error(err) => {
            println!("FAIL {name}: {err}");
            println!("---- offending source ----\n{src}\n--------------------------");
            false
        }
    }
}

fn main() {
    let mut ok = true;

    // (a) real plan fixture (35 items at snapshot time, so it also exercises
    // real truncation) + (a23) the original 23-item shape.
    let fixture = include_str!("swarm_plan_fixture.json");
    let real: Vec<PlanItem> = serde_json::from_str(fixture).expect("fixture parses");
    println!("fixture items: {}", real.len());
    let src_a = swarm_plan_mermaid(&real).expect("graph");
    println!(
        "--- (a) real plan mermaid ({} lines) ---\n{src_a}",
        src_a.lines().count()
    );
    ok &= probe("a-real-plan-full", &src_a);
    let src_a23 = swarm_plan_mermaid(&real[..real.len().min(23)]).expect("graph");
    ok &= probe("a23-real-plan-23", &src_a23);

    // (b) truncation: 40 nodes, chain deps, disconnected 'more' summary node.
    let items_b: Vec<PlanItem> = (0..40)
        .map(|i: usize| {
            let dep = format!("t{}", i.saturating_sub(1));
            let deps: Vec<&str> = if i == 0 { vec![] } else { vec![dep.as_str()] };
            item(
                &format!("t{i}"),
                &format!("task number {i}"),
                "queued",
                &deps,
            )
        })
        .collect();
    let src_b = swarm_plan_mermaid(&items_b).expect("graph");
    assert!(src_b.contains("…and 10 more tasks"), "summary node missing");
    ok &= probe("b-truncation-40", &src_b);

    // (c) label at exactly MAX_LABEL_CHARS with unicode glyphs (no truncation
    // ellipsis should appear) and one char over (ellipsis should appear).
    let exact: String = "日本語テスト🎯émü→"
        .chars()
        .cycle()
        .take(MAX_LABEL_CHARS)
        .collect();
    assert_eq!(exact.chars().count(), MAX_LABEL_CHARS);
    let over: String = "日本語テスト🎯émü→"
        .chars()
        .cycle()
        .take(MAX_LABEL_CHARS + 1)
        .collect();
    let items_c = vec![
        item("uni-exact", &exact, "running", &[]),
        item("uni-over", &over, "queued", &["uni-exact"]),
    ];
    let src_c = swarm_plan_mermaid(&items_c).expect("graph");
    println!("--- (c) unicode mermaid ---\n{src_c}");
    assert!(src_c.lines().next().is_some(), "unreachable");
    ok &= probe("c-unicode-max-label", &src_c);

    // (d) hostile label chars that sanitize_label passes through unchanged:
    // backtick, backslash, #, %, &, <, >, |, semicolon, parens.
    let items_d = vec![
        item("h1", "backtick `code` and backslash \\ path", "queued", &[]),
        item("h2", "hash # percent % amp & semi;colon", "queued", &["h1"]),
        item("h3", "angle <b>bold</b> pipe | (parens)", "queued", &["h2"]),
        item(
            "h4",
            "entity-ish &lt;#35; #quot; %%{init}%%",
            "queued",
            &["h3"],
        ),
    ];
    let src_d = swarm_plan_mermaid(&items_d).expect("graph");
    println!("--- (d) hostile-label mermaid ---\n{src_d}");
    ok &= probe("d-hostile-labels", &src_d);

    // (d2) each hostile char in isolation so a failure names the culprit.
    for (tag, s) in [
        ("backtick", "a `b` c"),
        ("backslash", "a \\ b"),
        ("hash", "a # b"),
        ("percent", "a % b"),
        ("amp", "a & b"),
        ("lt-gt", "a <b> c"),
        ("pipe", "a | b"),
        ("semicolon", "a ; b"),
        ("parens", "a (b) c"),
        ("entity", "&amp; &#35;"),
        ("percent-directive", "%%{init: {'theme':'dark'}}%%"),
    ] {
        let its = vec![item("only", s, "queued", &[])];
        let src = swarm_plan_mermaid(&its).expect("graph");
        ok &= probe(&format!("d2-{tag}"), &src);
    }

    // (e) duplicate sanitized ids: a-1 and a_1 both -> t_a_1, plus a
    // self-dependency (a-1 blocked_by a-1 -> t_a_1 --> t_a_1).
    let items_e = vec![
        item("a-1", "first flavor of a1", "completed", &["a-1"]),
        item("a_1", "second flavor of a1", "running", &["a-1"]),
        item("b", "depends on both", "queued", &["a-1", "a_1"]),
    ];
    let src_e = swarm_plan_mermaid(&items_e).expect("graph");
    println!("--- (e) duplicate-id + self-dep mermaid ---\n{src_e}");
    let decl_count = src_e
        .lines()
        .filter(|l| l.trim_start().starts_with("t_a_1["))
        .count();
    println!("     (e) t_a_1 declared {decl_count} times (collision => silent merge)");
    let self_edge = src_e.contains("t_a_1 --> t_a_1");
    println!("     (e) self-edge t_a_1 --> t_a_1 present: {self_edge}");
    ok &= probe("e-dup-ids-self-dep", &src_e);

    // (f) an item whose id is exactly "more" plus enough items to force the
    // truncation summary node also named `more` (unprefixed).
    let mut items_f: Vec<PlanItem> = (0..MAX_GRAPH_NODES - 1)
        .map(|i| item(&format!("f{i}"), &format!("filler {i}"), "queued", &[]))
        .collect();
    items_f.insert(
        0,
        item("more", "a task literally named more", "running", &[]),
    );
    // push past MAX so the summary node is emitted
    for i in 0..5 {
        items_f.push(item(
            &format!("extra{i}"),
            &format!("extra {i}"),
            "queued",
            &[],
        ));
    }
    let src_f = swarm_plan_mermaid(&items_f).expect("graph");
    assert!(src_f.contains("t_more["), "prefixed more node missing");
    assert!(src_f.contains("\n    more["), "summary more node missing");
    ok &= probe("f-more-id-collision", &src_f);

    println!(
        "\nresult: {}",
        if ok { "ALL OK" } else { "FAILURES PRESENT" }
    );
    std::process::exit(if ok { 0 } else { 1 });
}
