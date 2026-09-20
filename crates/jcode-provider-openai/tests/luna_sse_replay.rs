//! Read-only replay harness for a captured real Copilot `/responses` SSE stream
//! (model `gpt-5.6-luna`), used to confirm/deny the 0.85.0 "empty response after
//! tool results" regression at the parsing layer (issue #1336).
//!
//! It drives the *same* entry point the real HTTP streaming path uses:
//! `OpenAIResponsesStream` (whose `poll_next` only ever calls
//! `parse_openai_response_event`), plus a direct feed of
//! `parse_openai_response_event` per SSE `data:` payload, mirroring
//! `crates/jcode-provider-openai-runtime/src/openai_stream_runtime.rs`.
//!
//! The capture is committed verbatim next to this file so the case runs
//! everywhere (including CI), where the original `$HOME` dump does not exist.

use std::collections::{HashMap, HashSet, VecDeque};

use bytes::Bytes;
use futures::{FutureExt, StreamExt};
use jcode_message_types::StreamEvent;
use jcode_provider_openai::stream::{OpenAIResponsesStream, parse_openai_response_event};
use serde_json::Value;

const EXPECTED_CALL_ID: &str = "call_plSP1O5CGpJV7RLo7mKfGYqK";
const EXPECTED_NAME: &str = "bash";
const EXPECTED_ARGS: &str = r#"{"command":"echo HELLO-TOOLS-85"}"#;

/// The capture this harness replays, committed byte-for-byte under
/// `tests/fixtures/`.
const CAPTURED_SSE: &str = include_str!("fixtures/gpt56_responses_tool_call.sse");

/// `LUNA_SSE_RAW` points the harness at a different dump (for example a fresh
/// capture on the machine that reproduced the bug) without editing the test.
fn raw_sse() -> String {
    match std::env::var("LUNA_SSE_RAW") {
        Ok(path) => std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}")),
        Err(_) => CAPTURED_SSE.to_string(),
    }
}

/// Control 1 transform: replace every per-event `item_id` / `item.id` with the
/// single fixed id `"a"` (the shape upstream's `stream_tool_tests.rs` assumes),
/// leaving `call_id` and everything else untouched.
fn with_uniform_item_id(raw: &str) -> String {
    let mut out = String::new();
    for block in raw.split("\n\n") {
        let mut lines: Vec<String> = Vec::new();
        for line in block.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    lines.push(line.to_string());
                    continue;
                }
                let mut value: Value = serde_json::from_str(data).expect("valid SSE JSON");
                if let Some(item_id) = value.get_mut("item_id") {
                    *item_id = Value::String("a".to_string());
                }
                if let Some(id) = value.get_mut("item").and_then(|i| i.get_mut("id")) {
                    *id = Value::String("a".to_string());
                }
                lines.push(format!("data: {value}"));
            } else if !line.is_empty() {
                lines.push(line.to_string());
            }
        }
        if !lines.is_empty() {
            out.push_str(&lines.join("\n"));
            out.push_str("\n\n");
        }
    }
    out
}

/// Path A: byte-exact replay through the real HTTP-SSE stream type.
fn collect_via_responses_stream(raw: &str) -> Vec<StreamEvent> {
    let inner = futures::stream::iter(vec![Ok::<Bytes, reqwest::Error>(Bytes::from(
        raw.to_string(),
    ))]);
    let mut stream = OpenAIResponsesStream::new(inner);
    let mut out = Vec::new();
    loop {
        match stream.next().now_or_never() {
            Some(Some(Ok(event))) => out.push(event),
            Some(Some(Err(err))) => panic!("stream error: {err}"),
            Some(None) => break,
            None => break,
        }
    }
    out
}

/// Path B: feed each SSE `data:` payload straight to the parser, exactly the way
/// the runtime consumes it (send parsed event, then drain `pending`).
fn collect_via_parse_entry(raw: &str) -> Vec<StreamEvent> {
    let mut saw_text_delta = false;
    let mut saw_thinking_delta = false;
    let mut streaming_tool_calls = HashMap::new();
    let mut completed_tool_items = HashSet::new();
    let mut pending: VecDeque<StreamEvent> = VecDeque::new();
    let mut out = Vec::new();

    for block in raw.split("\n\n") {
        let mut data_lines: Vec<&str> = Vec::new();
        for line in block.lines() {
            if let Some(data) = line.strip_prefix("data:") {
                data_lines.push(data.strip_prefix(' ').unwrap_or(data));
            }
        }
        if data_lines.is_empty() {
            continue;
        }
        let data = data_lines.join("\n");
        if let Some(event) = parse_openai_response_event(
            &data,
            &mut saw_text_delta,
            &mut saw_thinking_delta,
            &mut streaming_tool_calls,
            &mut completed_tool_items,
            &mut pending,
        ) {
            out.push(event);
        }
        while let Some(event) = pending.pop_front() {
            out.push(event);
        }
    }
    out
}

fn dump(label: &str, events: &[StreamEvent]) {
    println!("==== {label}: {} StreamEvent(s) ====", events.len());
    for (i, event) in events.iter().enumerate() {
        println!("  [{i}] {event:?}");
    }
}

fn starts(events: &[StreamEvent]) -> Vec<(String, String)> {
    events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ToolUseStart { id, name } => Some((id.clone(), name.clone())),
            _ => None,
        })
        .collect()
}

fn arguments(events: &[StreamEvent]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ToolInputDelta(delta) => Some(delta.as_str()),
            _ => None,
        })
        .collect::<String>()
}

fn ends(events: &[StreamEvent]) -> usize {
    events
        .iter()
        .filter(|event| matches!(event, StreamEvent::ToolUseEnd))
        .count()
}

fn assert_complete_tool_call(label: &str, events: &[StreamEvent]) {
    let starts = starts(events);
    let arguments = arguments(events);
    let ends = ends(events);
    println!("---- {label} summary: starts={starts:?} ends={ends} arguments={arguments:?}");
    assert_eq!(
        starts.len(),
        1,
        "{label}: expected exactly ONE ToolUseStart for {EXPECTED_CALL_ID}, got {starts:?}"
    );
    assert_eq!(
        starts[0],
        (EXPECTED_CALL_ID.to_string(), EXPECTED_NAME.to_string()),
        "{label}: unexpected ToolUseStart"
    );
    assert_eq!(
        arguments, EXPECTED_ARGS,
        "{label}: tool arguments were lost, reordered or truncated"
    );
    assert_eq!(ends, 1, "{label}: expected exactly one ToolUseEnd");
}

// ---------------------------------------------------------------------------
// Real capture (per-event item_id all different, call_id stable).
// ---------------------------------------------------------------------------

#[test]
fn real_capture_via_responses_stream_yields_complete_bash_call() {
    let events = collect_via_responses_stream(&raw_sse());
    dump("real capture / OpenAIResponsesStream", &events);
    assert_complete_tool_call("real capture / OpenAIResponsesStream", &events);
}

#[test]
fn real_capture_via_parse_entry_yields_complete_bash_call() {
    let events = collect_via_parse_entry(&raw_sse());
    dump("real capture / parse_openai_response_event", &events);
    assert_complete_tool_call("real capture / parse_openai_response_event", &events);
}

// ---------------------------------------------------------------------------
// Control 1: same sequence, but every item id collapsed to "a" (upstream test
// shape). Must be green on every ref, otherwise the harness itself is red.
// ---------------------------------------------------------------------------

#[test]
fn control_uniform_item_id_via_responses_stream_yields_complete_bash_call() {
    let events = collect_via_responses_stream(&with_uniform_item_id(&raw_sse()));
    dump("uniform item_id / OpenAIResponsesStream", &events);
    assert_complete_tool_call("uniform item_id / OpenAIResponsesStream", &events);
}

#[test]
fn control_uniform_item_id_via_parse_entry_yields_complete_bash_call() {
    let events = collect_via_parse_entry(&with_uniform_item_id(&raw_sse()));
    dump("uniform item_id / parse_openai_response_event", &events);
    assert_complete_tool_call("uniform item_id / parse_openai_response_event", &events);
}
