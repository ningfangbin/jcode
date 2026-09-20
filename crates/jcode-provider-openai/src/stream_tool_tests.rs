use super::*;
use futures::{FutureExt, StreamExt, channel::mpsc};
use serde_json::json;

type Sender = mpsc::UnboundedSender<Result<Bytes, reqwest::Error>>;

fn stream() -> (Sender, OpenAIResponsesStream) {
    let (tx, rx) = mpsc::unbounded();
    (tx, OpenAIResponsesStream::new(rx))
}

fn send(tx: &Sender, event: Value) {
    tx.unbounded_send(Ok(Bytes::from(format!("data: {event}\n\n"))))
        .unwrap();
}

fn next(stream: &mut OpenAIResponsesStream) -> StreamEvent {
    // The upstream connection remains open, with no later event supplied. A
    // buffering parser returns Pending here instead of passing this assertion.
    stream
        .next()
        .now_or_never()
        .expect("event must be visible now")
        .expect("stream remains open")
        .expect("valid stream event")
}

fn idle(stream: &mut OpenAIResponsesStream) {
    assert!(stream.next().now_or_never().is_none());
}

fn added(tx: &Sender, id: &str, name: &str) {
    send(
        tx,
        json!({"type":"response.output_item.added", "item": {
            "type":"function_call", "id":id, "call_id":format!("call_{id}"),
            "name":name, "arguments":""
        }}),
    );
}

fn delta(tx: &Sender, id: &str, fragment: &str) {
    send(
        tx,
        json!({"type":"response.function_call_arguments.delta", "item_id":id, "delta":fragment}),
    );
}

fn done(tx: &Sender, id: &str, arguments: &str) {
    send(
        tx,
        json!({"type":"response.function_call_arguments.done", "item_id":id, "arguments":arguments}),
    );
}

fn assert_start(stream: &mut OpenAIResponsesStream, id: &str, name: &str) {
    assert!(
        matches!(next(stream), StreamEvent::ToolUseStart { id: actual_id, name: actual_name }
        if actual_id == format!("call_{id}") && actual_name == name)
    );
}

// ---------------------------------------------------------------------------
// Real `/responses` shape (Copilot + `gpt-5.6-luna`, capture in
// `08_ai-memory/jcode/sse-raw.txt`): every event carries a *different* random
// `item_id`, `call_id` is stable on item events only, and `output_index` is the
// single identity shared by `output_item.added`, every argument delta,
// `*_arguments.done` and `output_item.done`. The helpers above only cover the
// "one fixed item id" shape, so the tests below use dedicated ones.
// ---------------------------------------------------------------------------

fn real_added(tx: &Sender, index: i64, id: &str, call_id: &str, name: &str) {
    send(
        tx,
        json!({"type":"response.output_item.added", "output_index":index, "item": {
            "type":"function_call", "id":id, "call_id":call_id,
            "name":name, "arguments":"", "status":"in_progress"
        }}),
    );
}

fn real_delta(tx: &Sender, index: i64, id: &str, fragment: &str) {
    send(
        tx,
        json!({"type":"response.function_call_arguments.delta", "output_index":index,
        "item_id":id, "delta":fragment}),
    );
}

fn real_arguments_done(tx: &Sender, index: i64, id: &str, arguments: &str) {
    send(
        tx,
        json!({"type":"response.function_call_arguments.done", "output_index":index,
        "item_id":id, "arguments":arguments}),
    );
}

fn real_item_done(tx: &Sender, index: i64, id: &str, call_id: &str, name: &str, arguments: &str) {
    send(
        tx,
        json!({"type":"response.output_item.done", "output_index":index, "item": {
            "type":"function_call", "id":id, "call_id":call_id,
            "name":name, "arguments":arguments, "status":"completed"
        }}),
    );
}

fn assert_started_as(stream: &mut OpenAIResponsesStream, id: &str, name: &str) {
    assert!(
        matches!(next(stream), StreamEvent::ToolUseStart { id: actual_id, name: actual_name }
        if actual_id == id && actual_name == name),
        "expected ToolUseStart({id}, {name})"
    );
}

fn assert_ended(stream: &mut OpenAIResponsesStream) {
    assert!(matches!(next(stream), StreamEvent::ToolUseEnd));
}

fn assert_delta(stream: &mut OpenAIResponsesStream, expected: &str) {
    assert!(matches!(next(stream), StreamEvent::ToolInputDelta(actual) if actual == expected));
}

#[test]
fn tool_name_and_argument_fragments_are_visible_before_done() {
    for name in ["read", "batch", "multi_tool_use.parallel"] {
        let (tx, mut stream) = stream();
        added(&tx, "a", name);
        assert_start(&mut stream, "a", name);
        idle(&mut stream);
        delta(&tx, "a", "{\"intent\":\"Read files\",");
        assert_delta(&mut stream, "{\"intent\":\"Read files\",");
        idle(&mut stream);
        delta(&tx, "a", "\"path\":\"文档\"}");
        assert_delta(&mut stream, "\"path\":\"文档\"}");
        idle(&mut stream);
        done(&tx, "a", "{\"intent\":\"Read files\",\"path\":\"文档\"}");
        assert!(matches!(next(&mut stream), StreamEvent::ToolUseEnd));
        idle(&mut stream);
    }
}

#[test]
fn done_snapshots_only_emit_unseen_suffix_and_never_duplicate_calls() {
    let (tx, mut stream) = stream();
    added(&tx, "a", "read");
    assert_start(&mut stream, "a", "read");
    delta(&tx, "a", "{\"path\":");
    assert_delta(&mut stream, "{\"path\":");
    done(&tx, "a", "{\"path\":\"README.md\"}");
    assert_delta(&mut stream, "\"README.md\"}");
    assert!(matches!(next(&mut stream), StreamEvent::ToolUseEnd));
    for _ in 0..2 {
        done(&tx, "a", "{\"path\":\"README.md\"}");
        send(
            &tx,
            json!({"type":"response.output_item.done", "item":{
                "id":"a", "type":"function_call", "call_id":"call_a", "name":"read",
                "arguments":"{\"path\":\"README.md\"}"
            }}),
        );
        idle(&mut stream);
    }
}

#[test]
fn output_item_done_finishes_started_call_without_arguments_done() {
    let (tx, mut stream) = stream();
    added(&tx, "a", "read");
    assert_start(&mut stream, "a", "read");
    delta(&tx, "a", "{");
    assert_delta(&mut stream, "{");
    send(
        &tx,
        json!({"type":"response.output_item.done", "item":{
            "id":"a", "type":"function_call", "call_id":"call_a", "name":"read", "arguments":"{}"
        }}),
    );
    assert_delta(&mut stream, "}");
    assert!(matches!(next(&mut stream), StreamEvent::ToolUseEnd));
    idle(&mut stream);
    assert!(stream.streaming_tool_calls.is_empty());
}

#[test]
fn interleaved_calls_keep_unkeyed_deltas_attached_to_their_own_start() {
    let (tx, mut stream) = stream();
    added(&tx, "a", "read");
    assert_start(&mut stream, "a", "read");
    delta(&tx, "a", "{\"path\":");
    assert_delta(&mut stream, "{\"path\":");
    added(&tx, "b", "bash");
    delta(&tx, "b", "{\"command\":\"pwd\"}");
    done(&tx, "b", "{\"command\":\"pwd\"}");
    added(&tx, "c", "ls");
    delta(&tx, "c", "{");
    idle(&mut stream);
    done(&tx, "a", "{\"path\":\"README.md\"}");
    assert_delta(&mut stream, "\"README.md\"}");
    assert!(matches!(next(&mut stream), StreamEvent::ToolUseEnd));
    assert_start(&mut stream, "b", "bash");
    assert_delta(&mut stream, "{\"command\":\"pwd\"}");
    assert!(matches!(next(&mut stream), StreamEvent::ToolUseEnd));
    // The next still-incomplete tool becomes visible immediately when the
    // single-current-tool protocol allows it, not when its own arguments end.
    assert_start(&mut stream, "c", "ls");
    assert_delta(&mut stream, "{");
    idle(&mut stream);
    done(&tx, "c", "{}");
    assert_delta(&mut stream, "}");
    assert!(matches!(next(&mut stream), StreamEvent::ToolUseEnd));
    idle(&mut stream);
}

#[test]
fn late_name_releases_accumulated_arguments_without_waiting_for_done() {
    let (tx, mut stream) = stream();
    delta(&tx, "a", "{");
    idle(&mut stream);
    send(
        &tx,
        json!({"type":"response.function_call_arguments.delta", "item_id":"a",
        "call_id":"call_a", "name":"read", "delta":"\"path\":"}),
    );
    assert_start(&mut stream, "a", "read");
    assert_delta(&mut stream, "{\"path\":");
    idle(&mut stream);
}

#[test]
fn done_only_call_keeps_compatibility() {
    let (tx, mut stream) = stream();
    send(
        &tx,
        json!({"type":"response.function_call_arguments.done", "item_id":"a",
        "call_id":"call_a", "name":"read", "arguments":"{}"}),
    );
    assert_start(&mut stream, "a", "read");
    assert_delta(&mut stream, "{}");
    assert!(matches!(next(&mut stream), StreamEvent::ToolUseEnd));
    idle(&mut stream);
}

#[test]
fn null_and_empty_arguments_are_normalized_without_delaying_start() {
    for arguments in ["", " ", "null", " null "] {
        let (tx, mut stream) = stream();
        added(&tx, "a", "ls");
        assert_start(&mut stream, "a", "ls");
        for ch in arguments.chars() {
            delta(&tx, "a", &ch.to_string());
            idle(&mut stream);
        }
        done(&tx, "a", arguments);
        assert_delta(&mut stream, "{}");
        assert!(matches!(next(&mut stream), StreamEvent::ToolUseEnd));
        idle(&mut stream);
    }
}

#[test]
fn mismatched_done_arguments_fail_instead_of_corrupting_tool_input() {
    let (tx, mut stream) = stream();
    added(&tx, "a", "read");
    assert_start(&mut stream, "a", "read");
    delta(&tx, "a", "{\"path\":\"文档");
    assert_delta(&mut stream, "{\"path\":\"文档");
    done(&tx, "a", "{}");
    assert!(matches!(next(&mut stream), StreamEvent::Error { .. }));
    idle(&mut stream);
}

#[test]
fn custom_tool_input_events_stream_before_completion() {
    let (tx, mut stream) = stream();
    send(
        &tx,
        json!({"type":"response.output_item.added", "item":{
            "id":"a", "type":"custom_tool_call", "call_id":"call_a", "name":"apply_patch", "input":""
        }}),
    );
    assert_start(&mut stream, "a", "apply_patch");
    send(
        &tx,
        json!({"type":"response.custom_tool_call_input.delta", "item_id":"a", "delta":"*** Begin Patch\n"}),
    );
    assert_delta(&mut stream, "*** Begin Patch\n");
    idle(&mut stream);
    send(
        &tx,
        json!({"type":"response.custom_tool_call_input.done", "item_id":"a", "input":"*** Begin Patch\n*** End Patch"}),
    );
    assert_delta(&mut stream, "*** End Patch");
    assert!(matches!(next(&mut stream), StreamEvent::ToolUseEnd));
    idle(&mut stream);
}

// ---------------------------------------------------------------------------
// Regressions for the real `/responses` shape. Before these existed the suite
// only ever used one fixed item id, which is why the defect shipped.
// ---------------------------------------------------------------------------

const REAL_CALL_ID: &str = "call_plSP1O5CGpJV7RLo7mKfGYqK";
const REAL_ARGUMENTS: &str = r#"{"command":"echo HELLO-TOOLS-85"}"#;
/// The exact 11 fragments of the captured stream.
const REAL_FRAGMENTS: [&str; 11] = [
    "{\"", "command", "\":\"", "echo", " HEL", "LO", "-", "TOOLS", "-", "85", "\"}",
];

#[test]
fn real_form_fresh_item_ids_stream_complete_call_with_name_first() {
    let (tx, mut stream) = stream();

    real_added(&tx, 0, "l/w78RDKiz1CqrZt+w==", REAL_CALL_ID, "bash");
    // The tool name must reach the consumer before a single argument fragment.
    assert_started_as(&mut stream, REAL_CALL_ID, "bash");
    idle(&mut stream);

    let mut streamed = String::new();
    for (i, fragment) in REAL_FRAGMENTS.iter().enumerate() {
        // Every event gets its own random item id, like the real stream.
        real_delta(&tx, 0, &format!("random-delta-{i}=="), fragment);
        match next(&mut stream) {
            StreamEvent::ToolInputDelta(chunk) => {
                assert_eq!(chunk, *fragment);
                streamed.push_str(&chunk);
            }
            other => panic!("expected ToolInputDelta({fragment:?}), got {other:?}"),
        }
    }
    idle(&mut stream);
    assert_eq!(
        streamed,
        REAL_ARGUMENTS,
        "the {} fragments must concatenate to the snapshot verbatim",
        REAL_FRAGMENTS.len()
    );

    // `*_arguments.done` arrives with yet another item id and only closes the
    // call; it must not repeat the already streamed arguments.
    real_arguments_done(&tx, 0, "9frYyXme3r7hqirQeNEWk==", REAL_ARGUMENTS);
    assert_ended(&mut stream);
    idle(&mut stream);

    // `output_item.done` carries a fourth item id for the same call.
    real_item_done(
        &tx,
        0,
        "rAiYkO9MdGfZG7Id==",
        REAL_CALL_ID,
        "bash",
        REAL_ARGUMENTS,
    );
    idle(&mut stream);
    assert!(stream.streaming_tool_calls.is_empty());
}

#[test]
fn real_form_output_item_done_finishes_call_without_arguments_done() {
    let (tx, mut stream) = stream();

    real_added(&tx, 0, "item-a==", "call_snapshot", "read");
    assert_started_as(&mut stream, "call_snapshot", "read");
    real_delta(&tx, 0, "item-b==", "{\"path\":");
    assert_delta(&mut stream, "{\"path\":");

    // No `function_call_arguments.done` at all: the self-contained item snapshot
    // has to close the call, emitting only the unseen suffix.
    real_item_done(
        &tx,
        0,
        "item-c==",
        "call_snapshot",
        "read",
        "{\"path\":\"README.md\"}",
    );
    assert_delta(&mut stream, "\"README.md\"}");
    assert_ended(&mut stream);
    idle(&mut stream);
    assert!(stream.streaming_tool_calls.is_empty());
}

#[test]
fn real_form_without_output_index_reattaches_item_done_to_the_started_call() {
    let (tx, mut stream) = stream();

    // Old-style events: no `output_index`, and the item ids still differ per
    // event. The stable `call_id` on the item events is the only link left.
    send(
        &tx,
        json!({"type":"response.output_item.added", "item": {
            "type":"function_call", "id":"added-1==", "call_id":"call_reattach",
            "name":"bash", "arguments":""
        }}),
    );
    assert_started_as(&mut stream, "call_reattach", "bash");

    for (i, fragment) in REAL_FRAGMENTS.iter().enumerate() {
        send(
            &tx,
            json!({"type":"response.function_call_arguments.delta",
            "item_id":format!("delta-{i}=="), "delta":fragment}),
        );
    }
    idle(&mut stream);

    send(
        &tx,
        json!({"type":"response.output_item.done", "item": {
            "type":"function_call", "id":"done-9==", "call_id":"call_reattach",
            "name":"bash", "arguments":REAL_ARGUMENTS
        }}),
    );
    assert_delta(&mut stream, REAL_ARGUMENTS);
    assert_ended(&mut stream);
    idle(&mut stream);
    // The unkeyed fragments opened inert states of their own (no name, never
    // started, so never selected), but the call the consumer saw is closed: no
    // state is still streaming.
    assert!(
        !stream
            .streaming_tool_calls
            .values()
            .any(|state| state.started),
        "no call may still be open: {:?}",
        stream.streaming_tool_calls
    );
}

#[test]
fn real_form_interleaved_output_indexes_never_cross_arguments() {
    let (tx, mut stream) = stream();

    real_added(&tx, 0, "a-1==", "call_read", "read");
    assert_started_as(&mut stream, "call_read", "read");
    real_delta(&tx, 0, "a-2==", "{\"path\":");
    assert_delta(&mut stream, "{\"path\":");

    // A second call starts while the first is still open. Deltas are unkeyed on
    // the wire, so the second call stays invisible until the first one closes.
    real_added(&tx, 1, "b-1==", "call_bash", "bash");
    idle(&mut stream);
    real_delta(&tx, 1, "b-2==", "{\"command\":");
    idle(&mut stream);
    real_delta(&tx, 1, "b-3==", "\"pwd\"}");
    idle(&mut stream);

    real_arguments_done(&tx, 0, "a-3==", "{\"path\":\"README.md\"}");
    assert_delta(&mut stream, "\"README.md\"}");
    assert_ended(&mut stream);
    assert_started_as(&mut stream, "call_bash", "bash");
    assert_delta(&mut stream, "{\"command\":\"pwd\"}");

    real_arguments_done(&tx, 1, "b-4==", "{\"command\":\"pwd\"}");
    assert_ended(&mut stream);
    idle(&mut stream);
    assert!(stream.streaming_tool_calls.is_empty());
}

#[test]
fn real_form_custom_tool_call_input_streams_and_completes() {
    let (tx, mut stream) = stream();

    send(
        &tx,
        json!({"type":"response.output_item.added", "output_index":0, "item": {
            "id":"ct-1==", "type":"custom_tool_call", "call_id":"call_patch",
            "name":"apply_patch", "input":""
        }}),
    );
    assert_started_as(&mut stream, "call_patch", "apply_patch");

    send(
        &tx,
        json!({"type":"response.custom_tool_call_input.delta", "output_index":0,
        "item_id":"ct-2==", "delta":"*** Begin Patch\n"}),
    );
    assert_delta(&mut stream, "*** Begin Patch\n");

    send(
        &tx,
        json!({"type":"response.custom_tool_call_input.done", "output_index":0,
        "item_id":"ct-3==", "input":"*** Begin Patch\n*** End Patch"}),
    );
    assert_delta(&mut stream, "*** End Patch");
    assert_ended(&mut stream);

    send(
        &tx,
        json!({"type":"response.output_item.done", "output_index":0, "item": {
            "id":"ct-4==", "type":"custom_tool_call", "call_id":"call_patch",
            "name":"apply_patch", "input":"*** Begin Patch\n*** End Patch"
        }}),
    );
    idle(&mut stream);
    assert!(stream.streaming_tool_calls.is_empty());
}

#[test]
fn real_form_repeated_completions_never_duplicate_start_end_or_arguments() {
    let (tx, mut stream) = stream();

    real_added(&tx, 0, "dup-1==", "call_dup", "bash");
    assert_started_as(&mut stream, "call_dup", "bash");
    real_delta(&tx, 0, "dup-2==", REAL_ARGUMENTS);
    assert_delta(&mut stream, REAL_ARGUMENTS);

    for _ in 0..2 {
        real_arguments_done(&tx, 0, "dup-3==", REAL_ARGUMENTS);
        real_item_done(&tx, 0, "dup-4==", "call_dup", "bash", REAL_ARGUMENTS);
    }

    // Exactly one ToolUseEnd, and nothing left over: a duplicated start, end or
    // argument fragment would surface here.
    assert_ended(&mut stream);
    idle(&mut stream);
    assert!(stream.streaming_tool_calls.is_empty());
}

#[test]
fn real_form_late_arguments_delta_after_item_done_is_ignored() {
    let (tx, mut stream) = stream();

    real_added(&tx, 0, "late-1==", "call_late", "bash");
    assert_started_as(&mut stream, "call_late", "bash");
    real_delta(&tx, 0, "late-2==", "{\"command\":");
    assert_delta(&mut stream, "{\"command\":");

    // The `output_item.done` snapshot closes the call with the unseen suffix.
    real_item_done(&tx, 0, "late-3==", "call_late", "bash", REAL_ARGUMENTS);
    assert_delta(&mut stream, "\"echo HELLO-TOOLS-85\"}");
    assert_ended(&mut stream);

    // A late or duplicated fragment on the same `output_index` must not reopen
    // the finished call, re-emit arguments, or push a second end. With the old
    // `item_id` keying nothing matched `completed_tool_items`, so these opened
    // fresh inert states instead of being dropped.
    real_delta(&tx, 0, "late-4==", "\"echo HELLO-TOOLS-85\"}");
    idle(&mut stream);
    real_arguments_done(&tx, 0, "late-5==", REAL_ARGUMENTS);
    idle(&mut stream);
    real_item_done(&tx, 0, "late-6==", "call_late", "bash", REAL_ARGUMENTS);
    idle(&mut stream);
    assert!(stream.streaming_tool_calls.is_empty());
}
