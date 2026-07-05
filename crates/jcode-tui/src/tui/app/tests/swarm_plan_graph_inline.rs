// Tests for the SwarmPlan -> inline chat plan-graph pipeline and the
// plan-scope notification quieting (status line only, no chat card).

// The SwarmPlan handler reads the process-global JCODE_ENABLE_MERMAID env
// var, so every test in this file serializes on one lock and the env-mutating
// test restores the previous value via a drop guard.
static SWARM_PLAN_MERMAID_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn swarm_plan_mermaid_env_lock() -> std::sync::MutexGuard<'static, ()> {
    SWARM_PLAN_MERMAID_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

struct MermaidEnvGuard {
    prev: Option<std::ffi::OsString>,
}

impl MermaidEnvGuard {
    fn set(value: &str) -> Self {
        let prev = std::env::var_os("JCODE_ENABLE_MERMAID");
        // SAFETY: guarded by SWARM_PLAN_MERMAID_ENV_LOCK held by the caller
        // for the guard's whole lifetime, so no concurrent env access races
        // with this write within the tests that consult this variable.
        unsafe { std::env::set_var("JCODE_ENABLE_MERMAID", value) };
        Self { prev }
    }
}

impl Drop for MermaidEnvGuard {
    fn drop(&mut self) {
        match self.prev.take() {
            // SAFETY: see MermaidEnvGuard::set.
            Some(prev) => unsafe { std::env::set_var("JCODE_ENABLE_MERMAID", prev) },
            None => unsafe { std::env::remove_var("JCODE_ENABLE_MERMAID") },
        }
    }
}

fn swarm_plan_graph_item(id: &str, content: &str) -> crate::plan::PlanItem {
    crate::plan::PlanItem {
        content: content.to_string(),
        status: "running".to_string(),
        priority: "high".to_string(),
        id: id.to_string(),
        subsystem: None,
        file_scope: Vec::new(),
        blocked_by: Vec::new(),
        assigned_to: Some("worker-fox".to_string()),
    }
}

fn swarm_plan_event(version: u64, items: Vec<crate::plan::PlanItem>) -> crate::protocol::ServerEvent {
    crate::protocol::ServerEvent::SwarmPlan {
        swarm_id: "test-swarm".to_string(),
        version,
        items,
        participants: vec!["session_a".to_string()],
        reason: None,
        summary: None,
    }
}

fn plan_graph_titles(app: &App) -> Vec<String> {
    app.display_messages()
        .iter()
        .filter(|m| {
            m.role == "swarm"
                && m.title
                    .as_deref()
                    .is_some_and(|t| t.starts_with("Plan graph · "))
        })
        .filter_map(|m| m.title.clone())
        .collect()
}

fn rendered_lines_to_text(lines: &[ratatui::text::Line<'static>]) -> String {
    lines
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Restores the process-global markdown diagram-mode override on drop.
struct DiagramModeOverrideGuard {
    prev: Option<crate::config::DiagramDisplayMode>,
}

impl DiagramModeOverrideGuard {
    fn pinned() -> Self {
        let prev = crate::tui::markdown::get_diagram_mode_override();
        crate::tui::markdown::set_diagram_mode_override(Some(
            crate::config::DiagramDisplayMode::Pinned,
        ));
        Self { prev }
    }
}

impl Drop for DiagramModeOverrideGuard {
    fn drop(&mut self) {
        crate::tui::markdown::set_diagram_mode_override(self.prev);
    }
}

// ---------------------------------------------------------------------------
// wiring-audit.pinned-pane-verify: behavioral checks for the pinned diagram
// pane vs. the upsert-in-place plan-graph message. ACTIVE_DIAGRAMS is a
// process-global registry (mermaid_active.rs), so these tests serialize on
// the same lock the other diagram-mutating tests use
// (`scroll_render_test_lock`) plus this file's mermaid env lock.
// ---------------------------------------------------------------------------

/// Claims 1 + 2: replacing the trailing plan-graph message in place leaves
/// the previously rendered diagram registered in ACTIVE_DIAGRAMS (no
/// unregistration path), so the pinned pane count inflates and Ctrl+arrow
/// cycling walks stale plan versions. Refinement of claim 1: accumulation is
/// per distinct *mermaid content* hash, not per plan version number. A
/// version bump whose items (and therefore graph source) are unchanged does
/// NOT add an entry.
#[test]
fn test_upsert_in_place_plan_bump_accumulates_stale_active_diagrams() {
    let _env_lock = swarm_plan_mermaid_env_lock();
    let _render_lock = scroll_render_test_lock();
    let _mode_guard = DiagramModeOverrideGuard::pinned();
    let mut app = create_test_app();
    app.diagram_mode = crate::config::DiagramDisplayMode::Pinned;
    app.diagram_pane_enabled = true;
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let mut remote = crate::tui::backend::RemoteConnection::dummy();
    remote.mark_history_loaded();

    crate::tui::mermaid::clear_active_diagrams();

    // v1: task running.
    app.handle_server_event(
        swarm_plan_event(1, vec![swarm_plan_graph_item("haiku-1", "write a haiku")]),
        &mut remote,
    );
    let v1_msg = app
        .display_messages()
        .iter()
        .rev()
        .find(|m| m.role == "swarm")
        .expect("plan graph message")
        .clone();
    // Render through the real swarm-message markdown path (synchronous
    // mermaid render outside the deferred draw context) so the diagram
    // registers exactly like a transcript render would.
    let lines = crate::tui::ui::render_swarm_message(
        &v1_msg,
        80,
        crate::config::DiffDisplayMode::Inline,
    );
    assert!(
        !rendered_lines_to_text(&lines).is_empty(),
        "swarm plan message should render"
    );
    assert_eq!(
        crate::tui::mermaid::active_diagram_count(),
        1,
        "first plan render registers one active diagram"
    );
    let v1_hash = crate::tui::mermaid::get_active_diagrams()[0].hash;

    // v2: same task flips to completed -> graph content changes -> the
    // trailing message is replaced IN PLACE (one transcript message)...
    let mut done = swarm_plan_graph_item("haiku-1", "write a haiku");
    done.status = "completed".to_string();
    app.handle_server_event(swarm_plan_event(2, vec![done.clone()]), &mut remote);
    assert_eq!(
        plan_graph_titles(&app),
        vec!["Plan graph · v2".to_string()],
        "upsert must keep a single transcript plan-graph message"
    );
    let v2_msg = app
        .display_messages()
        .iter()
        .rev()
        .find(|m| m.role == "swarm")
        .expect("plan graph message")
        .clone();
    assert_ne!(v1_msg.content, v2_msg.content, "status flip changes graph source");
    let _ = crate::tui::ui::render_swarm_message(
        &v2_msg,
        80,
        crate::config::DiffDisplayMode::Inline,
    );

    // ...but ACTIVE_DIAGRAMS now holds BOTH versions: nothing unregisters
    // the stale v1 diagram when its transcript message was overwritten.
    let diagrams = crate::tui::mermaid::get_active_diagrams();
    assert_eq!(
        diagrams.len(),
        2,
        "claim 1 CONFIRMED: in-place plan bump leaks a stale ACTIVE_DIAGRAMS entry"
    );
    assert_ne!(diagrams[0].hash, v1_hash, "newest-first: index 0 is the v2 diagram");
    assert_eq!(
        diagrams[1].hash, v1_hash,
        "the replaced v1 diagram is still registered (stale)"
    );

    // Ctrl+arrow cycling reaches the stale version and the counter reads 2.
    app.diagram_index = 0;
    app.cycle_diagram(1);
    assert_eq!(app.diagram_index, 1, "cycling lands on the stale v1 diagram");
    assert_eq!(
        app.last_visible_diagram_hash,
        Some(v1_hash),
        "claim 2 CONFIRMED: the pane can show the outdated plan version"
    );
    let notice = crate::tui::TuiState::status_notice(&app);
    assert_eq!(
        notice.as_deref(),
        Some("Diagram 2/2"),
        "counter inflates to include the stale version"
    );

    // Refinement: a version bump with UNCHANGED items produces identical
    // mermaid source, so it does NOT add a third entry (dedup is by content
    // hash, `register_active_diagram` moves the entry to the fresh end).
    app.handle_server_event(swarm_plan_event(3, vec![done]), &mut remote);
    let v3_msg = app
        .display_messages()
        .iter()
        .rev()
        .find(|m| m.role == "swarm")
        .expect("plan graph message")
        .clone();
    assert_eq!(
        v2_msg.content, v3_msg.content,
        "version-only bump keeps identical graph content"
    );
    let _ = crate::tui::ui::render_swarm_message(
        &v3_msg,
        80,
        crate::config::DiffDisplayMode::Inline,
    );
    assert_eq!(
        crate::tui::mermaid::active_diagram_count(),
        2,
        "claim 1 REFINED: accumulation is per distinct graph content, not per version"
    );

    crate::tui::mermaid::clear_active_diagrams();
}

/// Claim 3: `get_active_diagrams` returns newest-first (insertion order
/// reversed), and `diagram_index` is positional, so a user parked at index
/// k > 0 is silently shifted to a different diagram whenever a new diagram
/// registers. Nothing re-anchors the selection by hash.
#[test]
fn test_new_registration_silently_shifts_parked_diagram_selection() {
    let _render_lock = scroll_render_test_lock();
    let mut app = create_test_app();
    app.diagram_mode = crate::config::DiagramDisplayMode::Pinned;
    app.diagram_pane_enabled = true;

    crate::tui::mermaid::clear_active_diagrams();
    crate::tui::mermaid::register_active_diagram(0xA, 100, 80, None);
    crate::tui::mermaid::register_active_diagram(0xB, 100, 80, None);
    crate::tui::mermaid::register_active_diagram(0xC, 100, 80, None);

    // Newest-first: [C, B, A]. Park the user on B (index 1).
    let before = crate::tui::mermaid::get_active_diagrams();
    assert_eq!(
        before.iter().map(|d| d.hash).collect::<Vec<_>>(),
        vec![0xC, 0xB, 0xA]
    );
    app.diagram_index = 1;
    app.sync_diagram_fit_context();
    assert_eq!(app.last_visible_diagram_hash, Some(0xB));

    // A new diagram registers (e.g. a plan bump): everything shifts by one.
    crate::tui::mermaid::register_active_diagram(0xD, 100, 80, None);
    let after = crate::tui::mermaid::get_active_diagrams();
    assert_eq!(
        after.iter().map(|d| d.hash).collect::<Vec<_>>(),
        vec![0xD, 0xC, 0xB, 0xA]
    );
    assert_eq!(
        after[app.diagram_index].hash, 0xC,
        "claim 3 CONFIRMED: index 1 now points at C, not the parked B"
    );
    // normalize_diagram_state does not re-anchor by hash; it only clamps the
    // index, so the silent shift persists (the fit-context sync then resets
    // the viewport because the hash under the index changed).
    app.normalize_diagram_state();
    assert_eq!(app.diagram_index, 1, "index is kept, content under it changed");
    assert_eq!(
        app.last_visible_diagram_hash,
        Some(0xC),
        "selection silently moved from B to C"
    );

    // Re-registering an EXISTING hash also reorders (moves it to front),
    // which shifts a parked selection the same way.
    app.diagram_index = 2; // parked on B in [D, C, B, A]
    app.sync_diagram_fit_context();
    assert_eq!(app.last_visible_diagram_hash, Some(0xB));
    crate::tui::mermaid::register_active_diagram(0xA, 100, 80, None);
    let reordered = crate::tui::mermaid::get_active_diagrams();
    assert_eq!(
        reordered.iter().map(|d| d.hash).collect::<Vec<_>>(),
        vec![0xA, 0xD, 0xC, 0xB],
        "re-registration moves an existing hash to the fresh end"
    );
    assert_eq!(
        reordered[app.diagram_index].hash, 0xC,
        "parked index 2 shifted from B to C without any user action"
    );

    crate::tui::mermaid::clear_active_diagrams();
}

/// Claim 5: when the 129th distinct diagram registers, ACTIVE_DIAGRAMS_MAX
/// eviction drops the oldest entry. If the pane was showing that entry, the
/// index silently lands on a different diagram (no crash, no reset: the
/// count stays at the cap so `normalize_diagram_state` never clamps).
#[test]
fn test_active_diagrams_cap_eviction_swaps_currently_shown_diagram() {
    let _render_lock = scroll_render_test_lock();
    let mut app = create_test_app();
    app.diagram_mode = crate::config::DiagramDisplayMode::Pinned;
    app.diagram_pane_enabled = true;

    crate::tui::mermaid::clear_active_diagrams();
    for i in 1..=128u64 {
        crate::tui::mermaid::register_active_diagram(i, 100, 80, None);
    }
    assert_eq!(crate::tui::mermaid::active_diagram_count(), 128);

    // Park on the OLDEST diagram (hash 1, last position in newest-first
    // order).
    app.diagram_index = 127;
    app.sync_diagram_fit_context();
    assert_eq!(app.last_visible_diagram_hash, Some(1));

    // The 129th diagram evicts hash 1 (the one being shown).
    crate::tui::mermaid::register_active_diagram(129, 100, 80, None);
    let diagrams = crate::tui::mermaid::get_active_diagrams();
    assert_eq!(diagrams.len(), 128, "cap holds at ACTIVE_DIAGRAMS_MAX");
    assert!(
        !diagrams.iter().any(|d| d.hash == 1),
        "the shown diagram was evicted from the registry"
    );

    // Count stayed at the cap, so index 127 is still in range: no clamp, no
    // reset, the pane just shows a different diagram.
    app.normalize_diagram_state();
    assert_eq!(app.diagram_index, 127);
    assert_eq!(
        app.last_visible_diagram_hash,
        Some(2),
        "claim 5 CONFIRMED: eviction silently swaps the shown diagram (1 -> 2)"
    );

    crate::tui::mermaid::clear_active_diagrams();
}

/// Claim 4: the chat body prepare path renders EVERY display message at full
/// fidelity regardless of scroll position (`prepare_body` does not take the
/// scroll offset; viewport windowing only slices already-prepared lines in
/// `draw_messages`). So a plan-graph message scrolled far off-screen still
/// goes through the mermaid pipeline and registers in ACTIVE_DIAGRAMS.
/// (`render_markdown_lazy`'s visible-range skipping is not used by the chat
/// body path, and even that function renders mermaid blocks unconditionally.)
#[test]
fn test_offscreen_plan_graph_message_still_registers_active_diagram() {
    let _env_lock = swarm_plan_mermaid_env_lock();
    let _render_lock = scroll_render_test_lock();
    let _mode_guard = DiagramModeOverrideGuard::pinned();
    let mut app = create_test_app();
    app.diagram_mode = crate::config::DiagramDisplayMode::Pinned;
    app.diagram_pane_enabled = true;
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let mut remote = crate::tui::backend::RemoteConnection::dummy();
    remote.mark_history_loaded();

    crate::tui::mermaid::clear_active_diagrams();

    // Plan graph message lands first...
    app.handle_server_event(
        swarm_plan_event(7, vec![swarm_plan_graph_item("haiku-1", "write a haiku")]),
        &mut remote,
    );
    // ...then enough transcript follows to push it far above a 24-row
    // viewport (tail-follow keeps the view at the bottom).
    for i in 0..80 {
        app.push_display_message(DisplayMessage::system(format!("filler line {i}")));
    }

    // Full-frame draw through the real UI entry point (TestBackend). The
    // draw wraps rendering in the deferred mermaid context; an uncached
    // diagram is queued to the background worker, so poll for registration.
    let backend = ratatui::backend::TestBackend::new(80, 24);
    let mut terminal = ratatui::Terminal::new(backend).expect("test terminal");
    terminal
        .draw(|f| crate::tui::ui::draw(f, &app))
        .expect("draw failed");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut registered = crate::tui::mermaid::active_diagram_count();
    while registered == 0 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(25));
        // Redraw so a completed deferred render (epoch bump) re-runs the
        // message render and registers via the now-warm cache.
        terminal
            .draw(|f| crate::tui::ui::draw(f, &app))
            .expect("draw failed");
        registered = crate::tui::mermaid::active_diagram_count();
    }
    assert!(
        registered >= 1,
        "claim 4 RESOLVED: off-screen plan-graph messages DO register \
         (body prepare renders all messages; windowing only slices lines)"
    );

    crate::tui::mermaid::clear_active_diagrams();
}

/// wiring-audit.session-switch-diagram-leak: a session-changing History event
/// clears display messages and the swarm plan snapshot (server_events.rs
/// ~1592, ~1636-1639) but never touches the process-global ACTIVE_DIAGRAMS
/// registry (mermaid_active.rs: `clear_active_diagrams` is only called from
/// debug/bench/test paths). A plan-graph diagram registered in the PREVIOUS
/// session therefore survives the switch: it still counts in the pinned pane
/// counter, is reachable via Ctrl+arrow cycling, and is listed by
/// `get_active_diagrams` (the Margin info widget source), even though its
/// transcript message is gone.
#[test]
fn test_session_change_history_leaks_previous_session_active_diagram() {
    let _env_lock = swarm_plan_mermaid_env_lock();
    let _render_lock = scroll_render_test_lock();
    let _mode_guard = DiagramModeOverrideGuard::pinned();
    let mut app = create_test_app();
    app.diagram_mode = crate::config::DiagramDisplayMode::Pinned;
    app.diagram_pane_enabled = true;
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let mut remote = crate::tui::backend::RemoteConnection::dummy();
    remote.mark_history_loaded();

    crate::tui::mermaid::clear_active_diagrams();

    // Session A: a plan-graph message lands and renders, registering its
    // diagram in the global registry (same path as the transcript render).
    app.remote_session_id = Some("session_old".to_string());
    app.handle_server_event(
        swarm_plan_event(1, vec![swarm_plan_graph_item("haiku-1", "write a haiku")]),
        &mut remote,
    );
    let plan_msg = app
        .display_messages()
        .iter()
        .rev()
        .find(|m| m.role == "swarm")
        .expect("plan graph message")
        .clone();
    let _ = crate::tui::ui::render_swarm_message(
        &plan_msg,
        80,
        crate::config::DiffDisplayMode::Inline,
    );
    assert_eq!(
        crate::tui::mermaid::active_diagram_count(),
        1,
        "session A plan render registers one active diagram"
    );
    let stale_hash = crate::tui::mermaid::get_active_diagrams()[0].hash;

    // Switch to session B via a session-changing History event. The handler
    // clears the transcript and the swarm plan snapshot...
    app.handle_server_event(history_event_for_session("session_new"), &mut remote);
    assert!(
        plan_graph_titles(&app).is_empty(),
        "session switch removes the plan-graph transcript message"
    );
    assert!(
        app.swarm_plan_items.is_empty(),
        "session switch clears the swarm plan snapshot"
    );
    assert_eq!(app.swarm_plan_version, None);

    // ...but the diagram registry is NOT re-scoped: the previous session's
    // plan graph is still registered and returned to the info widget.
    let diagrams = crate::tui::mermaid::get_active_diagrams();
    assert_eq!(
        diagrams.len(),
        1,
        "LEAK CONFIRMED: session-changing History leaves the previous \
         session's diagram in ACTIVE_DIAGRAMS"
    );
    assert_eq!(
        diagrams[0].hash, stale_hash,
        "the surviving entry is exactly the stale session-A plan graph"
    );

    // The pinned pane still targets it: the fit-context sync anchors on the
    // stale hash and cycling reports it in the counter, with no transcript
    // message backing it anymore.
    app.diagram_index = 0;
    app.sync_diagram_fit_context();
    assert_eq!(
        app.last_visible_diagram_hash,
        Some(stale_hash),
        "pinned pane shows the previous session's diagram after the switch"
    );
    app.cycle_diagram(1);
    let notice = crate::tui::TuiState::status_notice(&app);
    assert_eq!(
        notice.as_deref(),
        Some("Diagram 1/1"),
        "Ctrl+arrow cycling counts the stale cross-session diagram"
    );

    crate::tui::mermaid::clear_active_diagrams();
}

fn history_event_for_session(session_id: &str) -> crate::protocol::ServerEvent {
    crate::protocol::ServerEvent::History {
        id: 1,
        session_id: session_id.to_string(),
        messages: vec![],
        images: vec![],
        provider_name: Some("claude".to_string()),
        provider_model: Some("claude-sonnet-4-20250514".to_string()),
        subagent_model: None,
        autoreview_enabled: None,
        autojudge_enabled: None,
        available_models: vec![],
        available_model_routes: vec![],
        mcp_servers: vec![],
        skills: vec![],
        total_tokens: None,
        token_usage_totals: None,
        all_sessions: vec![],
        client_count: None,
        is_canary: None,
        reload_recovery: None,
        server_version: None,
        server_name: None,
        server_icon: None,
        server_has_update: None,
        was_interrupted: None,
        connection_type: None,
        status_detail: None,
        upstream_provider: None,
        resolved_credential: None,
        reasoning_effort: None,
        service_tier: None,
        compaction_mode: crate::config::CompactionMode::Reactive,
        activity: None,
        side_panel: crate::side_panel::SidePanelSnapshot::default(),
    }
}

#[test]
fn test_swarm_plan_event_pushes_inline_plan_graph_message() {
    let _env_lock = swarm_plan_mermaid_env_lock();
    let mut app = create_test_app();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let mut remote = crate::tui::backend::RemoteConnection::dummy();
    remote.mark_history_loaded();

    let item = crate::plan::PlanItem {
        content: "write a haiku".to_string(),
        status: "running".to_string(),
        priority: "high".to_string(),
        id: "haiku-1".to_string(),
        subsystem: None,
        file_scope: Vec::new(),
        blocked_by: Vec::new(),
        assigned_to: Some("worker-fox".to_string()),
    };

    app.handle_server_event(
        crate::protocol::ServerEvent::SwarmPlan {
            swarm_id: "test-swarm".to_string(),
            version: 3,
            items: vec![item.clone()],
            participants: vec!["session_a".to_string()],
            reason: None,
            summary: None,
        },
        &mut remote,
    );

    let graph_msg = app
        .display_messages()
        .iter()
        .find(|m| m.role == "swarm" && m.title.as_deref() == Some("Plan graph · v3"))
        .expect("SwarmPlan event should push an inline plan graph chat message");
    assert!(
        graph_msg.content.starts_with("```mermaid\nflowchart TD"),
        "plan graph message should carry a mermaid fence: {}",
        &graph_msg.content[..graph_msg.content.len().min(80)]
    );
    assert!(
        graph_msg.content.contains("t_haiku_1") && graph_msg.content.contains("write a haiku"),
        "graph should include the task node: {}",
        graph_msg.content
    );

    // A follow-up plan version updates the trailing graph message in place
    // instead of stacking a second diagram.
    let count_before = app.display_messages().len();
    let mut updated = item;
    updated.status = "completed".to_string();
    app.handle_server_event(
        crate::protocol::ServerEvent::SwarmPlan {
            swarm_id: "test-swarm".to_string(),
            version: 4,
            items: vec![updated],
            participants: vec!["session_a".to_string()],
            reason: None,
            summary: None,
        },
        &mut remote,
    );
    assert_eq!(
        app.display_messages().len(),
        count_before,
        "rapid plan updates must coalesce into the trailing plan graph message"
    );
    let graph_count = app
        .display_messages()
        .iter()
        .filter(|m| {
            m.role == "swarm"
                && m.title
                    .as_deref()
                    .is_some_and(|t| t.starts_with("Plan graph · "))
        })
        .count();
    assert_eq!(graph_count, 1, "only one trailing plan graph message expected");
    let latest = app
        .display_messages()
        .iter()
        .find(|m| m.title.as_deref() == Some("Plan graph · v4"))
        .expect("trailing graph message should carry the new version");
    assert!(latest.content.contains(":::done"), "updated status should recolor the node");
}

#[test]
fn test_plan_scope_notification_stays_off_the_transcript() {
    let _env_lock = swarm_plan_mermaid_env_lock();
    let mut app = create_test_app();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let mut remote = crate::tui::backend::RemoteConnection::dummy();
    remote.mark_history_loaded();

    let count_before = app.display_messages().len();
    app.handle_server_event(
        crate::protocol::ServerEvent::Notification {
            from_session: "session_dove_123".to_string(),
            from_name: Some("dove".to_string()),
            notification_type: crate::protocol::NotificationType::Message {
                scope: Some("plan".to_string()),
                channel: None,
                tldr: None,
            },
            message: "Plan updated: task 'fix-debug-tests' assigned to session_blowfish_9."
                .to_string(),
        },
        &mut remote,
    );

    assert_eq!(
        app.display_messages().len(),
        count_before,
        "plan-scope churn must not add chat messages"
    );

    // Non-plan swarm notifications still land in the transcript.
    app.handle_server_event(
        crate::protocol::ServerEvent::Notification {
            from_session: "session_dove_123".to_string(),
            from_name: Some("dove".to_string()),
            notification_type: crate::protocol::NotificationType::Message {
                scope: Some("dm".to_string()),
                channel: None,
                tldr: None,
            },
            message: "DM from dove: hello".to_string(),
        },
        &mut remote,
    );
    assert_eq!(
        app.display_messages().len(),
        count_before + 1,
        "dm notifications keep their chat card"
    );
}

#[test]
fn test_non_plan_swarm_message_between_plan_versions_stacks_second_plan_graph() {
    // Wiring-audit claim 1: a non-plan-scope swarm chat card (e.g. a DM)
    // landing between two SwarmPlan events breaks trailing coalescing, so a
    // second "Plan graph · vN" diagram is appended instead of updating the
    // first in place.
    let _env_lock = swarm_plan_mermaid_env_lock();
    let mut app = create_test_app();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let mut remote = crate::tui::backend::RemoteConnection::dummy();
    remote.mark_history_loaded();

    app.handle_server_event(
        swarm_plan_event(3, vec![swarm_plan_graph_item("haiku-1", "write a haiku")]),
        &mut remote,
    );
    assert_eq!(plan_graph_titles(&app), vec!["Plan graph · v3".to_string()]);

    // A DM notification lands as a normal swarm chat card between the two
    // plan versions.
    app.handle_server_event(
        crate::protocol::ServerEvent::Notification {
            from_session: "session_dove_123".to_string(),
            from_name: Some("dove".to_string()),
            notification_type: crate::protocol::NotificationType::Message {
                scope: Some("dm".to_string()),
                channel: None,
                tldr: None,
            },
            message: "DM from dove: hello".to_string(),
        },
        &mut remote,
    );

    let mut updated = swarm_plan_graph_item("haiku-1", "write a haiku");
    updated.status = "completed".to_string();
    app.handle_server_event(swarm_plan_event(4, vec![updated]), &mut remote);

    let titles = plan_graph_titles(&app);
    assert_eq!(
        titles,
        vec!["Plan graph · v3".to_string(), "Plan graph · v4".to_string()],
        "a swarm DM between plan versions breaks coalescing and stacks a second diagram: {titles:?}"
    );
}

#[test]
fn test_out_of_order_older_swarm_plan_version_overwrites_newer_plan_graph_in_place() {
    // Wiring-audit claim 2: there is no version monotonicity guard, so an
    // out-of-order (older) SwarmPlan event overwrites a newer diagram.
    let _env_lock = swarm_plan_mermaid_env_lock();
    let mut app = create_test_app();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let mut remote = crate::tui::backend::RemoteConnection::dummy();
    remote.mark_history_loaded();

    let mut newer_item = swarm_plan_graph_item("haiku-1", "write a haiku");
    newer_item.status = "completed".to_string();
    app.handle_server_event(swarm_plan_event(5, vec![newer_item]), &mut remote);
    assert_eq!(plan_graph_titles(&app), vec!["Plan graph · v5".to_string()]);

    // A stale (older-version) broadcast arrives afterwards.
    app.handle_server_event(
        swarm_plan_event(4, vec![swarm_plan_graph_item("haiku-1", "write a haiku")]),
        &mut remote,
    );

    let titles = plan_graph_titles(&app);
    assert_eq!(
        titles,
        vec!["Plan graph · v4".to_string()],
        "an older plan version overwrites the newer trailing diagram in place (no monotonicity guard): {titles:?}"
    );
    assert_eq!(app.swarm_plan_version, Some(4), "snapshot state also regresses to the older version");
}

#[test]
fn test_history_session_change_clears_swarm_plan_state_and_plan_graph_does_not_reappear() {
    // Wiring-audit claim 3: the History server event clears swarm_plan_items
    // (server_events.rs ~1637) on session change and the plan-graph chat
    // message does not reappear from the restored history.
    let _env_lock = swarm_plan_mermaid_env_lock();
    let mut app = create_test_app();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let mut remote = crate::tui::backend::RemoteConnection::dummy();
    remote.mark_history_loaded();

    app.remote_session_id = Some("session_same".to_string());
    app.handle_server_event(
        swarm_plan_event(3, vec![swarm_plan_graph_item("haiku-1", "write a haiku")]),
        &mut remote,
    );
    assert!(!app.swarm_plan_items.is_empty());
    assert_eq!(plan_graph_titles(&app).len(), 1);

    // Same-session history refresh does NOT clear the plan snapshot or the
    // inline diagram (the clearing block is scoped to session_changed).
    app.handle_server_event(history_event_for_session("session_same"), &mut remote);
    assert!(
        !app.swarm_plan_items.is_empty(),
        "same-session history refresh keeps swarm_plan_items"
    );
    assert_eq!(
        plan_graph_titles(&app).len(),
        1,
        "same-session history refresh keeps the inline plan graph message"
    );

    // Session-changing history clears the plan snapshot and the diagram does
    // not come back from the (empty) restored history.
    app.handle_server_event(history_event_for_session("session_other"), &mut remote);
    assert!(
        app.swarm_plan_items.is_empty(),
        "session-change history must clear swarm_plan_items"
    );
    assert_eq!(app.swarm_plan_version, None);
    assert_eq!(app.swarm_plan_swarm_id, None);
    assert!(
        plan_graph_titles(&app).is_empty(),
        "plan graph message must not reappear after history restore: {:?}",
        plan_graph_titles(&app)
    );
}

#[test]
fn test_swarm_plan_pushes_no_plan_graph_message_when_mermaid_disabled() {
    // Wiring-audit claim 4: with JCODE_ENABLE_MERMAID=0 the SwarmPlan handler
    // pushes no inline plan-graph message (raw mermaid source would be noise),
    // while the plan snapshot state is still applied.
    let _env_lock = swarm_plan_mermaid_env_lock();
    let _env_guard = MermaidEnvGuard::set("0");
    let mut app = create_test_app();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let mut remote = crate::tui::backend::RemoteConnection::dummy();
    remote.mark_history_loaded();

    let count_before = app.display_messages().len();
    app.handle_server_event(
        swarm_plan_event(7, vec![swarm_plan_graph_item("haiku-1", "write a haiku")]),
        &mut remote,
    );

    assert_eq!(
        app.display_messages().len(),
        count_before,
        "JCODE_ENABLE_MERMAID=0 must suppress the inline plan graph message"
    );
    assert!(plan_graph_titles(&app).is_empty());
    assert_eq!(
        app.swarm_plan_version,
        Some(7),
        "plan snapshot state still applies even when the diagram is suppressed"
    );
    assert!(!app.swarm_plan_items.is_empty());
}

// ---------------------------------------------------------------------------
// wiring-audit.transcript-clear-diagram-leak: the session-switch audit only
// covered the remote History `session_changed` path (server_events.rs ~1637),
// which is the ONLY transcript-clear path that resets swarm_plan_items /
// swarm_plan_version / swarm_plan_swarm_id. Every other transcript-clear path
// goes through `clear_display_messages` (state_ui_messages.rs:392), which
// touches neither the process-global ACTIVE_DIAGRAMS registry
// (mermaid_active.rs; `clear_active_diagrams` is only called from
// debug_bench.rs) nor the swarm plan snapshot fields. The tests below pin
// that behavior for each path:
//   1. local `/clear` -> reset_current_session (commands.rs:1703 ->
//      commands_review.rs:270)
//   2. local `/rewind N` (commands.rs ~2004) and `/rewind undo`
//      (commands.rs ~1933)
//   3. local session recovery (conversation_state.rs ~809)
//   4. remote `/clear` (remote/key_handling.rs ~1610)
//   5. disconnected Ctrl+L (remote.rs ~1670; connected-remote Ctrl+L at
//      key_handling.rs ~611 and local Ctrl+L at input.rs ~1922 are no-ops
//      that clear nothing)
// ---------------------------------------------------------------------------

/// Seeds one plan-graph message via a SwarmPlan event and renders it through
/// the real swarm-message markdown path so its diagram registers in
/// ACTIVE_DIAGRAMS exactly like a transcript render would. Returns the
/// registered diagram hash.
fn seed_rendered_plan_graph(
    app: &mut App,
    remote: &mut crate::tui::backend::RemoteConnection,
) -> u64 {
    crate::tui::mermaid::clear_active_diagrams();
    app.handle_server_event(
        swarm_plan_event(1, vec![swarm_plan_graph_item("haiku-1", "write a haiku")]),
        remote,
    );
    let plan_msg = app
        .display_messages()
        .iter()
        .rev()
        .find(|m| m.role == "swarm")
        .expect("plan graph message")
        .clone();
    let _ =
        crate::tui::ui::render_swarm_message(&plan_msg, 80, crate::config::DiffDisplayMode::Inline);
    assert_eq!(
        crate::tui::mermaid::active_diagram_count(),
        1,
        "seed: plan render registers exactly one active diagram"
    );
    assert!(!app.swarm_plan_items.is_empty(), "seed: plan snapshot applied");
    crate::tui::mermaid::get_active_diagrams()[0].hash
}

/// Shared post-clear assertions: the plan-graph transcript message is gone,
/// but the ACTIVE_DIAGRAMS registry still holds the orphaned diagram (the
/// pinned pane keeps showing it) and the swarm plan snapshot fields survive
/// untouched.
fn assert_transcript_clear_leaks_diagram_and_plan_state(
    app: &mut App,
    stale_hash: u64,
    path: &str,
) {
    assert!(
        plan_graph_titles(app).is_empty(),
        "{path}: transcript wiped, no plan-graph message remains"
    );
    let diagrams = crate::tui::mermaid::get_active_diagrams();
    assert_eq!(
        diagrams.len(),
        1,
        "{path}: LEAK CONFIRMED - ACTIVE_DIAGRAMS still holds the cleared transcript's diagram"
    );
    assert_eq!(
        diagrams[0].hash, stale_hash,
        "{path}: the surviving entry is exactly the stale plan graph"
    );
    assert!(
        !app.swarm_plan_items.is_empty(),
        "{path}: STALE STATE CONFIRMED - swarm_plan_items persist after the transcript is wiped"
    );
    assert_eq!(
        app.swarm_plan_version,
        Some(1),
        "{path}: stale swarm_plan_version persists"
    );
    assert_eq!(
        app.swarm_plan_swarm_id.as_deref(),
        Some("test-swarm"),
        "{path}: stale swarm_plan_swarm_id persists"
    );
    // The pinned pane still anchors on the orphaned diagram even though no
    // transcript message backs it anymore.
    app.diagram_index = 0;
    app.sync_diagram_fit_context();
    assert_eq!(
        app.last_visible_diagram_hash,
        Some(stale_hash),
        "{path}: pinned pane still shows the orphaned plan graph"
    );
}

/// Path 1: local `/clear` (commands.rs:1703 -> reset_current_session at
/// commands_review.rs:270). It resets the session, provider messages, and
/// display messages but never calls `clear_active_diagrams` and never touches
/// the swarm plan snapshot fields.
#[test]
fn test_local_clear_command_leaves_stale_active_diagram_and_swarm_plan_state() {
    let _env_lock = swarm_plan_mermaid_env_lock();
    let _render_lock = scroll_render_test_lock();
    let _mode_guard = DiagramModeOverrideGuard::pinned();
    let mut app = create_test_app();
    app.diagram_mode = crate::config::DiagramDisplayMode::Pinned;
    app.diagram_pane_enabled = true;
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let mut remote = crate::tui::backend::RemoteConnection::dummy();
    remote.mark_history_loaded();

    let stale_hash = seed_rendered_plan_graph(&mut app, &mut remote);

    assert!(super::commands::handle_session_command(&mut app, "/clear"));

    assert_transcript_clear_leaks_diagram_and_plan_state(
        &mut app,
        stale_hash,
        "local /clear (reset_current_session)",
    );

    crate::tui::mermaid::clear_active_diagrams();
}

/// Path 2: local `/rewind N` (commands.rs ~2004) and `/rewind undo`
/// (commands.rs ~1933). Both rebuild the transcript via
/// `clear_display_messages` + re-render; neither unregisters the plan-graph
/// diagram nor resets the swarm plan snapshot.
#[test]
fn test_local_rewind_and_undo_leave_stale_active_diagram_and_swarm_plan_state() {
    let _env_lock = swarm_plan_mermaid_env_lock();
    let _render_lock = scroll_render_test_lock();
    let _mode_guard = DiagramModeOverrideGuard::pinned();
    let mut app = create_test_app();
    app.diagram_mode = crate::config::DiagramDisplayMode::Pinned;
    app.diagram_pane_enabled = true;
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let mut remote = crate::tui::backend::RemoteConnection::dummy();
    remote.mark_history_loaded();

    // `/rewind N` needs rewindable stored messages.
    app.session.replace_messages(Vec::new());
    for idx in 1..=2 {
        let text = format!("msg-{idx}");
        app.add_provider_message(Message::user(&text));
        app.session.add_message(
            Role::User,
            vec![ContentBlock::Text {
                text,
                cache_control: None,
            }],
        );
    }

    let stale_hash = seed_rendered_plan_graph(&mut app, &mut remote);

    // Truncating rewind (commands.rs ~2004).
    assert!(super::commands::handle_session_command(&mut app, "/rewind 1"));
    assert_transcript_clear_leaks_diagram_and_plan_state(&mut app, stale_hash, "local /rewind N");

    // Rewind undo (commands.rs ~1933) restores the transcript from the
    // snapshot; the plan-graph display message was never stored, so it does
    // not come back, but the diagram and plan state stay stale.
    assert!(super::commands::handle_session_command(
        &mut app,
        "/rewind undo"
    ));
    assert_transcript_clear_leaks_diagram_and_plan_state(
        &mut app,
        stale_hash,
        "local /rewind undo",
    );

    crate::tui::mermaid::clear_active_diagrams();
}

/// Path 3: local session recovery (`recover_session_without_tools`,
/// conversation_state.rs ~809). It rebuilds the session into a fresh one and
/// clears the display transcript, but leaks the diagram and plan state the
/// same way.
#[test]
fn test_recover_session_without_tools_leaves_stale_active_diagram_and_swarm_plan_state() {
    let _env_lock = swarm_plan_mermaid_env_lock();
    let _render_lock = scroll_render_test_lock();
    let _mode_guard = DiagramModeOverrideGuard::pinned();
    let mut app = create_test_app();
    app.diagram_mode = crate::config::DiagramDisplayMode::Pinned;
    app.diagram_pane_enabled = true;
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let mut remote = crate::tui::backend::RemoteConnection::dummy();
    remote.mark_history_loaded();

    let stale_hash = seed_rendered_plan_graph(&mut app, &mut remote);

    app.recover_session_without_tools();

    assert_transcript_clear_leaks_diagram_and_plan_state(
        &mut app,
        stale_hash,
        "local Ctrl+R recovery (recover_session_without_tools)",
    );

    crate::tui::mermaid::clear_active_diagrams();
}

/// Path 4: remote `/clear` (remote/key_handling.rs ~1610). Unlike the
/// session-changing History event (which resets the swarm plan snapshot),
/// the remote clear command wipes provider/display messages only.
#[test]
fn test_remote_clear_command_leaves_stale_active_diagram_and_swarm_plan_state() {
    let _env_lock = swarm_plan_mermaid_env_lock();
    let _render_lock = scroll_render_test_lock();
    let _mode_guard = DiagramModeOverrideGuard::pinned();
    let mut app = create_test_app();
    app.diagram_mode = crate::config::DiagramDisplayMode::Pinned;
    app.diagram_pane_enabled = true;
    app.is_remote = true;
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let mut remote = crate::tui::backend::RemoteConnection::dummy();
    remote.mark_history_loaded();

    let stale_hash = seed_rendered_plan_graph(&mut app, &mut remote);

    app.input = "/clear".to_string();
    app.cursor_pos = app.input.len();
    rt.block_on(app.handle_remote_key(KeyCode::Enter, KeyModifiers::empty(), &mut remote))
        .expect("remote /clear should succeed");
    assert_eq!(
        crate::tui::TuiState::status_notice(&app).as_deref(),
        Some("Session cleared"),
        "remote /clear path executed"
    );

    assert_transcript_clear_leaks_diagram_and_plan_state(&mut app, stale_hash, "remote /clear");

    crate::tui::mermaid::clear_active_diagrams();
}

/// Path 5: disconnected Ctrl+L (remote.rs ~1670) clears the display
/// transcript and queued messages, again without touching the diagram
/// registry or the swarm plan snapshot. (The connected-remote Ctrl+L branch
/// at key_handling.rs ~611 and the local Ctrl+L branch at input.rs ~1922 are
/// deliberate no-ops, so the disconnected handler is the only Ctrl+L
/// transcript clear.)
#[test]
fn test_disconnected_ctrl_l_clear_leaves_stale_active_diagram_and_swarm_plan_state() {
    let _env_lock = swarm_plan_mermaid_env_lock();
    let _render_lock = scroll_render_test_lock();
    let _mode_guard = DiagramModeOverrideGuard::pinned();
    let mut app = create_test_app();
    app.diagram_mode = crate::config::DiagramDisplayMode::Pinned;
    app.diagram_pane_enabled = true;
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let mut remote = crate::tui::backend::RemoteConnection::dummy();
    remote.mark_history_loaded();

    let stale_hash = seed_rendered_plan_graph(&mut app, &mut remote);
    app.queued_messages.push("queued".to_string());

    super::remote::handle_disconnected_key(&mut app, KeyCode::Char('l'), KeyModifiers::CONTROL)
        .expect("disconnected Ctrl+L should succeed");
    assert!(
        app.queued_messages.is_empty(),
        "disconnected Ctrl+L clears queued messages (proves the clear branch ran)"
    );

    assert_transcript_clear_leaks_diagram_and_plan_state(
        &mut app,
        stale_hash,
        "disconnected Ctrl+L",
    );

    crate::tui::mermaid::clear_active_diagrams();
}
