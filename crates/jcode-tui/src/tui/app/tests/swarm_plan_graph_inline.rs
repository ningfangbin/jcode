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
