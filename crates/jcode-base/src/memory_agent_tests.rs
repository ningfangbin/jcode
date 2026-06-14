use super::*;
use crate::memory::MemoryCategory;

#[test]
fn infer_candidate_tag_uses_repeated_non_stopword() {
    let tag =
        infer_candidate_tag("scheduler retries failed jobs and scheduler metrics update dashboard");
    assert_eq!(tag.as_deref(), Some("scheduler"));
}

#[test]
fn apply_cluster_assignment_links_members() {
    let mut graph = MemoryGraph::new();
    let mut a = MemoryEntry::new(MemoryCategory::Fact, "A");
    a.embedding = Some(vec![1.0, 0.0]);
    let id_a = graph.add_memory(a);

    let mut b = MemoryEntry::new(MemoryCategory::Fact, "B");
    b.embedding = Some(vec![0.0, 1.0]);
    let id_b = graph.add_memory(b);

    let stats = apply_cluster_assignment(
        &mut graph,
        "project",
        &[id_a.clone(), id_b.clone()],
        Utc::now(),
    );

    assert_eq!(stats.clusters_touched, 1);
    assert_eq!(stats.member_links, 2);
    assert_eq!(graph.clusters.len(), 1);

    let cluster_id = graph
        .clusters
        .keys()
        .next()
        .expect("cluster id")
        .to_string();
    assert!(
        graph
            .get_edges(&id_a)
            .iter()
            .any(|e| e.target == cluster_id && matches!(e.kind, EdgeKind::InCluster))
    );
    assert!(
        graph
            .get_edges(&id_b)
            .iter()
            .any(|e| e.target == cluster_id && matches!(e.kind, EdgeKind::InCluster))
    );
}

#[test]
fn apply_confidence_updates_batches_boost_and_decay() {
    let _guard = crate::storage::lock_test_env();
    let old = std::env::var("JCODE_HOME").ok();
    let dir = std::env::temp_dir().join(format!(
        "jcode-conf-test-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    crate::env::set_var("JCODE_HOME", &dir);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let manager = crate::memory::MemoryManager::new().with_project_dir("/tmp/jcode-conf-batch");

        let mut keep_entry = MemoryEntry::new(MemoryCategory::Fact, "verified memory")
            .with_embedding(vec![1.0, 0.0]);
        keep_entry.confidence = 0.5; // below cap so a boost is observable
        let keep = manager.remember_project(keep_entry).unwrap();
        let stale = manager
            .remember_project(
                MemoryEntry::new(MemoryCategory::Fact, "rejected memory")
                    .with_embedding(vec![0.0, 1.0]),
            )
            .unwrap();

        let conf_before = |id: &str| {
            manager
                .load_project_graph()
                .unwrap()
                .get_memory(id)
                .unwrap()
                .confidence
        };
        let keep_before = conf_before(&keep);
        let stale_before = conf_before(&stale);

        let (boosted, decayed) =
            apply_confidence_updates(&manager, &[keep.clone()], &[stale.clone()]);
        assert_eq!(boosted, 1, "one verified memory boosted");
        assert_eq!(decayed, 1, "one rejected memory decayed");

        let keep_after = conf_before(&keep);
        let stale_after = conf_before(&stale);
        assert!(keep_after > keep_before, "verified confidence should rise");
        assert!(stale_after < stale_before, "rejected confidence should fall");
    }));

    match old {
        Some(v) => crate::env::set_var("JCODE_HOME", v),
        None => crate::env::remove_var("JCODE_HOME"),
    }
    let _ = std::fs::remove_dir_all(&dir);
    if let Err(p) = result {
        std::panic::resume_unwind(p);
    }
}
