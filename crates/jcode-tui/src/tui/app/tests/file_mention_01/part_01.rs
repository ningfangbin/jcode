use crate::tui::ui::input_ui::{extract_at_query, replace_at_query};

#[test]
fn extract_query_parses_last_at_token() {
    assert_eq!(
        extract_at_query("fix @src/main.rs"),
        Some((4, "src/main.rs".into()))
    );
    assert_eq!(extract_at_query("hello world"), None);
    assert_eq!(extract_at_query("@ src/main.rs"), None);
}

#[test]
fn extract_query_multiple_ats() {
    assert_eq!(
        extract_at_query("fix @src/a.rs and @src/b.rs"),
        Some((18, "src/b.rs".into()))
    );
}

#[test]
fn replace_query_inserts_path() {
    let (r, c) = replace_at_query("fix @src/cli/st", "src/cli/startup.rs").unwrap();
    assert_eq!(r, "fix src/cli/startup.rs");
    assert_eq!(c, "fix src/cli/startup.rs".len());
}

#[test]
fn replace_query_preserves_trailing_text() {
    let (r, _) = replace_at_query("fix @src/cli/st please", "src/cli/startup.rs").unwrap();
    assert_eq!(r, "fix src/cli/startup.rs please");
}
