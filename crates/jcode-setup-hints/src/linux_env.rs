//! Multi-compositor Linux launch-hotkey support.
//!
//! niri has its own dedicated module ([`crate::linux_niri`]) because it splices
//! KDL inside a `binds { }` block. Every other supported environment
//! (Hyprland, as shipped by omarchy, plus sway and i3) uses flat `#`-commented
//! config files where a bind is a single top-level line, so they share this
//! module:
//!
//! * [`detect_compositor_from`] decides which environment the session runs,
//!   from the standard env vars. Pure over an env lookup so it is unit-tested.
//! * The **pure** renderers ([`render_hyprland_block`], [`render_sway_block`])
//!   turn resolved launch hotkeys into the exact bind lines we manage. Instead
//!   of inlining shell one-liners (a quoting minefield across three different
//!   config grammars), each bind simply executes a launch script jcode writes
//!   to `~/.jcode/hotkey/`.
//! * [`splice_flat_managed_block`] replaces/creates the sentinel-delimited
//!   managed region at file scope, leaving every other line untouched.
//!
//! The managed region is delimited by `#` sentinel comments so re-installs are
//! idempotent and a user can hand-remove it cleanly:
//!
//! ```text
//! # >>> jcode launch hotkeys (managed) >>>
//! bind = SUPER, semicolon, exec, '/home/u/.jcode/hotkey/launch_jcode_0_cmd_semicolon.sh'
//! # <<< jcode launch hotkeys (managed) <<<
//! ```

use crate::keymap::KeyChord;

/// Opening sentinel for the managed region in `#`-commented configs.
pub(crate) const HASH_BLOCK_BEGIN: &str = "# >>> jcode launch hotkeys (managed) >>>";
/// Closing sentinel for the managed region in `#`-commented configs.
pub(crate) const HASH_BLOCK_END: &str = "# <<< jcode launch hotkeys (managed) <<<";

/// A Linux desktop environment / compositor jcode can install launch hotkeys
/// into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LinuxCompositor {
    Niri,
    /// Hyprland (what omarchy ships).
    Hyprland,
    Sway,
    I3,
}

impl LinuxCompositor {
    pub(crate) fn name(&self) -> &'static str {
        match self {
            LinuxCompositor::Niri => "niri",
            LinuxCompositor::Hyprland => "Hyprland",
            LinuxCompositor::Sway => "sway",
            LinuxCompositor::I3 => "i3",
        }
    }
}

/// Detect the running compositor from an env-var lookup. Pure so the detection
/// matrix is unit-testable; production passes `std::env::var`.
///
/// Socket/instance env vars are checked first (they are authoritative for the
/// *current* session), then the XDG desktop names, so a stale
/// `XDG_CURRENT_DESKTOP` inherited across a compositor switch loses to a live
/// socket.
pub(crate) fn detect_compositor_from(
    get: &dyn Fn(&str) -> Option<String>,
) -> Option<LinuxCompositor> {
    let desktop_is = |name: &str| -> bool {
        let matches_var = |var: &str| {
            get(var)
                .map(|v| v.split(':').any(|d| d.eq_ignore_ascii_case(name)))
                .unwrap_or(false)
        };
        matches_var("XDG_CURRENT_DESKTOP") || matches_var("XDG_SESSION_DESKTOP")
    };

    if get("NIRI_SOCKET").is_some() || desktop_is("niri") {
        return Some(LinuxCompositor::Niri);
    }
    if get("HYPRLAND_INSTANCE_SIGNATURE").is_some() || desktop_is("hyprland") {
        return Some(LinuxCompositor::Hyprland);
    }
    if get("SWAYSOCK").is_some() || desktop_is("sway") {
        return Some(LinuxCompositor::Sway);
    }
    if get("I3SOCK").is_some() || desktop_is("i3") {
        return Some(LinuxCompositor::I3);
    }
    None
}

/// One launch hotkey resolved down to "this chord runs this script".
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ScriptBind {
    pub chord: KeyChord,
    /// Absolute path of the executable launch script.
    pub script: String,
    /// Short human label, e.g. the repo's directory name.
    pub label: String,
    pub self_dev: bool,
}

/// Translate a canonical jcode key token into an XKB keysym name (the
/// vocabulary Hyprland, sway, and i3 all accept). Returns `None` for tokens
/// with no stable spelling.
pub(crate) fn xkb_key_name(key: &str) -> Option<String> {
    let named = match key {
        ";" => "semicolon",
        "'" => "apostrophe",
        "[" => "bracketleft",
        "]" => "bracketright",
        "\\" => "backslash",
        "/" => "slash",
        "," => "comma",
        "." => "period",
        "-" => "minus",
        "=" => "equal",
        "`" => "grave",
        "left" => "Left",
        "right" => "Right",
        "up" => "Up",
        "down" => "Down",
        "pageup" => "Prior",
        "pagedown" => "Next",
        "home" => "Home",
        "end" => "End",
        "insert" => "Insert",
        "delete" => "Delete",
        "backspace" => "BackSpace",
        "enter" => "Return",
        "esc" => "Escape",
        "tab" => "Tab",
        "space" => "space",
        other => {
            if other.len() == 1 && other.chars().all(|c| c.is_ascii_alphanumeric()) {
                return Some(other.to_string());
            }
            if let Some(rest) = other.strip_prefix('f')
                && !rest.is_empty()
                && rest.chars().all(|c| c.is_ascii_digit())
            {
                return Some(format!("F{rest}"));
            }
            return None;
        }
    };
    Some(named.to_string())
}

/// POSIX-shell single-quote escaping for a path embedded in a bind line.
pub(crate) fn sh_single_quote(input: &str) -> String {
    format!("'{}'", input.replace('\'', r#"'\''"#))
}

/// Render one Hyprland `bind` line, or `None` if the chord cannot be
/// expressed. jcode's `cmd` modifier maps to `SUPER`.
///
/// Hyprland's `exec` dispatcher passes the remainder of the line to a shell,
/// so the script path is single-quoted to survive spaces.
pub(crate) fn render_hyprland_bind_line(bind: &ScriptBind) -> Option<String> {
    let key = xkb_key_name(&bind.chord.key)?;
    let mods = hyprland_mods(&bind.chord);
    Some(format!(
        "bind = {mods}, {key}, exec, {script}",
        script = sh_single_quote(&bind.script),
    ))
}

/// Hyprland modifier list (space-separated, e.g. `SUPER SHIFT`). jcode `cmd`
/// maps to `SUPER`.
fn hyprland_mods(chord: &KeyChord) -> String {
    let mut parts: Vec<&str> = Vec::new();
    if chord.cmd {
        parts.push("SUPER");
    }
    if chord.ctrl {
        parts.push("CTRL");
    }
    if chord.alt {
        parts.push("ALT");
    }
    if chord.shift {
        parts.push("SHIFT");
    }
    parts.join(" ")
}

/// Render one sway/i3 `bindsym` line, or `None` if the chord cannot be
/// expressed. jcode's `cmd` maps to `Mod4` (super) and `alt` to `Mod1`.
///
/// i3 (and sway, which accepts the same grammar) treats `,` and `;` as command
/// separators inside a bind, so the exec payload is a quoted script path with
/// no shell metacharacters. `--no-startup-id` suppresses i3's startup-
/// notification cursor; sway accepts and ignores it.
pub(crate) fn render_sway_bind_line(bind: &ScriptBind) -> Option<String> {
    let key = xkb_key_name(&bind.chord.key)?;
    let mut parts: Vec<String> = Vec::new();
    if bind.chord.cmd {
        parts.push("Mod4".to_string());
    }
    if bind.chord.ctrl {
        parts.push("Ctrl".to_string());
    }
    if bind.chord.alt {
        parts.push("Mod1".to_string());
    }
    if bind.chord.shift {
        parts.push("Shift".to_string());
    }
    parts.push(key);
    let combo = parts.join("+");
    let script = bind.script.replace('\\', "\\\\").replace('"', "\\\"");
    Some(format!(
        "bindsym {combo} exec --no-startup-id \"{script}\""
    ))
}

/// Render the full managed block for a flat `#`-commented config. `render_line`
/// is the per-compositor bind renderer. Each bind is preceded by a label
/// comment on its own line (i3 only supports whole-line comments, so labels
/// are never appended to the bind line itself). Hotkeys that cannot be
/// expressed are skipped; returns `None` when nothing could be rendered.
pub(crate) fn render_flat_block(
    binds: &[ScriptBind],
    render_line: impl Fn(&ScriptBind) -> Option<String>,
) -> Option<String> {
    let mut lines: Vec<String> = Vec::new();
    for bind in binds {
        if let Some(line) = render_line(bind) {
            lines.push(format!("# jcode: {label}", label = bind.label));
            lines.push(line);
        }
    }
    if lines.is_empty() {
        return None;
    }
    let mut out = String::new();
    out.push_str(HASH_BLOCK_BEGIN);
    out.push('\n');
    for line in &lines {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(HASH_BLOCK_END);
    Some(out)
}

/// Render the managed Hyprland block.
pub(crate) fn render_hyprland_block(binds: &[ScriptBind]) -> Option<String> {
    render_flat_block(binds, render_hyprland_bind_line)
}

/// Render the managed sway/i3 block.
pub(crate) fn render_sway_block(binds: &[ScriptBind]) -> Option<String> {
    render_flat_block(binds, render_sway_bind_line)
}

/// Result of splicing the managed block into a config: the new text plus
/// whether anything actually changed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FlatSpliceResult {
    pub text: String,
    pub changed: bool,
}

/// Splice `block` (a fully-rendered managed region, no trailing newline) into a
/// flat config file: replace an existing sentinel-delimited region in place, or
/// append the block at the end of the file.
///
/// Returns `changed = false` (and the original text) when the existing managed
/// region already equals `block`, so callers can skip a no-op write.
pub(crate) fn splice_flat_managed_block(config: &str, block: &str) -> FlatSpliceResult {
    if let Some((begin, end)) = find_flat_managed_region(config) {
        let before = &config[..begin];
        let after = &config[end..];
        let new_text = format!("{before}{block}\n{after}");
        let changed = new_text != config;
        return FlatSpliceResult {
            text: new_text,
            changed,
        };
    }

    let mut new_text = config.to_string();
    if !new_text.is_empty() && !new_text.ends_with('\n') {
        new_text.push('\n');
    }
    if !new_text.is_empty() {
        new_text.push('\n');
    }
    new_text.push_str(block);
    new_text.push('\n');
    FlatSpliceResult {
        text: new_text,
        changed: true,
    }
}

/// Byte range of an existing managed region: `(start_of_BEGIN_line,
/// end_of_END_line_including_newline)`. Returns `None` when either sentinel is
/// missing so a half-deleted region is never mangled.
fn find_flat_managed_region(config: &str) -> Option<(usize, usize)> {
    let begin_pos = config.find(HASH_BLOCK_BEGIN)?;
    let line_start = config[..begin_pos].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let end_pos = config[begin_pos..].find(HASH_BLOCK_END)? + begin_pos;
    let line_end = match config[end_pos..].find('\n') {
        Some(nl) => end_pos + nl + 1,
        None => config.len(),
    };
    Some((line_start, line_end))
}

/// Build the shell command a launch script uses to open jcode in the user's
/// terminal, as a `argv`-quoted string (e.g. `'kitty' '/bin/jcode' 'self-dev'`).
/// Terminals differ in how they accept a command to run.
pub(crate) fn terminal_exec_command(terminal: &str, exe_path: &str, self_dev: bool) -> String {
    let base = std::path::Path::new(terminal)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| terminal.to_string());
    let mut argv: Vec<String> = match base.as_str() {
        "wezterm" => vec![terminal.to_string(), "start".to_string(), "--".to_string()],
        "alacritty" | "ghostty" | "konsole" | "xterm" => {
            vec![terminal.to_string(), "-e".to_string()]
        }
        // kitty, foot, and most others accept the command as direct argv.
        _ => vec![terminal.to_string()],
    };
    argv.push(exe_path.to_string());
    if self_dev {
        argv.push("self-dev".to_string());
    }
    argv.iter()
        .map(|a| sh_single_quote(a))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chord(s: &str) -> KeyChord {
        KeyChord::parse(s).unwrap()
    }

    fn bind(chord_str: &str, script: &str, label: &str, self_dev: bool) -> ScriptBind {
        ScriptBind {
            chord: chord(chord_str),
            script: script.to_string(),
            label: label.to_string(),
            self_dev,
        }
    }

    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |key: &str| {
            pairs
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| v.to_string())
        }
    }

    #[test]
    fn detects_compositors_from_sockets_and_desktop_names() {
        let cases: Vec<(Vec<(&str, &str)>, Option<LinuxCompositor>)> = vec![
            (vec![("NIRI_SOCKET", "/run/niri.sock")], Some(LinuxCompositor::Niri)),
            (
                vec![("HYPRLAND_INSTANCE_SIGNATURE", "abc123")],
                Some(LinuxCompositor::Hyprland),
            ),
            // omarchy sets XDG_CURRENT_DESKTOP=Hyprland.
            (
                vec![("XDG_CURRENT_DESKTOP", "Hyprland")],
                Some(LinuxCompositor::Hyprland),
            ),
            (vec![("SWAYSOCK", "/run/sway.sock")], Some(LinuxCompositor::Sway)),
            (vec![("XDG_CURRENT_DESKTOP", "sway")], Some(LinuxCompositor::Sway)),
            (vec![("I3SOCK", "/run/i3.sock")], Some(LinuxCompositor::I3)),
            (vec![("XDG_SESSION_DESKTOP", "i3")], Some(LinuxCompositor::I3)),
            (vec![("XDG_CURRENT_DESKTOP", "GNOME")], None),
            (vec![], None),
        ];
        for (pairs, expected) in cases {
            let got = detect_compositor_from(&env(&pairs));
            assert_eq!(got, expected, "env {pairs:?}");
        }
    }

    #[test]
    fn live_socket_beats_stale_desktop_name() {
        // A user who switched from GNOME to Hyprland keeps a live Hyprland
        // instance signature; the stale desktop var must not win.
        let pairs = vec![
            ("HYPRLAND_INSTANCE_SIGNATURE", "sig"),
            ("XDG_CURRENT_DESKTOP", "GNOME"),
        ];
        assert_eq!(
            detect_compositor_from(&env(&pairs)),
            Some(LinuxCompositor::Hyprland)
        );
    }

    #[test]
    fn hyprland_bind_line_maps_cmd_to_super() {
        let line = render_hyprland_bind_line(&bind(
            "cmd+;",
            "/home/u/.jcode/hotkey/launch_jcode_0_cmd_semicolon.sh",
            "jcode",
            false,
        ))
        .unwrap();
        assert_eq!(
            line,
            "bind = SUPER, semicolon, exec, '/home/u/.jcode/hotkey/launch_jcode_0_cmd_semicolon.sh'"
        );

        let shifted = render_hyprland_bind_line(&bind("cmd+shift+'", "/s.sh", "x", true)).unwrap();
        assert!(shifted.starts_with("bind = SUPER SHIFT, apostrophe, exec, "));
    }

    #[test]
    fn sway_bind_line_maps_cmd_to_mod4_and_alt_to_mod1() {
        let line = render_sway_bind_line(&bind("cmd+;", "/s.sh", "jcode", false)).unwrap();
        assert_eq!(line, "bindsym Mod4+semicolon exec --no-startup-id \"/s.sh\"");

        let alt = render_sway_bind_line(&bind("alt+shift+[", "/s.sh", "x", false)).unwrap();
        assert_eq!(
            alt,
            "bindsym Mod1+Shift+bracketleft exec --no-startup-id \"/s.sh\""
        );
    }

    #[test]
    fn unmappable_keys_are_skipped_not_fatal() {
        assert!(xkb_key_name("scrolllock").is_none());
        let block = render_hyprland_block(&[
            bind("cmd+scrolllock", "/a.sh", "a", false),
            bind("cmd+]", "/b.sh", "b", false),
        ])
        .unwrap();
        assert_eq!(block.matches("bind = ").count(), 1);
        assert!(block.contains("bracketright"));
    }

    #[test]
    fn blocks_wrap_sentinels_and_carry_labels() {
        let block = render_hyprland_block(&[
            bind("cmd+;", "/a.sh", "jcode", false),
            bind("cmd+'", "/b.sh", "home", false),
        ])
        .unwrap();
        assert!(block.starts_with(HASH_BLOCK_BEGIN));
        assert!(block.ends_with(HASH_BLOCK_END));
        assert!(block.contains("# jcode: jcode"));
        assert!(block.contains("# jcode: home"));

        let sway = render_sway_block(&[bind("cmd+;", "/a.sh", "jcode", false)]).unwrap();
        assert!(sway.starts_with(HASH_BLOCK_BEGIN));
        assert!(sway.contains("bindsym Mod4+semicolon"));
    }

    #[test]
    fn render_returns_none_when_nothing_renderable() {
        assert!(render_hyprland_block(&[]).is_none());
        assert!(
            render_hyprland_block(&[bind("cmd+scrolllock", "/a.sh", "a", false)]).is_none()
        );
    }

    #[test]
    fn splice_appends_then_replaces_idempotently() {
        let cfg = "# my config\nbind = SUPER, Q, killactive\n";
        let block_v1 = render_hyprland_block(&[bind("cmd+;", "/a.sh", "a", false)]).unwrap();

        let first = splice_flat_managed_block(cfg, &block_v1);
        assert!(first.changed);
        assert!(first.text.contains(HASH_BLOCK_BEGIN));
        assert!(first.text.contains("bind = SUPER, Q, killactive"));

        // Re-splicing the same block is a no-op.
        let again = splice_flat_managed_block(&first.text, &block_v1);
        assert!(!again.changed);
        assert_eq!(again.text, first.text);

        // A new block replaces the old region in place, exactly once.
        let block_v2 = render_hyprland_block(&[bind("cmd+[", "/b.sh", "b", false)]).unwrap();
        let replaced = splice_flat_managed_block(&first.text, &block_v2);
        assert!(replaced.changed);
        assert_eq!(replaced.text.matches(HASH_BLOCK_BEGIN).count(), 1);
        assert!(replaced.text.contains("/b.sh"));
        assert!(!replaced.text.contains("/a.sh"));
        assert!(replaced.text.contains("bind = SUPER, Q, killactive"));
    }

    #[test]
    fn splice_handles_file_without_trailing_newline() {
        let cfg = "bind = SUPER, Q, killactive";
        let block = render_hyprland_block(&[bind("cmd+;", "/a.sh", "a", false)]).unwrap();
        let res = splice_flat_managed_block(cfg, &block);
        assert!(res.changed);
        assert!(res.text.contains("killactive\n"));
        assert!(res.text.ends_with(&format!("{HASH_BLOCK_END}\n")));
    }

    #[test]
    fn terminal_exec_command_varies_by_terminal() {
        assert_eq!(
            terminal_exec_command("kitty", "/bin/jcode", false),
            "'kitty' '/bin/jcode'"
        );
        assert_eq!(
            terminal_exec_command("alacritty", "/bin/jcode", true),
            "'alacritty' '-e' '/bin/jcode' 'self-dev'"
        );
        assert_eq!(
            terminal_exec_command("wezterm", "/bin/jcode", false),
            "'wezterm' 'start' '--' '/bin/jcode'"
        );
        assert_eq!(
            terminal_exec_command("foot", "/bin/jcode", false),
            "'foot' '/bin/jcode'"
        );
        // Full paths keep the path but dispatch on the basename.
        assert_eq!(
            terminal_exec_command("/usr/bin/ghostty", "/bin/jcode", false),
            "'/usr/bin/ghostty' '-e' '/bin/jcode'"
        );
    }
}
