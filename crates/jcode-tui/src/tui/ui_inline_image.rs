//! Inline image transcript section.
//!
//! Images attached to the conversation (pasted screenshots, `read` of an image
//! file, generated images) render directly in the chat flow, sized to fit the
//! chat width with a capped height. This replaces the old "pinned image side
//! panel" surface.
//!
//! Design goals:
//! * **Lazy.** Prepare only needs each image's `(id, width, height)`, obtained
//!   from a cheap header parse (no full decode, no disk write, no retained
//!   bytes). The full decode + terminal transmit happens at draw time, and only
//!   for images currently on screen.
//! * **Single source of pixels.** The base64 payloads stay in their existing
//!   owner (`App::side_pane_images()`); this section keeps only ids and a small
//!   ingest-time payload registry so the draw step can materialize on demand.
//! * **Correct fit.** Images scale to fit width (preserving aspect) and cap at a
//!   fraction of the viewport so a tall screenshot never buries the transcript.

use crate::tui::mermaid;
use jcode_tui_messages::{ImageRegion, ImageRegionRender, PreparedMessages};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use std::collections::{HashMap, HashSet};
use std::sync::{LazyLock, Mutex, OnceLock, mpsc};

#[inline]
fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    let divisor = divisor.max(1);
    value.div_ceil(divisor)
}

/// One image to render inline, resolved from a `RenderedImage`.
#[derive(Clone)]
pub(crate) struct InlineImageItem {
    pub id: u64,
    pub width: u32,
    pub height: u32,
    pub label: String,
}

/// Default font cell size when the terminal has not reported one yet.
const DEFAULT_CELL: (u16, u16) = (8, 16);
/// Cap an inline image at this fraction of the chat viewport height so a tall
/// image cannot push the rest of the transcript off-screen.
const MAX_VIEWPORT_FRACTION_PERCENT: u16 = 55;
/// Never shrink an inline image below this many rows.
const MIN_IMAGE_ROWS: u16 = 3;
/// Fixed row cap for images anchored inside the transcript body. The body is
/// prepared and cached independently of the viewport height, so anchored
/// placeholder geometry must not depend on it; a fixed cap keeps tall
/// screenshots from dominating the flow while staying resize-stable.
const ANCHORED_MAX_ROWS: u16 = 16;

/// Discrete per-image expand levels. Clicking the `expand` badge cycles an
/// image through these caps. The caps are *fixed row counts* (not a fraction of
/// the viewport) on purpose: anchored placeholder geometry feeds the body cache
/// which is keyed by width only, so the expand level must stay viewport
/// independent or it would break that invariant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ImageExpandLevel {
    /// Default fit-to-flow size (`ANCHORED_MAX_ROWS`).
    #[default]
    Fit,
    /// Roughly 2.5x taller, for a closer look without leaving the transcript.
    Large,
    /// Maximum inline size before the user should use the side panel.
    Huge,
}

impl ImageExpandLevel {
    /// Next level in the click cycle (Fit -> Large -> Huge -> Fit).
    pub(crate) fn next(self) -> Self {
        match self {
            ImageExpandLevel::Fit => ImageExpandLevel::Large,
            ImageExpandLevel::Large => ImageExpandLevel::Huge,
            ImageExpandLevel::Huge => ImageExpandLevel::Fit,
        }
    }

    /// Anchored row cap for this level. Stays viewport independent so the
    /// width-keyed body cache remains valid across resizes.
    fn anchored_cap_rows(self) -> u16 {
        match self {
            ImageExpandLevel::Fit => ANCHORED_MAX_ROWS,
            ImageExpandLevel::Large => 40,
            ImageExpandLevel::Huge => 80,
        }
    }

    /// 0-based filled-dot count for the click-progress badge (Fit shows none).
    fn filled_dots(self) -> usize {
        match self {
            ImageExpandLevel::Fit => 0,
            ImageExpandLevel::Large => 1,
            ImageExpandLevel::Huge => 2,
        }
    }
}

/// Resolve the expand level for an image id. Implemented by `App` so the lookup
/// stays close to the persisted/live state, while this module owns the geometry.
pub(crate) trait ImageExpandLevels {
    fn expand_level(&self, id: u64) -> ImageExpandLevel;
}

/// A levels source that reports every image as `Fit`. Used by tests that
/// exercise section/line building without an `App` to resolve expand state.
#[cfg(test)]
pub(crate) struct AllFit;
#[cfg(test)]
impl ImageExpandLevels for AllFit {
    fn expand_level(&self, _id: u64) -> ImageExpandLevel {
        ImageExpandLevel::Fit
    }
}

/// Adapter so prepare code can resolve per-image expand levels straight from the
/// app state without copying the whole map into this module.
pub(crate) struct AppExpandLevels<'a>(pub &'a dyn crate::tui::TuiState);
impl ImageExpandLevels for AppExpandLevels<'_> {
    fn expand_level(&self, id: u64) -> ImageExpandLevel {
        self.0.image_expand_level(id)
    }
}

/// Ingest-time registry mapping image id -> (media_type, base64) so the draw
/// step can materialize bytes without threading payloads through the cached
/// prepared-frame model. Bounded; entries are cheap (two `String`s + id).
static PAYLOAD_REGISTRY: LazyLock<Mutex<PayloadRegistry>> =
    LazyLock::new(|| Mutex::new(PayloadRegistry::new()));

const PAYLOAD_REGISTRY_MAX: usize = 512;

struct PayloadRegistry {
    map: std::collections::HashMap<u64, (String, String)>,
    order: std::collections::VecDeque<u64>,
}

impl PayloadRegistry {
    fn new() -> Self {
        Self {
            map: std::collections::HashMap::new(),
            order: std::collections::VecDeque::new(),
        }
    }

    fn insert(&mut self, id: u64, media_type: &str, data_b64: &str) {
        if self.map.contains_key(&id) {
            return;
        }
        self.map
            .insert(id, (media_type.to_string(), data_b64.to_string()));
        self.order.push_back(id);
        while self.order.len() > PAYLOAD_REGISTRY_MAX {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
    }

    fn get(&self, id: u64) -> Option<(String, String)> {
        self.map.get(&id).cloned()
    }
}

/// Record an image payload so [`materialize_visible`] can decode it on demand.
pub(crate) fn register_payload(id: u64, media_type: &str, data_b64: &str) {
    if let Ok(mut reg) = PAYLOAD_REGISTRY.lock() {
        reg.insert(id, media_type, data_b64);
    }
}

/// Ensure the image with `id` is materialized (decoded + cached) so it can be
/// drawn. Returns true on success.
///
/// Steady-state frames hit a cheap in-memory presence probe (no payload clone,
/// no payload hash); only the first visible frame for an image pays the decode
/// + cache cost.
pub(crate) fn materialize_visible(id: u64) -> bool {
    if mermaid::inline_image_is_materialized(id) {
        return true;
    }
    if let Some((media_type, data_b64)) = PAYLOAD_REGISTRY.lock().ok().and_then(|reg| reg.get(id)) {
        return mermaid::materialize_inline_image_by_id(id, &media_type, &data_b64).is_some();
    }
    false
}

/// One pending prewarm request: build everything needed to draw image `id`
/// at the given placeholder geometry (decode payload, write cache file, scale
/// to the target box, escape-encode for Kitty).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PrewarmRequest {
    id: u64,
    target_cols: u16,
    target_rows: u16,
}

static PREWARM_TX: OnceLock<mpsc::Sender<PrewarmRequest>> = OnceLock::new();
/// Requests queued or in flight, so a 60fps scroll doesn't enqueue the same
/// image dozens of times before the worker finishes it.
static PREWARM_INFLIGHT: LazyLock<Mutex<HashSet<PrewarmRequest>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

fn prewarm_sender() -> &'static mpsc::Sender<PrewarmRequest> {
    PREWARM_TX.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<PrewarmRequest>();
        if let Err(err) = std::thread::Builder::new()
            .name("jcode-inline-image-prewarm".to_string())
            .spawn(move || prewarm_worker(rx))
        {
            crate::logging::warn(&format!(
                "Failed to spawn inline-image prewarm worker; first view will decode on the UI thread: {}",
                err
            ));
        }
        tx
    })
}

fn prewarm_worker(rx: mpsc::Receiver<PrewarmRequest>) {
    for req in rx {
        materialize_visible(req.id);
        mermaid::prewarm_inline_fit_state(req.id, req.target_cols, req.target_rows, true);
        if let Ok(mut inflight) = PREWARM_INFLIGHT.lock() {
            inflight.remove(&req);
        }
        // Nudge the UI exactly like a finished deferred Mermaid render so the
        // placeholder fills in on the next frame without user input. The
        // prepared placeholder geometry is unchanged, so no prepare-cache
        // invalidation is needed - just a repaint.
        crate::bus::Bus::global().publish(crate::bus::BusEvent::MermaidRenderCompleted);
    }
}

/// Make sure image `id` can be drawn cheaply this frame.
///
/// Returns true when the draw path can run now without heavy work (image
/// decoded and, on Kitty, scale+transmit state matches the placeholder
/// geometry). Returns false after scheduling background preparation; the
/// caller should skip drawing this frame and rely on the completion nudge to
/// repaint.
pub(crate) fn ensure_drawable(id: u64, target_cols: u16, target_rows: u16) -> bool {
    let materialized = mermaid::inline_image_is_materialized(id);
    let readiness = if materialized {
        mermaid::inline_fit_readiness(id, target_cols, target_rows, true)
    } else {
        // Not decoded yet. On any protocol the first draw would block on a
        // full decode, so prewarm regardless of protocol support.
        mermaid::InlineFitReadiness::NeedsPrewarm
    };

    match readiness {
        mermaid::InlineFitReadiness::Ready => true,
        mermaid::InlineFitReadiness::Unsupported => {
            // Non-Kitty fallback renderers manage their own protocol state;
            // just make sure the bytes are decoded, off-thread if possible.
            if materialized {
                true
            } else {
                schedule_prewarm(id, target_cols, target_rows);
                false
            }
        }
        mermaid::InlineFitReadiness::NeedsPrewarm => {
            schedule_prewarm(id, target_cols, target_rows);
            false
        }
    }
}

fn schedule_prewarm(id: u64, target_cols: u16, target_rows: u16) {
    let req = PrewarmRequest {
        id,
        target_cols,
        target_rows,
    };
    if let Ok(mut inflight) = PREWARM_INFLIGHT.lock()
        && !inflight.insert(req)
    {
        return;
    }
    if prewarm_sender().send(req).is_err() {
        // Worker unavailable: fall back to synchronous work on next frame.
        if let Ok(mut inflight) = PREWARM_INFLIGHT.lock() {
            inflight.remove(&req);
        }
        materialize_visible(id);
    }
}

/// Warm an inline image that is *not* on screen yet so it is ready to draw the
/// instant it scrolls into view. Unlike [`ensure_drawable`], this never blocks
/// and never draws: if the image still needs decode/scale/transmit work it is
/// scheduled on the background prewarm worker (deduped against in-flight and
/// already-warm state), otherwise it is a cheap no-op.
///
/// Callers pass the same `(target_cols, target_rows)` placeholder geometry the
/// draw path will use, so the prewarmed Kitty fit-state matches exactly and the
/// first on-screen frame hits the `Ready` fast path with no rescale.
pub(crate) fn prefetch(id: u64, target_cols: u16, target_rows: u16) {
    let readiness = if mermaid::inline_image_is_materialized(id) {
        mermaid::inline_fit_readiness(id, target_cols, target_rows, true)
    } else {
        mermaid::InlineFitReadiness::NeedsPrewarm
    };
    match readiness {
        // Already drawable, or a protocol that builds its state synchronously
        // at draw time (nothing useful to prewarm ahead).
        mermaid::InlineFitReadiness::Ready | mermaid::InlineFitReadiness::Unsupported => {}
        mermaid::InlineFitReadiness::NeedsPrewarm => {
            schedule_prewarm(id, target_cols, target_rows);
        }
    }
}

fn resolve_item(image: &crate::session::RenderedImage) -> Option<InlineImageItem> {
    let (id, width, height) = mermaid::inline_image_dims(&image.media_type, &image.data)?;
    register_payload(id, &image.media_type, &image.data);
    let label = image
        .label
        .clone()
        .unwrap_or_else(|| image.media_type.clone());
    Some(InlineImageItem {
        id,
        width,
        height,
        label,
    })
}

/// Inline images split by their transcript anchor so the body renderer can
/// place each one at the message that produced it.
#[derive(Default)]
pub(crate) struct AnchoredInlineImages {
    /// Images anchored to a tool result, keyed by tool call id.
    pub by_tool: HashMap<String, Vec<InlineImageItem>>,
    /// Images anchored to the nth (0-based) rendered user prompt.
    pub by_prompt: HashMap<usize, Vec<InlineImageItem>>,
    /// Images with no usable anchor; rendered at the end of the transcript.
    pub unanchored: Vec<InlineImageItem>,
}

impl AnchoredInlineImages {
    #[cfg(test)]
    pub(crate) fn has_anchored(&self) -> bool {
        !self.by_tool.is_empty() || !self.by_prompt.is_empty()
    }

    /// Items that will NOT be placed inside the transcript body: unanchored
    /// images plus anchored images whose anchor target does not exist among
    /// `messages` (e.g. live images for a tool whose transcript entry was
    /// replaced). These fall back to the bottom inline-images section so no
    /// image ever silently disappears.
    pub(crate) fn unplaced_items(
        &self,
        messages: &[jcode_tui_messages::DisplayMessage],
    ) -> Vec<InlineImageItem> {
        let mut items: Vec<InlineImageItem> = self.unanchored.clone();
        if self.by_tool.is_empty() && self.by_prompt.is_empty() {
            return items;
        }

        let mut tool_ids: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut prompt_count = 0usize;
        for msg in messages {
            use crate::tui::DisplayMessageRoleExt as _;
            match msg.effective_role() {
                "tool" => {
                    if let Some(tool) = &msg.tool_data {
                        tool_ids.insert(tool.id.as_str());
                    }
                }
                "user" => {
                    if !crate::session::is_attached_image_label_text(&msg.content) {
                        prompt_count += 1;
                    }
                }
                _ => {}
            }
        }

        for (id, bucket) in &self.by_tool {
            if !tool_ids.contains(id.as_str()) {
                items.extend(bucket.iter().cloned());
            }
        }
        for (ordinal, bucket) in &self.by_prompt {
            if *ordinal >= prompt_count {
                items.extend(bucket.iter().cloned());
            }
        }
        items
    }
}

/// Resolve rendered images into anchored buckets (tool call / user prompt /
/// unanchored). Same lazy header-only cost profile as [`resolve_item`].
pub(crate) fn resolve_anchored_items(
    images: &[crate::session::RenderedImage],
) -> AnchoredInlineImages {
    let mut result = AnchoredInlineImages::default();
    for image in images {
        let Some(item) = resolve_item(image) else {
            continue;
        };
        match &image.anchor {
            Some(crate::session::RenderedImageAnchor::ToolCall { id }) => {
                result.by_tool.entry(id.clone()).or_default().push(item);
            }
            Some(crate::session::RenderedImageAnchor::UserPrompt { ordinal }) => {
                result.by_prompt.entry(*ordinal).or_default().push(item);
            }
            None => result.unanchored.push(item),
        }
    }
    result
}

/// One-slot cache for [`resolve_anchored_items`], keyed by the image-set
/// signature. Resolving hashes every image payload (for ids), so body
/// preparation must not redo it per rebuild; the signature is already cached
/// per transcript version on the app side.
type AnchoredCache = Mutex<Option<((usize, u64), std::sync::Arc<AnchoredInlineImages>)>>;
static ANCHORED_CACHE: LazyLock<AnchoredCache> = LazyLock::new(|| Mutex::new(None));

/// Resolve the app's images into anchored buckets, cached by the image-set
/// signature. Returns an empty result without touching payloads when the app
/// has no images or inline images are hidden.
pub(crate) fn resolve_anchored_items_cached(
    app: &dyn crate::tui::TuiState,
) -> std::sync::Arc<AnchoredInlineImages> {
    if !app.pin_images() {
        return std::sync::Arc::new(AnchoredInlineImages::default());
    }
    let signature = app.side_pane_images_signature();
    if signature.0 == 0 {
        return std::sync::Arc::new(AnchoredInlineImages::default());
    }
    if let Ok(cache) = ANCHORED_CACHE.lock()
        && let Some((cached_sig, cached)) = cache.as_ref()
        && *cached_sig == signature
    {
        return cached.clone();
    }
    let resolved = std::sync::Arc::new(resolve_anchored_items(&app.side_pane_images()));
    if let Ok(mut cache) = ANCHORED_CACHE.lock() {
        *cache = Some((signature, resolved.clone()));
    }
    resolved
}

/// Compute how many `(rows, cols)` an inline image occupies at `chat_width`,
/// capped at `cap_rows`. `cols` includes the 2-cell left border, matching what
/// the draw step actually paints, so layout (e.g. info widget placement) can
/// know the real horizontal extent.
fn fit_geometry_with_cap(width: u32, height: u32, chat_width: u16, cap_rows: u16) -> (u16, u16) {
    if width == 0 || height == 0 {
        return (MIN_IMAGE_ROWS, chat_width.min(2));
    }
    let (cell_w, cell_h) = mermaid::get_font_size().unwrap_or(DEFAULT_CELL);
    let cell_w = cell_w.max(1) as u32;
    let cell_h = cell_h.max(1) as u32;

    // Available width in pixels (border bar + padding take 2 cells, matching
    // the renderer's BORDER_WIDTH).
    let avail_cells = chat_width.saturating_sub(2).max(1) as u32;
    let avail_px = avail_cells * cell_w;

    let cap_rows = (cap_rows as u32).max(MIN_IMAGE_ROWS as u32);
    let cap_px = cap_rows * cell_h;

    // Scale to fit *both* the width and the row cap, preserving aspect ratio,
    // exactly like the draw-time fit does. This keeps the placeholder geometry
    // and the rendered pixels in lockstep so borders/labels hug the image.
    let scale_num_w = avail_px.min(width);
    let scaled_h_by_w = height.saturating_mul(scale_num_w) / width.max(1);
    let (final_w_px, final_h_px) = if scaled_h_by_w <= cap_px {
        (scale_num_w, scaled_h_by_w)
    } else {
        // Height-bound: shrink further so the height fits the cap.
        let w = width.saturating_mul(cap_px) / height.max(1);
        (w.min(avail_px).max(1), cap_px)
    };

    let rows = div_ceil_u32(final_h_px.max(1), cell_h).max(MIN_IMAGE_ROWS as u32) as u16;
    let cols = (div_ceil_u32(final_w_px.max(1), cell_w) as u16)
        .saturating_add(2)
        .min(chat_width);
    (
        rows.min(cap_rows.min(u16::MAX as u32) as u16)
            .max(MIN_IMAGE_ROWS),
        cols,
    )
}

/// Compute `(rows, cols)` for an inline image at `chat_width`, given a viewport
/// height to cap against.
fn fit_geometry(width: u32, height: u32, chat_width: u16, viewport_height: u16) -> (u16, u16) {
    let cap_rows = ((viewport_height as u32 * MAX_VIEWPORT_FRACTION_PERCENT as u32) / 100)
        .clamp(MIN_IMAGE_ROWS as u32, u16::MAX as u32) as u16;
    fit_geometry_with_cap(width, height, chat_width, cap_rows)
}

/// Compute `(rows, cols)` for an image anchored inside the transcript body at a
/// given expand level. Uses a fixed (viewport-independent) row cap so the body's
/// prepared lines stay valid across resizes (the body cache is keyed by width
/// only); the expand level only swaps which fixed cap applies.
pub(crate) fn fit_geometry_anchored(
    width: u32,
    height: u32,
    chat_width: u16,
    level: ImageExpandLevel,
) -> (u16, u16) {
    fit_geometry_with_cap(width, height, chat_width, level.anchored_cap_rows())
}

/// Compute how many rows an inline image should occupy at `chat_width`, given a
/// viewport height to cap against.
#[cfg(test)]
fn fit_rows(width: u32, height: u32, chat_width: u16, viewport_height: u16) -> u16 {
    fit_geometry(width, height, chat_width, viewport_height).0
}

/// Build the dim label line shown above an inline image, e.g.
/// `  🖼 screenshot.png  1920×1080`, with a trailing show/hide badge
/// (`[Alt] [⇧] [I] hide` / `[Alt] [⇧] [I] show image`) so the toggle is
/// discoverable right where the image renders, plus a clickable `expand` badge
/// that cycles the per-image size. The expand badge renders a three-dot
/// progress indicator (`○○○` -> `●○○` -> `●●○`) reflecting the current level so
/// clicking it feels like a multi-step zoom.
/// Click/cursor icon that prefixes the clickable expand badge. Doubles as the
/// anchor cell for badge hit detection (see `expand_badge_start_col`), so any
/// change here must stay in sync with that scanner.
pub(crate) const EXPAND_BADGE_CLICK_ICON: &str = "🖱";

pub(crate) fn image_label_line(
    item: &InlineImageItem,
    images_visible: bool,
    level: ImageExpandLevel,
) -> Line<'static> {
    let dims = format!("{}×{}", item.width, item.height);
    let label = if item.label.is_empty() {
        dims
    } else {
        format!("{}  {}", item.label, dims)
    };
    let dim = Style::default().add_modifier(Modifier::DIM);
    let mut spans = vec![
        Span::styled("  🖼 ", dim),
        Span::styled(label, dim),
        Span::raw("  "),
        Span::styled(super::viewport::copy_badge_alt_badge(), dim),
        Span::styled(" [⇧] [I] ", dim),
    ];
    if images_visible {
        spans.push(Span::styled("hide", dim));
        // Clickable expand badge: a click/cursor icon signals the badge is
        // clickable, the dots show how far through the size cycle we are, then a
        // verb hints the next click's effect. The icon is also the first cell of
        // the badge hit-region (see `expand_badge_start_col`).
        spans.push(Span::raw("   "));
        spans.push(Span::styled(EXPAND_BADGE_CLICK_ICON, dim));
        spans.push(Span::raw(" "));
        spans.push(Span::styled(expand_dots(level), dim));
        spans.push(Span::raw(" "));
        spans.push(Span::styled(expand_verb(level), dim));
    } else {
        spans.push(Span::styled(
            "show image",
            Style::default().add_modifier(Modifier::DIM | Modifier::ITALIC),
        ));
    }
    Line::from(spans)
}

/// Three-dot progress indicator for the expand badge, filled to the current
/// level (`○○○` at Fit, `●○○` at Large, `●●○` at Huge).
fn expand_dots(level: ImageExpandLevel) -> String {
    const SLOTS: usize = 3;
    let filled = level.filled_dots();
    let mut out = String::with_capacity(SLOTS * 3);
    for slot in 0..SLOTS {
        out.push(if slot < filled { '●' } else { '○' });
    }
    out
}

/// Verb describing what the next expand-badge click does.
fn expand_verb(level: ImageExpandLevel) -> &'static str {
    match level {
        ImageExpandLevel::Fit | ImageExpandLevel::Large => "expand",
        ImageExpandLevel::Huge => "reset",
    }
}

/// Lines for images anchored at a transcript message: per image, a leading
/// blank, a dim label, a geometry-encoding marker line plus blank placeholder
/// rows (recognized by the image-region scan), and a trailing blank. When
/// `images_visible` is false the image collapses to just its label stub (with
/// a `show image` badge) and no placeholder rows are emitted, so nothing is
/// painted.
pub(crate) fn anchored_image_lines(
    items: &[InlineImageItem],
    width: u16,
    images_visible: bool,
    levels: &dyn ImageExpandLevels,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for item in items {
        let level = levels.expand_level(item.id);
        lines.push(Line::from(""));
        lines.push(image_label_line(item, images_visible, level));
        if images_visible {
            let (rows, cols) = fit_geometry_anchored(item.width, item.height, width, level);
            lines.extend(mermaid::inline_image_placeholder_lines(item.id, rows, cols));
        }
        lines.push(Line::from(""));
    }
    lines
}

/// Build the inline-images prepared section: a heading + correctly-sized
/// placeholder per image, with explicit `image_regions` (render = Fit) that the
/// viewport draws lazily. When `images_visible` is false each image collapses
/// to its label stub and no regions are emitted.
pub(crate) fn build_section(
    items: &[InlineImageItem],
    width: u16,
    viewport_height: u16,
    prefix_blank: bool,
    images_visible: bool,
    levels: &dyn ImageExpandLevels,
) -> PreparedMessages {
    use std::sync::Arc;

    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut image_regions: Vec<ImageRegion> = Vec::new();

    if items.is_empty() {
        return empty();
    }

    if prefix_blank {
        lines.push(Line::from(""));
    }

    for item in items {
        let level = levels.expand_level(item.id);
        lines.push(image_label_line(item, images_visible, level));

        if images_visible {
            // The bottom (unanchored) section is rebuilt every frame, not body
            // cached, so a viewport-relative default fit is fine here. Expanded
            // levels use the discrete fixed caps so they grow predictably.
            let (rows, cols) = match level {
                ImageExpandLevel::Fit => {
                    fit_geometry(item.width, item.height, width, viewport_height)
                }
                _ => {
                    fit_geometry_with_cap(item.width, item.height, width, level.anchored_cap_rows())
                }
            };
            let region_start = lines.len();
            for _ in 0..rows {
                lines.push(Line::from(""));
            }
            image_regions.push(ImageRegion {
                abs_line_idx: region_start,
                end_line: region_start + rows as usize,
                hash: item.id,
                height: rows,
                width: cols,
                render: ImageRegionRender::Fit,
            });
        }
        // Trailing spacer between images.
        lines.push(Line::from(""));
    }

    let line_count = lines.len();
    let plain: Vec<String> = lines
        .iter()
        .map(jcode_tui_render::line_plain_text)
        .collect();

    PreparedMessages {
        wrapped_lines: lines,
        wrapped_plain_lines: Arc::new(plain),
        wrapped_copy_offsets: Arc::new(vec![0; line_count]),
        raw_plain_lines: Arc::new(Vec::new()),
        wrapped_line_map: Arc::new(Vec::new()),
        wrapped_user_indices: Vec::new(),
        wrapped_user_prompt_starts: Vec::new(),
        wrapped_user_prompt_ends: Vec::new(),
        user_prompt_texts: Vec::new(),
        image_regions,
        edit_tool_ranges: Vec::new(),
        copy_targets: Vec::new(),
        message_boundaries: Vec::new(),
    }
}

fn empty() -> PreparedMessages {
    use std::sync::Arc;
    PreparedMessages {
        wrapped_lines: Vec::new(),
        wrapped_plain_lines: Arc::new(Vec::new()),
        wrapped_copy_offsets: Arc::new(Vec::new()),
        raw_plain_lines: Arc::new(Vec::new()),
        wrapped_line_map: Arc::new(Vec::new()),
        wrapped_user_indices: Vec::new(),
        wrapped_user_prompt_starts: Vec::new(),
        wrapped_user_prompt_ends: Vec::new(),
        user_prompt_texts: Vec::new(),
        image_regions: Vec::new(),
        edit_tool_ranges: Vec::new(),
        copy_targets: Vec::new(),
        message_boundaries: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(width: u32, height: u32) -> InlineImageItem {
        InlineImageItem {
            id: 0xABCD,
            width,
            height,
            label: "test.png".to_string(),
        }
    }

    /// 1x1 transparent PNG used by the materialize tests below.
    const MATERIALIZE_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

    #[test]
    fn materialize_visible_probe_is_cheap_after_first_call() {
        let id = mermaid::inline_image_id("image/png", MATERIALIZE_PNG_B64);
        register_payload(id, "image/png", MATERIALIZE_PNG_B64);
        assert!(materialize_visible(id), "first call decodes and caches");
        // Steady state: the in-memory probe alone must report ready, without
        // needing the payload registry at all.
        assert!(
            mermaid::inline_image_is_materialized(id),
            "presence probe should hit after materialization"
        );
        assert!(materialize_visible(id), "repeat call stays true");
    }

    #[test]
    fn prefetch_is_noop_for_materialized_image_without_kitty() {
        // Without a Kitty picker the fit-state path is Unsupported, so a
        // materialized image has nothing to prewarm: prefetch must be a cheap
        // no-op (no panic, no scheduling) and the image stays materialized.
        let id = mermaid::inline_image_id("image/png", MATERIALIZE_PNG_B64);
        register_payload(id, "image/png", MATERIALIZE_PNG_B64);
        assert!(materialize_visible(id));
        prefetch(id, 80, 10);
        assert!(
            mermaid::inline_image_is_materialized(id),
            "prefetch must not disturb already-materialized state"
        );
    }

    #[test]
    fn ensure_drawable_true_for_materialized_image_without_kitty() {
        // In tests no picker is initialized, so the stable-fit path reports
        // Unsupported; a materialized image must still be drawable so the
        // fallback renderers can run.
        let id = mermaid::inline_image_id("image/png", MATERIALIZE_PNG_B64);
        register_payload(id, "image/png", MATERIALIZE_PNG_B64);
        assert!(materialize_visible(id));
        assert!(
            ensure_drawable(id, 80, 10),
            "materialized image must be drawable on non-Kitty protocols"
        );
    }

    #[test]
    fn fit_rows_caps_tall_image_to_viewport_fraction() {
        // A very tall image must be capped so it cannot bury the transcript.
        let rows = fit_rows(100, 100_000, 80, 40);
        let cap = ((40u32 * MAX_VIEWPORT_FRACTION_PERCENT as u32) / 100) as u16;
        assert!(rows <= cap, "rows {rows} should be <= cap {cap}");
        assert!(rows >= MIN_IMAGE_ROWS);
    }

    #[test]
    fn fit_rows_never_below_minimum() {
        let rows = fit_rows(10, 10, 80, 40);
        assert!(rows >= MIN_IMAGE_ROWS);
    }

    #[test]
    fn fit_geometry_height_bound_image_narrows_proportionally() {
        // Tall image hits the viewport cap; the recorded cols must shrink with
        // it so the border/label hug the actual rendered picture.
        let (rows, cols) = fit_geometry(1000, 4000, 100, 40);
        let cap = ((40u32 * MAX_VIEWPORT_FRACTION_PERCENT as u32) / 100) as u16;
        assert!(rows <= cap);
        // Width-bound it would be ~100 cols; height-bound it must be far less.
        assert!(cols < 50, "height-bound image should be narrow, got {cols}");
        assert!(cols > 2, "image must occupy some columns, got {cols}");
    }

    #[test]
    fn fit_geometry_small_window_never_exceeds_chat_width() {
        for chat_width in [1u16, 2, 3, 5, 10] {
            for viewport_height in [1u16, 2, 5, 10] {
                let (rows, cols) = fit_geometry(1920, 1080, chat_width, viewport_height);
                assert!(
                    cols <= chat_width.max(2),
                    "cols {cols} > width {chat_width}"
                );
                assert!(rows >= MIN_IMAGE_ROWS);
            }
        }
    }

    #[test]
    fn fit_geometry_zero_dims_safe() {
        let (rows, cols) = fit_geometry(0, 0, 80, 40);
        assert!(rows >= MIN_IMAGE_ROWS);
        assert!(cols <= 80);
    }

    #[test]
    fn build_section_records_region_width() {
        let items = vec![item(600, 400)];
        let section = build_section(&items, 80, 40, false, true, &AllFit);
        let region = &section.image_regions[0];
        assert!(
            region.width > 2,
            "region width should include the image, got {}",
            region.width
        );
        assert!(region.width <= 80);
    }

    #[test]
    fn build_section_emits_one_fit_region_per_image_with_label() {
        let items = vec![item(600, 400), item(800, 600)];
        let section = build_section(&items, 80, 40, true, true, &AllFit);
        assert_eq!(section.image_regions.len(), 2);
        for region in &section.image_regions {
            assert_eq!(region.render, ImageRegionRender::Fit);
            assert_eq!(region.hash, 0xABCD);
            // The region must point at blank placeholder lines, never the label.
            let first = &section.wrapped_lines[region.abs_line_idx];
            assert!(
                jcode_tui_render::line_plain_text(first).trim().is_empty(),
                "region should start on a blank placeholder line"
            );
            // Region height must match its line span.
            assert_eq!(
                region.end_line - region.abs_line_idx,
                region.height as usize
            );
        }
        // A dim label line precedes the first region.
        let label_line = jcode_tui_render::line_plain_text(&section.wrapped_lines[1]);
        assert!(
            label_line.contains("test.png"),
            "label missing: {label_line:?}"
        );
    }

    #[test]
    fn build_section_is_empty_for_no_items() {
        let section = build_section(&[], 80, 40, false, true, &AllFit);
        assert!(section.wrapped_lines.is_empty());
        assert!(section.image_regions.is_empty());
    }

    #[test]
    fn build_section_hidden_collapses_to_label_stub_with_show_badge() {
        let items = vec![item(600, 400)];
        let section = build_section(&items, 80, 40, false, false, &AllFit);
        assert!(
            section.image_regions.is_empty(),
            "hidden images must not emit drawable regions"
        );
        let text: String = section
            .wrapped_lines
            .iter()
            .map(jcode_tui_render::line_plain_text)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("test.png"), "label should remain: {text:?}");
        assert!(
            text.contains("show image"),
            "show badge should render: {text:?}"
        );
    }

    #[test]
    fn visible_label_line_advertises_hide_badge() {
        let line = image_label_line(&item(600, 400), true, ImageExpandLevel::Fit);
        let text = jcode_tui_render::line_plain_text(&line);
        assert!(text.contains("[⇧] [I]"), "badge keys missing: {text:?}");
        assert!(text.contains("hide"), "hide hint missing: {text:?}");
    }

    #[test]
    fn expand_badge_dots_track_level() {
        // Fit shows no filled dots; each level fills one more.
        assert_eq!(expand_dots(ImageExpandLevel::Fit), "○○○");
        assert_eq!(expand_dots(ImageExpandLevel::Large), "●○○");
        assert_eq!(expand_dots(ImageExpandLevel::Huge), "●●○");
    }

    #[test]
    fn expand_badge_verb_resets_at_max() {
        assert_eq!(expand_verb(ImageExpandLevel::Fit), "expand");
        assert_eq!(expand_verb(ImageExpandLevel::Large), "expand");
        assert_eq!(expand_verb(ImageExpandLevel::Huge), "reset");
    }

    #[test]
    fn visible_label_line_shows_expand_badge() {
        let line = image_label_line(&item(600, 400), true, ImageExpandLevel::Large);
        let text = jcode_tui_render::line_plain_text(&line);
        assert!(text.contains("expand"), "expand badge missing: {text:?}");
        assert!(text.contains("●○○"), "expand dots missing: {text:?}");
    }

    #[test]
    fn hidden_label_line_omits_expand_badge() {
        let line = image_label_line(&item(600, 400), false, ImageExpandLevel::Fit);
        let text = jcode_tui_render::line_plain_text(&line);
        assert!(text.contains("show image"), "show badge missing: {text:?}");
        assert!(
            !text.contains("expand"),
            "hidden image must hide expand badge: {text:?}"
        );
    }

    #[test]
    fn expand_level_cycles_fit_large_huge() {
        assert_eq!(ImageExpandLevel::Fit.next(), ImageExpandLevel::Large);
        assert_eq!(ImageExpandLevel::Large.next(), ImageExpandLevel::Huge);
        assert_eq!(ImageExpandLevel::Huge.next(), ImageExpandLevel::Fit);
    }

    #[test]
    fn expanded_level_makes_anchored_image_taller() {
        let fit = fit_geometry_anchored(1000, 4000, 100, ImageExpandLevel::Fit).0;
        let large = fit_geometry_anchored(1000, 4000, 100, ImageExpandLevel::Large).0;
        let huge = fit_geometry_anchored(1000, 4000, 100, ImageExpandLevel::Huge).0;
        assert!(large > fit, "Large ({large}) should exceed Fit ({fit})");
        assert!(huge > large, "Huge ({huge}) should exceed Large ({large})");
    }

    #[test]
    fn anchored_image_lines_hidden_emit_no_placeholder_markers() {
        let items = vec![item(600, 400)];
        let lines = anchored_image_lines(&items, 80, false, &AllFit);
        assert!(
            lines
                .iter()
                .filter_map(mermaid::parse_inline_image_placeholder)
                .next()
                .is_none(),
            "hidden images must not emit geometry markers"
        );
        let text: String = lines
            .iter()
            .map(jcode_tui_render::line_plain_text)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("show image"), "show badge missing: {text:?}");
    }

    #[test]
    fn payload_registry_roundtrips() {
        register_payload(0xDEAD_BEEF, "image/png", "AAAA");
        let got = PAYLOAD_REGISTRY.lock().unwrap().get(0xDEAD_BEEF);
        assert_eq!(got, Some(("image/png".to_string(), "AAAA".to_string())));
    }

    /// 1x1 transparent PNG, used to exercise the real header parse.
    const TINY_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

    fn rendered_image(
        anchor: Option<crate::session::RenderedImageAnchor>,
    ) -> crate::session::RenderedImage {
        crate::session::RenderedImage {
            media_type: "image/png".to_string(),
            data: TINY_PNG_B64.to_string(),
            label: Some("tiny.png".to_string()),
            source: crate::session::RenderedImageSource::ToolResult {
                tool_name: "read".to_string(),
            },
            anchor,
        }
    }

    #[test]
    fn resolve_anchored_items_buckets_by_anchor() {
        let images = vec![
            rendered_image(Some(crate::session::RenderedImageAnchor::ToolCall {
                id: "tool-1".to_string(),
            })),
            rendered_image(Some(crate::session::RenderedImageAnchor::UserPrompt {
                ordinal: 2,
            })),
            rendered_image(None),
        ];
        let anchored = resolve_anchored_items(&images);
        assert!(anchored.has_anchored());
        assert_eq!(anchored.by_tool.get("tool-1").map(Vec::len), Some(1));
        assert_eq!(anchored.by_prompt.get(&2).map(Vec::len), Some(1));
        assert_eq!(anchored.unanchored.len(), 1);
    }

    #[test]
    fn unplaced_items_falls_back_for_missing_anchor_targets() {
        use jcode_tui_messages::DisplayMessage;

        let images = vec![
            rendered_image(Some(crate::session::RenderedImageAnchor::ToolCall {
                id: "tool-present".to_string(),
            })),
            rendered_image(Some(crate::session::RenderedImageAnchor::ToolCall {
                id: "tool-missing".to_string(),
            })),
            rendered_image(Some(crate::session::RenderedImageAnchor::UserPrompt {
                ordinal: 0,
            })),
            rendered_image(Some(crate::session::RenderedImageAnchor::UserPrompt {
                ordinal: 5,
            })),
            rendered_image(None),
        ];
        let anchored = resolve_anchored_items(&images);

        let tool_call = crate::message::ToolCall {
            id: "tool-present".to_string(),
            name: "read".to_string(),
            input: serde_json::Value::Null,
            intent: None,
            thought_signature: None,
        };
        let messages = vec![
            DisplayMessage::user("show me"),
            DisplayMessage::tool("output", tool_call),
        ];

        let unplaced = anchored.unplaced_items(&messages);
        // tool-missing (1) + prompt ordinal 5 (1) + unanchored (1) = 3.
        // tool-present and prompt 0 are placed in the body, not here.
        assert_eq!(unplaced.len(), 3);
    }

    #[test]
    fn anchored_image_lines_round_trip_through_region_scan() {
        let items = vec![item(600, 400)];
        let lines = anchored_image_lines(&items, 80, true, &AllFit);
        // Find the marker line and verify its geometry parse.
        let parsed: Vec<(u64, u16, u16)> = lines
            .iter()
            .filter_map(mermaid::parse_inline_image_placeholder)
            .collect();
        assert_eq!(parsed.len(), 1);
        let (hash, rows, cols) = parsed[0];
        assert_eq!(hash, 0xABCD);
        let (expected_rows, expected_cols) =
            fit_geometry_anchored(600, 400, 80, ImageExpandLevel::Fit);
        assert_eq!(rows, expected_rows);
        assert_eq!(cols, expected_cols);
        // Marker line is followed by rows-1 blank placeholder lines.
        let marker_idx = lines
            .iter()
            .position(|line| mermaid::parse_inline_image_placeholder(line).is_some())
            .unwrap();
        for offset in 1..rows as usize {
            let line = &lines[marker_idx + offset];
            assert!(
                jcode_tui_render::line_plain_text(line).trim().is_empty(),
                "placeholder row {offset} should be blank"
            );
        }
    }

    #[test]
    fn anchored_geometry_is_viewport_independent() {
        // The anchored fit must not depend on any viewport height so the body
        // cache (keyed by width only) stays valid across resizes.
        let (rows, cols) = fit_geometry_anchored(1920, 1080, 100, ImageExpandLevel::Fit);
        assert!(rows >= MIN_IMAGE_ROWS);
        assert!(rows <= ANCHORED_MAX_ROWS);
        assert!(cols <= 100);
    }
}
