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
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

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
/// drawn. Returns true on success. Cheap and idempotent on repeat.
pub(crate) fn materialize_visible(id: u64) -> bool {
    if let Some((media_type, data_b64)) = PAYLOAD_REGISTRY.lock().ok().and_then(|reg| reg.get(id)) {
        return mermaid::materialize_inline_image(&media_type, &data_b64).is_some();
    }
    false
}

/// Resolve the app's rendered images into lazily-sized inline items. Performs
/// only header-level work (no full decode) and registers each payload for the
/// later draw-time materialize.
#[cfg(test)]
pub(crate) fn resolve_items(images: &[crate::session::RenderedImage]) -> Vec<InlineImageItem> {
    let mut items = Vec::new();
    for image in images {
        if let Some(item) = resolve_item(image) {
            items.push(item);
        }
    }
    items
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
/// unanchored). Same lazy header-only cost profile as [`resolve_items`].
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
static ANCHORED_CACHE: LazyLock<
    Mutex<Option<((usize, u64), std::sync::Arc<AnchoredInlineImages>)>>,
> = LazyLock::new(|| Mutex::new(None));

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

/// Compute `(rows, cols)` for an image anchored inside the transcript body.
/// Uses a fixed row cap so the body's prepared lines stay independent of the
/// viewport height (the body cache is keyed by width only).
pub(crate) fn fit_geometry_anchored(width: u32, height: u32, chat_width: u16) -> (u16, u16) {
    fit_geometry_with_cap(width, height, chat_width, ANCHORED_MAX_ROWS)
}

/// Compute how many rows an inline image should occupy at `chat_width`, given a
/// viewport height to cap against.
#[cfg(test)]
fn fit_rows(width: u32, height: u32, chat_width: u16, viewport_height: u16) -> u16 {
    fit_geometry(width, height, chat_width, viewport_height).0
}

/// Build the dim label line shown above an inline image, e.g.
/// `  🖼 screenshot.png  1920×1080`.
pub(crate) fn image_label_line(item: &InlineImageItem) -> Line<'static> {
    let dims = format!("{}×{}", item.width, item.height);
    let label = if item.label.is_empty() {
        dims
    } else {
        format!("{}  {}", item.label, dims)
    };
    Line::from(vec![
        Span::styled("  🖼 ", Style::default().add_modifier(Modifier::DIM)),
        Span::styled(label, Style::default().add_modifier(Modifier::DIM)),
    ])
}

/// Lines for images anchored at a transcript message: per image, a leading
/// blank, a dim label, a geometry-encoding marker line plus blank placeholder
/// rows (recognized by the image-region scan), and a trailing blank.
pub(crate) fn anchored_image_lines(items: &[InlineImageItem], width: u16) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for item in items {
        lines.push(Line::from(""));
        lines.push(image_label_line(item));
        let (rows, cols) = fit_geometry_anchored(item.width, item.height, width);
        lines.extend(mermaid::inline_image_placeholder_lines(item.id, rows, cols));
        lines.push(Line::from(""));
    }
    lines
}

/// Build the inline-images prepared section: a heading + correctly-sized
/// placeholder per image, with explicit `image_regions` (render = Fit) that the
/// viewport draws lazily.
pub(crate) fn build_section(
    items: &[InlineImageItem],
    width: u16,
    viewport_height: u16,
    prefix_blank: bool,
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
        lines.push(image_label_line(item));

        let (rows, cols) = fit_geometry(item.width, item.height, width, viewport_height);
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
        let section = build_section(&items, 80, 40, false);
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
        let section = build_section(&items, 80, 40, true);
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
        let section = build_section(&[], 80, 40, false);
        assert!(section.wrapped_lines.is_empty());
        assert!(section.image_regions.is_empty());
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
        let lines = anchored_image_lines(&items, 80);
        // Find the marker line and verify its geometry parse.
        let parsed: Vec<(u64, u16, u16)> = lines
            .iter()
            .filter_map(mermaid::parse_inline_image_placeholder)
            .collect();
        assert_eq!(parsed.len(), 1);
        let (hash, rows, cols) = parsed[0];
        assert_eq!(hash, 0xABCD);
        let (expected_rows, expected_cols) = fit_geometry_anchored(600, 400, 80);
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
        let (rows, cols) = fit_geometry_anchored(1920, 1080, 100);
        assert!(rows >= MIN_IMAGE_ROWS);
        assert!(rows <= ANCHORED_MAX_ROWS);
        assert!(cols <= 100);
    }
}
