mod animation;
mod desktop_app_driver;
mod desktop_benchmark;
mod desktop_config;
mod desktop_gallery;
mod desktop_ipc;
mod desktop_issue_browser;
mod desktop_issue_cache;
mod desktop_log;
mod desktop_prefs;
mod desktop_protocol;
mod desktop_rich_text;
mod desktop_scene;
mod desktop_session_events;
mod desktop_ui_engine;
mod desktop_worker_host;
mod power_inhibit;
mod render_helpers;
mod session_data;
mod session_launch;
mod single_session;
mod single_session_render;
#[cfg(test)]
#[path = "state_space_tests.rs"]
mod state_space_tests;
mod workspace;

mod workspace_vertices;
pub(crate) use workspace_vertices::*;
mod hero_mask;
pub(crate) use hero_mask::*;
mod canvas;
pub(crate) use canvas::*;
mod desktop_profiling;
pub(crate) use desktop_profiling::*;
mod desktop_benchmarks_run;
pub(crate) use desktop_benchmarks_run::*;
mod desktop_benchmarks_transcript;
pub(crate) use desktop_benchmarks_transcript::*;
mod desktop_benchmarks_scroll;
pub(crate) use desktop_benchmarks_scroll::*;
mod desktop_reload;
pub(crate) use desktop_reload::*;
mod desktop_events_glue;
pub(crate) use desktop_events_glue::*;
use ab_glyph::{Font, FontArc, Glyph as AbGlyph, PxScale, ScaleFont, point};
use animation::{
    APP_MODE_TRANSITION_DURATION, AnimatedRect, AnimatedViewport, ColorTransition, FocusPulse,
    StatusTextTransition, StatusTextTransitionFrame, StatusTextVisualFrame,
    SurfaceTransitionAnimator, SurfaceVisualFrame, SurfaceVisualTarget, VisibleColumnLayout,
    WorkspaceRenderLayout,
};
use anyhow::{Context, Result};
use base64::Engine;
use bytemuck::{Pod, Zeroable};
use desktop_app_driver::{
    DESKTOP_UI_SNAPSHOT_VERSION, DesktopAppDriver, DesktopAppRuntime, DesktopSceneBuildContext,
    DesktopSingleSessionSnapshot, DesktopSnapshotRestoreError, DesktopSurfaceSnapshot,
    DesktopUiSnapshot, DesktopWorkspaceSnapshot, DesktopWorkspaceSurfaceSnapshot,
};
use desktop_benchmark::*;
use desktop_config::*;
use desktop_ipc::{DesktopHostToWorkerEnvelope, write_desktop_ipc_frame};
#[cfg(test)]
pub(crate) use desktop_issue_browser::IssueBrowserLayoutMode;
use desktop_issue_browser::{
    IssueBrowserLayout, compose_single_session_issue_browser_vertices, issue_browser_layout,
};
use desktop_protocol::{
    DesktopHostToWorkerMessage, DesktopInputEvent, DesktopKeyEvent, DesktopKeyModifiers,
    DesktopMouseButton, DesktopMouseEvent, DesktopProtocolEnvelope, DesktopSceneUpdate,
    DesktopSessionEventBatchWire, DesktopSessionEventWire, DesktopSnapshotResponse,
    DesktopWindowEvent, DesktopWindowState, DesktopWorkerInit, DesktopWorkerMode,
    DesktopWorkerReady, DesktopWorkerShutdownReason, DesktopWorkerToHostMessage,
};
use desktop_scene::{
    DesktopColor, DesktopDisplayCommand, DesktopRect as DesktopSceneRect, DesktopRectPaint,
    DesktopScene, DesktopSceneViewport,
};
use desktop_session_events::{
    BACKEND_EVENT_FORWARD_INTERVAL, BACKEND_EVENT_FORWARD_MAX_PAYLOAD_BYTES,
    BACKEND_EVENT_FORWARD_MAX_RAW_EVENTS, DesktopSessionEventBatch,
    coalesce_desktop_session_events, collect_desktop_session_event_batch,
    spawn_session_event_forwarder,
};
use desktop_worker_host::DesktopWorkerConnection;
use glyphon::{
    Attrs, Buffer, Color as TextColor, Family, FontSystem, Metrics, Resolution, Shaping,
    SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Wrap,
};
use image::RgbaImage;
use render_helpers::*;
use session_launch::DesktopSessionStatus;
use single_session::{
    ReasoningEffortCycleOutcome, SINGLE_SESSION_FONT_FAMILY, SINGLE_SESSION_WELCOME_FONT_FAMILY,
    SelectionPoint, SingleSessionApp, SingleSessionLineStyle, SingleSessionMessage,
    SingleSessionStyledLine, handwritten_welcome_phrase, single_session_surface,
    single_session_typography, single_session_typography_for_scale,
};
use single_session_render::*;
use wgpu::{CompositeAlphaMode, PresentMode, SurfaceError, TextureUsages};
use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, Event, MouseButton, MouseScrollDelta, TouchPhase, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::{Fullscreen, Window, WindowBuilder};
use workspace::{InputMode, KeyInput, KeyOutcome, PanelSizePreset, Workspace};

use std::borrow::Cow;
use std::collections::{HashMap, HashSet, hash_map::DefaultHasher};
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_WINDOW_WIDTH: f64 = 1280.0;
const DEFAULT_WINDOW_HEIGHT: f64 = 800.0;
const DESKTOP_RELOAD_WINDOW_ENV: &str = "JCODE_DESKTOP_RELOAD_WINDOW";
const DESKTOP_RELOAD_HANDOFF_READY_ENV: &str = "JCODE_DESKTOP_RELOAD_READY_FILE";
const DESKTOP_RELOAD_HANDOFF_RELEASE_ENV: &str = "JCODE_DESKTOP_RELOAD_RELEASE_FILE";
const DESKTOP_RELOAD_HANDOFF_POLL_INTERVAL: Duration = Duration::from_millis(25);
const DESKTOP_RELOAD_HANDOFF_TIMEOUT: Duration = Duration::from_secs(8);
const DESKTOP_RELOAD_STARTUP_RELEASE_TIMEOUT: Duration = Duration::from_secs(3);
const DESKTOP_RELOAD_MAX_RESTORED_DIMENSION: u32 = 32_768;
const OUTER_PADDING: f32 = 12.0;
const GAP: f32 = 10.0;
const STATUS_BAR_HEIGHT: f32 = 30.0;
const FOCUSED_BORDER_WIDTH: f32 = 2.0;
const UNFOCUSED_BORDER_WIDTH: f32 = 1.5;
const PANEL_RADIUS: f32 = 12.0;
const STATUS_RADIUS: f32 = 9.0;
const ROUNDED_CORNER_SEGMENTS: usize = 6;
const PANEL_FIT_TOLERANCE: f32 = 0.15;
const STATUS_PREVIEW_LANE_RADIUS: i32 = 2;
const STATUS_PREVIEW_MAX_WIDTH: f32 = 420.0;
const STATUS_PREVIEW_HEIGHT: f32 = 14.0;
const STATUS_PREVIEW_PANEL_WIDTH: f32 = 9.0;
const STATUS_PREVIEW_PANEL_GAP: f32 = 2.0;
const STATUS_PREVIEW_GROUP_GAP: f32 = 10.0;
const STATUS_PREVIEW_SIDE_RESERVE: f32 = 74.0;
const STATUS_PREVIEW_MAX_TICKS_PER_LANE: i32 = 32;
const SPACE_HOLD_PROGRESS_HEIGHT: f32 = 7.0;
const SPACE_HOLD_PROGRESS_WIDTH_FRACTION: f32 = 0.36;
const SPACE_HOLD_PROGRESS_TRACK_COLOR: [f32; 4] = [0.055, 0.060, 0.075, 0.96];
const SPACE_HOLD_PROGRESS_FILL_COLOR: [f32; 4] = [0.180, 0.900, 0.470, 1.0];
const WORKSPACE_NUMBER_LEFT_PADDING: f32 = 14.0;
const WORKSPACE_NUMBER_DIGIT_WIDTH: f32 = 8.0;
const WORKSPACE_NUMBER_DIGIT_HEIGHT: f32 = 14.0;
const WORKSPACE_NUMBER_DIGIT_GAP: f32 = 4.0;
const WORKSPACE_NUMBER_STROKE: f32 = 2.0;
const BITMAP_TEXT_PIXEL: f32 = 2.0;
const STATUS_TEXT_RIGHT_PADDING: f32 = 14.0;
const PANEL_TITLE_LEFT_PADDING: f32 = 12.0;
const PANEL_TITLE_TOP_PADDING: f32 = 12.0;
const PANEL_BODY_TOP_PADDING: f32 = 38.0;
const PANEL_BODY_LINE_GAP: f32 = 8.0;
const SINGLE_SESSION_DRAFT_TOP_OFFSET: f32 = 158.0;
const SINGLE_SESSION_CARET_WIDTH: f32 = 2.0;
const SINGLE_SESSION_CARET_COLOR: [f32; 4] = [0.130, 0.150, 0.190, 0.92];
const SESSION_SPAWN_REFRESH_DELAY: Duration = Duration::from_millis(350);
const BACKGROUND_POLL_INTERVAL: Duration = Duration::from_millis(33);
const BACKEND_REDRAW_FRAME_INTERVAL: Duration = Duration::from_millis(16);
/// Minimum spacing between animation-driven redraws.
///
/// Without this, the desktop render loop re-requests a redraw immediately after
/// every animated frame (welcome-hero reveal, focus pulse, spinners, smooth
/// scroll, etc.). Because the surface uses non-blocking `Mailbox` presentation,
/// `present()` returns instantly, so the unthrottled loop renders at hundreds of
/// fps and pins the main thread near 100% CPU, starving input handling and the
/// compositor (the root cause of desktop lag/jank). ~16ms paces continuous
/// animations to about 60fps, matching typical display refresh.
const DESKTOP_ANIMATION_FRAME_INTERVAL: Duration = Duration::from_millis(16);
const SURFACE_TIMEOUT_BACKOFF_MIN: Duration = Duration::from_millis(16);
const SURFACE_TIMEOUT_BACKOFF_MAX: Duration = Duration::from_millis(250);
const HEADLESS_CHAT_SMOKE_TIMEOUT: Duration = Duration::from_secs(90);
const DESKTOP_SPINNER_FRAME_MS: u128 = 180;
const MOUSE_WHEEL_LINES_PER_DETENT: f32 = 3.0;
const MAX_MOUSE_SCROLL_LINES_PER_EVENT: f32 = 24.0;
const SCROLL_GESTURE_IDLE_RESET: Duration = Duration::from_millis(180);
const SCROLL_FRACTIONAL_EPSILON: f32 = 0.000_1;
const SCROLL_MOMENTUM_GAIN: f32 = 8.5;
const SCROLL_MOMENTUM_DECAY_PER_SECOND: f32 = 7.0;
const SCROLL_MOMENTUM_MAX_VELOCITY: f32 = 72.0;
const SCROLL_MOMENTUM_STOP_VELOCITY: f32 = 0.08;
const SCROLL_FRAME_MAX_DT_SECONDS: f32 = 0.050;
const SINGLE_SESSION_SCROLL_ANIMATION_DURATION: Duration = Duration::from_millis(90);
const SINGLE_SESSION_BODY_TEXT_WINDOW_BEFORE_LINES: usize = 8;
const SINGLE_SESSION_BODY_TEXT_WINDOW_AFTER_LINES: usize = 16;
const SINGLE_SESSION_STREAMING_BODY_TEXT_WINDOW_BEFORE_LINES: usize = 2;
const SINGLE_SESSION_STREAMING_BODY_TEXT_WINDOW_AFTER_LINES: usize = 4;
const STREAMING_TEXT_FADE_DURATION: Duration = Duration::from_millis(150);
const STREAMING_TEXT_FADE_START_OPACITY: f32 = 0.4;
const STREAMING_TEXT_RISE_START_OFFSET_PIXELS: f32 = 3.5;
const STREAMING_TEXT_HANDOFF_DURATION: Duration = Duration::from_millis(135);
const STREAMING_TEXT_HANDOFF_START_OPACITY: f32 = 0.18;
const DESKTOP_ASYNC_JOB_LIMIT: usize = 12;
const PRIMITIVE_VERTEX_BUFFER_MIN_CAPACITY: usize = 1024;
const PRIMITIVE_VERTEX_BUFFER_SHRINK_RATIO: usize = 4;
const WORKSPACE_BASE_VERTEX_CAPACITY_HINT: usize = 512;
const WORKSPACE_SURFACE_VERTEX_CAPACITY_HINT: usize = 2048;
static DESKTOP_ASYNC_JOB_COUNT: AtomicUsize = AtomicUsize::new(0);

struct DesktopAsyncJobPermit<'a> {
    counter: &'a AtomicUsize,
}

impl Drop for DesktopAsyncJobPermit<'_> {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::Relaxed);
    }
}

fn try_acquire_desktop_async_job_slot<'a>(
    counter: &'a AtomicUsize,
    limit: usize,
) -> Result<DesktopAsyncJobPermit<'a>> {
    let mut current = counter.load(Ordering::Relaxed);
    loop {
        if current >= limit {
            anyhow::bail!("desktop async job limit reached ({limit})");
        }
        match counter.compare_exchange_weak(
            current,
            current + 1,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => return Ok(DesktopAsyncJobPermit { counter }),
            Err(next_current) => current = next_current,
        }
    }
}

fn spawn_bounded_desktop_async_job(
    name: impl Into<String>,
    job: impl FnOnce() + Send + 'static,
) -> Result<()> {
    let name = name.into();
    let permit =
        try_acquire_desktop_async_job_slot(&DESKTOP_ASYNC_JOB_COUNT, DESKTOP_ASYNC_JOB_LIMIT)
            .with_context(|| format!("failed to start {name}"))?;
    std::thread::Builder::new()
        .name(name.clone())
        .spawn(move || {
            let _permit = permit;
            job();
        })
        .with_context(|| format!("failed to spawn {name}"))?;
    Ok(())
}

#[derive(Clone)]
struct DesktopReasoningEffortRequestQueue {
    request_tx: mpsc::Sender<DesktopReasoningEffortRequest>,
    latest_generation: Arc<AtomicU64>,
}

struct DesktopReasoningEffortRequest {
    generation: u64,
    effort: String,
    target_session_id: Option<String>,
    event_tx: session_launch::DesktopSessionEventSender,
}

impl DesktopReasoningEffortRequestQueue {
    fn request(
        &self,
        effort: String,
        target_session_id: Option<String>,
        event_tx: session_launch::DesktopSessionEventSender,
    ) -> Result<()> {
        let generation = self.latest_generation.fetch_add(1, Ordering::AcqRel) + 1;
        self.request_tx
            .send(DesktopReasoningEffortRequest {
                generation,
                effort,
                target_session_id,
                event_tx,
            })
            .context("failed to queue desktop reasoning effort change")
    }
}

fn spawn_desktop_reasoning_effort_request_queue() -> Result<DesktopReasoningEffortRequestQueue> {
    let (request_tx, request_rx) = mpsc::channel();
    let latest_generation = Arc::new(AtomicU64::new(0));
    let worker_latest_generation = Arc::clone(&latest_generation);
    std::thread::Builder::new()
        .name("jcode-desktop-effort-queue".to_string())
        .spawn(move || {
            run_desktop_reasoning_effort_request_queue(request_rx, worker_latest_generation);
        })
        .context("failed to spawn desktop reasoning effort queue")?;
    Ok(DesktopReasoningEffortRequestQueue {
        request_tx,
        latest_generation,
    })
}

fn run_desktop_reasoning_effort_request_queue(
    request_rx: mpsc::Receiver<DesktopReasoningEffortRequest>,
    latest_generation: Arc<AtomicU64>,
) {
    while let Ok(mut request) = request_rx.recv() {
        let mut coalesced = 0usize;
        let mut disconnected = false;
        loop {
            match request_rx.try_recv() {
                Ok(next_request) => {
                    request = next_request;
                    coalesced += 1;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        if coalesced > 0 {
            desktop_log::info(format_args!(
                "jcode-desktop: coalesced {coalesced} superseded reasoning effort request(s); applying {}",
                desktop_log::truncate_for_log(&request.effort, 64)
            ));
        }
        apply_desktop_reasoning_effort_request(request, &latest_generation);
        if disconnected {
            break;
        }
    }
}

fn apply_desktop_reasoning_effort_request(
    request: DesktopReasoningEffortRequest,
    latest_generation: &AtomicU64,
) {
    let (response_tx, response_rx) = mpsc::channel();
    let result = session_launch::set_reasoning_effort(
        &request.effort,
        request.target_session_id.as_deref(),
        Some(response_tx),
    );
    let still_latest = latest_generation.load(Ordering::Acquire) == request.generation;
    if still_latest {
        for event in response_rx.try_iter() {
            let _ = request.event_tx.send(event);
        }
        if let Err(error) = result {
            desktop_log::error(format_args!(
                "jcode-desktop: reasoning effort sync failed generation={} target_session={}: {error:#}",
                request.generation,
                request.target_session_id.as_deref().unwrap_or("<current>")
            ));
            let _ = request
                .event_tx
                .send(session_launch::DesktopSessionEvent::Status(
                    DesktopSessionStatus::ReasoningEffortFailed(format!("{error:#}")),
                ));
        }
    } else if let Err(error) = result {
        desktop_log::warn(format_args!(
            "jcode-desktop: stale reasoning effort sync failed generation={} target_session={}: {error:#}",
            request.generation,
            request.target_session_id.as_deref().unwrap_or("<current>")
        ));
    } else {
        let dropped = response_rx.try_iter().count();
        desktop_log::info(format_args!(
            "jcode-desktop: dropped stale reasoning effort response generation={} event_count={dropped}",
            request.generation
        ));
    }
}

#[derive(Clone, Debug, Default)]
struct SurfaceTimeoutBackoff {
    consecutive_timeouts: u32,
}

impl SurfaceTimeoutBackoff {
    fn reset(&mut self) {
        self.consecutive_timeouts = 0;
    }

    fn record_timeout(&mut self) -> (Duration, u32) {
        let exponent = self.consecutive_timeouts.min(4);
        self.consecutive_timeouts = self.consecutive_timeouts.saturating_add(1);
        let delay = SURFACE_TIMEOUT_BACKOFF_MIN
            .saturating_mul(1_u32 << exponent)
            .min(SURFACE_TIMEOUT_BACKOFF_MAX);
        (delay, self.consecutive_timeouts)
    }
}

fn desktop_surface_size_is_renderable(size: PhysicalSize<u32>) -> bool {
    size.width > 0 && size.height > 0
}

fn desktop_background_wake(
    now: Instant,
    surface_renderable: bool,
    frame_animation_active: bool,
) -> Option<Instant> {
    if surface_renderable && frame_animation_active {
        Some(now + BACKGROUND_POLL_INTERVAL)
    } else {
        None
    }
}

/// Compute the next paced animation redraw time.
///
/// Returns `Some(now + DESKTOP_ANIMATION_FRAME_INTERVAL)` while an animation is
/// active and `None` once it settles. Callers schedule this instead of calling
/// `request_redraw()` immediately, which would render as fast as the CPU allows
/// (the surface presents without blocking) and pin the main thread near 100%
/// CPU, starving input handling and the compositor.
fn next_animation_redraw_at(now: Instant, animation_active: bool) -> Option<Instant> {
    animation_active.then(|| now + DESKTOP_ANIMATION_FRAME_INTERVAL)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct StreamingTextArrivalStyle {
    opacity: f32,
    y_offset_pixels: f32,
    active: bool,
}

fn streaming_text_arrival_style_for_elapsed(elapsed: Duration) -> StreamingTextArrivalStyle {
    if animation::desktop_reduced_motion_enabled() {
        return StreamingTextArrivalStyle {
            opacity: 1.0,
            y_offset_pixels: 0.0,
            active: false,
        };
    }

    let progress =
        (elapsed.as_secs_f32() / STREAMING_TEXT_FADE_DURATION.as_secs_f32()).clamp(0.0, 1.0);
    if progress >= 1.0 {
        return StreamingTextArrivalStyle {
            opacity: 1.0,
            y_offset_pixels: 0.0,
            active: false,
        };
    }
    let eased = animation::ease_out_cubic(progress);
    StreamingTextArrivalStyle {
        opacity: STREAMING_TEXT_FADE_START_OPACITY
            + (1.0 - STREAMING_TEXT_FADE_START_OPACITY) * eased,
        y_offset_pixels: STREAMING_TEXT_RISE_START_OFFSET_PIXELS * (1.0 - eased),
        active: true,
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn streaming_text_fade_opacity_for_elapsed(elapsed: Duration) -> (f32, bool) {
    let style = streaming_text_arrival_style_for_elapsed(elapsed);
    (style.opacity, style.active)
}

fn streaming_text_fade_start_after_len_change(
    previous_len: usize,
    next_len: usize,
    current_started_at: Option<Instant>,
    now: Instant,
) -> Option<Instant> {
    if next_len == 0 {
        return None;
    }

    let fade_active = current_started_at.is_some_and(|started_at| {
        now.saturating_duration_since(started_at) < STREAMING_TEXT_FADE_DURATION
    });
    if fade_active {
        return current_started_at;
    }

    // Only fade in the beginning of a streaming response. Restarting after
    // every slow delta dims the already-visible response and reads as flicker.
    if previous_len == 0 && next_len > 0 {
        Some(now)
    } else {
        None
    }
}

fn streaming_text_handoff_style_for_elapsed(elapsed: Duration) -> StreamingTextArrivalStyle {
    if animation::desktop_reduced_motion_enabled() {
        return StreamingTextArrivalStyle {
            opacity: 0.0,
            y_offset_pixels: 0.0,
            active: false,
        };
    }

    let progress =
        (elapsed.as_secs_f32() / STREAMING_TEXT_HANDOFF_DURATION.as_secs_f32()).clamp(0.0, 1.0);
    if progress >= 1.0 {
        return StreamingTextArrivalStyle {
            opacity: 0.0,
            y_offset_pixels: 0.0,
            active: false,
        };
    }

    let eased = animation::ease_out_cubic(progress);
    StreamingTextArrivalStyle {
        opacity: STREAMING_TEXT_HANDOFF_START_OPACITY * (1.0 - eased),
        y_offset_pixels: 0.0,
        active: true,
    }
}

/// Body cache key that also tracks how much of the streaming response is
/// revealed, so the cached wrapped lines rebuild as the reveal advances.
fn streaming_reveal_body_cache_key(
    rendered_body_key: u64,
    streaming_response_empty: bool,
    revealed_bytes: usize,
) -> u64 {
    if streaming_response_empty {
        return rendered_body_key;
    }
    let mut hasher = DefaultHasher::new();
    rendered_body_key.hash(&mut hasher);
    revealed_bytes.hash(&mut hasher);
    hasher.finish()
}

fn streaming_text_handoff_start_after_len_change(
    previous_len: usize,
    next_len: usize,
    has_visible_streaming_buffer: bool,
    current_started_at: Option<Instant>,
    now: Instant,
) -> Option<Instant> {
    if animation::desktop_reduced_motion_enabled() || next_len > 0 {
        return None;
    }

    if previous_len > 0 && has_visible_streaming_buffer {
        return Some(now);
    }

    current_started_at.filter(|started_at| {
        now.saturating_duration_since(*started_at) < STREAMING_TEXT_HANDOFF_DURATION
    })
}
const DESKTOP_120FPS_FRAME_BUDGET: Duration = Duration::from_micros(8_333);
const DESKTOP_PRESENT_STALL_BUDGET: Duration = Duration::from_millis(33);
const DESKTOP_INPUT_LATENCY_BUDGET: Duration = Duration::from_millis(25);
const DESKTOP_NO_PAINT_BUDGET: Duration = Duration::from_millis(250);
const DESKTOP_FRAME_PROFILE_REPORT_INTERVAL: Duration = Duration::from_secs(1);

const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.955,
    g: 0.965,
    b: 0.985,
    a: 1.0,
};

const BACKGROUND_TOP_LEFT: [f32; 4] = [0.890, 0.930, 1.000, 1.0];
const BACKGROUND_TOP_RIGHT: [f32; 4] = [0.960, 0.910, 1.000, 1.0];
const BACKGROUND_BOTTOM_RIGHT: [f32; 4] = [0.875, 0.980, 0.930, 1.0];
const BACKGROUND_BOTTOM_LEFT: [f32; 4] = [0.945, 0.960, 0.995, 1.0];
const FOCUS_RING_COLOR: [f32; 4] = [0.135, 0.155, 0.205, 0.90];
const NAV_STATUS_COLOR: [f32; 4] = [0.145, 0.165, 0.220, 1.0];
const INSERT_STATUS_COLOR: [f32; 4] = [0.245, 0.395, 0.340, 1.0];
const STATUS_PREVIEW_ACTIVE_GROUP_COLOR: [f32; 4] = [0.953, 0.965, 0.984, 0.16];
const STATUS_PREVIEW_EMPTY_FOCUSED_COLOR: [f32; 4] = [0.953, 0.965, 0.984, 0.50];
const STATUS_PREVIEW_VIEWPORT_COLOR: [f32; 4] = [0.953, 0.965, 0.984, 0.78];
const WORKSPACE_NUMBER_COLOR: [f32; 4] = [0.953, 0.965, 0.984, 0.90];
const STATUS_TEXT_COLOR: [f32; 4] = [0.953, 0.965, 0.984, 0.88];
const PANEL_TITLE_COLOR: [f32; 4] = [0.010, 0.014, 0.025, 1.0];
const PANEL_BODY_COLOR: [f32; 4] = [0.008, 0.012, 0.020, 1.0];
const ASSISTANT_TEXT_COLOR: [f32; 4] = [0.026, 0.034, 0.052, 1.0];
const ASSISTANT_HEADING_TEXT_COLOR: [f32; 4] = [0.030, 0.095, 0.300, 1.0];
const ASSISTANT_QUOTE_TEXT_COLOR: [f32; 4] = [0.210, 0.090, 0.355, 1.0];
const ASSISTANT_TABLE_TEXT_COLOR: [f32; 4] = [0.000, 0.155, 0.185, 1.0];
const ASSISTANT_LINK_TEXT_COLOR: [f32; 4] = [0.000, 0.170, 0.430, 1.0];
const USER_TEXT_COLOR: [f32; 4] = [0.012, 0.030, 0.180, 1.0];
const USER_CONTINUATION_TEXT_COLOR: [f32; 4] = [0.018, 0.035, 0.155, 1.0];
const TOOL_TEXT_COLOR: [f32; 4] = [0.150, 0.095, 0.325, 1.0];
const TOOL_DETAIL_TEXT_COLOR: [f32; 4] = [0.135, 0.155, 0.220, 1.0];
const TOOL_MUTED_TEXT_COLOR: [f32; 4] = [0.345, 0.365, 0.430, 0.96];
const TOOL_RUNNING_TEXT_COLOR: [f32; 4] = [0.045, 0.265, 0.640, 1.0];
const TOOL_SUCCESS_TEXT_COLOR: [f32; 4] = [0.035, 0.360, 0.220, 1.0];
const TOOL_FAILED_TEXT_COLOR: [f32; 4] = [0.560, 0.070, 0.095, 1.0];
const TOOL_PENDING_TEXT_COLOR: [f32; 4] = [0.320, 0.345, 0.405, 1.0];
const TOOL_CARD_BACKGROUND_COLOR: [f32; 4] = [0.985, 0.990, 1.000, 0.68];
const TOOL_CARD_ACTIVE_BACKGROUND_COLOR: [f32; 4] = [0.890, 0.945, 1.000, 0.72];
const TOOL_CARD_SUCCESS_BACKGROUND_COLOR: [f32; 4] = [0.875, 0.975, 0.925, 0.56];
const TOOL_CARD_FAILED_BACKGROUND_COLOR: [f32; 4] = [1.000, 0.900, 0.910, 0.64];
const TOOL_CARD_GROUP_BACKGROUND_COLOR: [f32; 4] = [0.945, 0.930, 1.000, 0.50];
const TOOL_CARD_BORDER_COLOR: [f32; 4] = [0.105, 0.165, 0.295, 0.22];
const TOOL_CARD_ACTIVE_BORDER_COLOR: [f32; 4] = [0.000, 0.260, 0.720, 0.36];
const TOOL_TIMELINE_RAIL_COLOR: [f32; 4] = [0.105, 0.165, 0.295, 0.20];
const TOOL_TIMELINE_ACTIVE_RAIL_COLOR: [f32; 4] = [0.000, 0.260, 0.720, 0.46];
const TOOL_OUTPUT_DRAWER_COLOR: [f32; 4] = [0.030, 0.055, 0.095, 0.070];
const TOOL_STATUS_CHIP_COLOR: [f32; 4] = [1.000, 1.000, 1.000, 0.42];
const META_TEXT_COLOR: [f32; 4] = [0.095, 0.110, 0.155, 0.98];
const CODE_TEXT_COLOR: [f32; 4] = [0.055, 0.065, 0.095, 1.0];
const STATUS_TEXT_ACCENT_COLOR: [f32; 4] = [0.030, 0.125, 0.080, 1.0];
const ERROR_TEXT_COLOR: [f32; 4] = [0.360, 0.000, 0.000, 1.0];
const OVERLAY_TEXT_COLOR: [f32; 4] = [0.030, 0.045, 0.075, 1.0];
const OVERLAY_SELECTION_TEXT_COLOR: [f32; 4] = [0.010, 0.035, 0.105, 1.0];
const USER_PROMPT_ACCENT_COLOR: [f32; 4] = [0.000, 0.105, 0.250, 1.0];
const PANEL_SECTION_COLOR: [f32; 4] = [0.045, 0.055, 0.080, 0.95];
const SELECTION_HIGHLIGHT_COLOR: [f32; 4] = [0.220, 0.420, 0.700, 0.22];
const WELCOME_AURORA_BLUE: [f32; 4] = [0.250, 0.520, 1.000, 0.145];
const WELCOME_AURORA_VIOLET: [f32; 4] = [0.720, 0.360, 0.980, 0.125];
const WELCOME_AURORA_MINT: [f32; 4] = [0.220, 0.840, 0.660, 0.115];
const WELCOME_AURORA_WARM: [f32; 4] = [1.000, 0.620, 0.360, 0.075];
const WELCOME_HANDWRITING_COLOR: [f32; 4] = [0.012, 0.080, 0.250, 0.94];
const NATIVE_SPINNER_HEAD_COLOR: [f32; 4] = [0.000, 0.260, 0.720, 1.0];
const CODE_BLOCK_BACKGROUND_COLOR: [f32; 4] = [0.075, 0.095, 0.135, 0.105];
const INLINE_CODE_BACKGROUND_COLOR: [f32; 4] = [0.075, 0.095, 0.135, 0.175];
const QUOTE_CARD_BACKGROUND_COLOR: [f32; 4] = [0.520, 0.330, 0.760, 0.090];
const TABLE_CARD_BACKGROUND_COLOR: [f32; 4] = [0.080, 0.460, 0.520, 0.085];
const ERROR_CARD_BACKGROUND_COLOR: [f32; 4] = [0.850, 0.170, 0.170, 0.105];
const OVERLAY_SELECTION_BACKGROUND_COLOR: [f32; 4] = [0.280, 0.470, 0.780, 0.115];
const STATUS_PREVIEW_ACCENTS: [[f32; 3]; 8] = [
    [0.560, 0.690, 0.980],
    [0.780, 0.610, 0.910],
    [0.520, 0.760, 0.620],
    [0.900, 0.650, 0.450],
    [0.600, 0.780, 0.840],
    [0.880, 0.580, 0.690],
    [0.720, 0.740, 0.820],
    [0.810, 0.760, 0.520],
];



fn main() {
    desktop_log::init();
    install_desktop_diagnostic_hooks();
    desktop_log::info(format_args!(
        "jcode-desktop: starting pid={} version={} build_hash={}",
        std::process::id(),
        desktop_header_version_label(),
        desktop_build_hash_label()
    ));

    if let Err(error) = pollster::block_on(run()) {
        desktop_log::error(format_args!("jcode-desktop: fatal error: {error:#}"));
        std::process::exit(1);
    }
}

fn install_desktop_diagnostic_hooks() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        desktop_log::error(format_args!("jcode-desktop: panic: {panic_info}"));
        desktop_log::error(format_args!(
            "jcode-desktop: panic backtrace: {}",
            std::backtrace::Backtrace::force_capture()
        ));
        default_hook(panic_info);
    }));
}

async fn run() -> Result<()> {
    log_desktop_platform_support_warning();
    let args = std::env::args().collect::<Vec<_>>();
    let startup_benchmark = startup_benchmark_requested(&args);
    let startup_content_benchmark = startup_content_benchmark_requested(&args);
    let startup_trace = DesktopStartupTrace::new(
        startup_benchmark || startup_content_benchmark || startup_log_requested(&args),
    );
    startup_trace.mark("args parsed");
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{}", desktop_help_text());
        return Ok(());
    }
    if args.iter().any(|arg| arg == "--version" || arg == "-V") {
        println!(
            "{} {}",
            desktop_header_version_label(),
            desktop_build_hash_label()
        );
        return Ok(());
    }
    if let Some(message) = headless_chat_smoke_message(&args) {
        return run_headless_chat_smoke(message);
    }
    if let Some(frames) = resize_render_benchmark_frames(&args) {
        return run_resize_render_benchmark(frames);
    }
    if let Some(frames) = scroll_render_benchmark_frames(&args) {
        return run_scroll_render_benchmark(frames);
    }
    if let Some(frames) = real_transcript_scroll_benchmark_frames(&args) {
        return run_real_transcript_scroll_benchmark(frames);
    }
    if let Some(frames) = real_transcript_action_benchmark_frames(&args) {
        return run_real_transcript_action_benchmark(frames);
    }
    if let Some(output_dir) = hero_screenshot_capture_dir(&args) {
        return run_hero_screenshot_capture(&output_dir).await;
    }
    if let Some(capture) = gallery_screenshot_capture_request(&args) {
        return run_gallery_screenshot_capture(&capture).await;
    }
    if let Some(raw_events) = stream_e2e_benchmark_raw_events(&args) {
        return run_stream_e2e_benchmark(raw_events);
    }
    if desktop_gallery::launcher_requested(&args) {
        return desktop_gallery::launch_temporary_windows();
    }
    let fullscreen = args.iter().any(|arg| arg == "--fullscreen");
    let desktop_gallery_state = desktop_gallery::state_from_args(&args);
    let desktop_gallery = desktop_gallery_state.is_some();
    let process_role = desktop_process_role_from_args(args.iter().map(String::as_str));
    let desktop_mode = desktop_mode_from_args(args.iter().map(String::as_str));
    if process_role == DesktopProcessRole::AppWorker {
        return run_desktop_app_worker_process(desktop_mode);
    }
    let resume_session_id = desktop_resume_session_id_from_args(args.iter().map(String::as_str));
    let desktop_reload_startup = DesktopReloadStartup::from_env();
    emit_desktop_profile_event(
        "jcode-desktop-launch-profile",
        serde_json::json!({
            "mode": desktop_mode.as_str(),
            "process_role": process_role.as_str(),
            "version": desktop_header_version_label(),
            "build_hash": desktop_build_hash_label(),
            "pid": std::process::id(),
        }),
    );
    let event_loop = EventLoopBuilder::<DesktopUserEvent>::with_user_event()
        .build()
        .context("failed to create event loop")?;
    let event_loop_proxy = event_loop.create_proxy();
    startup_trace.mark("event loop created");
    let mut window_builder = WindowBuilder::new().with_title("Jcode Desktop");
    if let Some(placement) = desktop_reload_startup.window_placement {
        window_builder = placement.apply_to_window_builder(window_builder);
    } else {
        window_builder = window_builder.with_inner_size(LogicalSize::new(
            DEFAULT_WINDOW_WIDTH,
            DEFAULT_WINDOW_HEIGHT,
        ));
    }

    if desktop_reload_startup.hidden_until_handoff_release() {
        window_builder = window_builder.with_visible(false);
    }

    if fullscreen {
        window_builder = window_builder.with_fullscreen(Some(Fullscreen::Borderless(None)));
    }

    let window = Arc::new(
        window_builder
            .build(&event_loop)
            .context("failed to create desktop window")?,
    );
    startup_trace.mark("window created");
    let mut renderer = DesktopHostRendererState::NoGpuBoot;
    renderer.start_gpu_init(window.clone(), event_loop_proxy.clone(), startup_trace)?;
    startup_trace.mark("canvas init spawned");

    let mut pending_workspace_startup_load = false;
    let mut pending_workspace_startup_preferences = None;
    let mut app = if let Some(gallery_state) = desktop_gallery_state.as_deref() {
        desktop_gallery::temporary_app(gallery_state)
    } else if desktop_mode == DesktopMode::WorkspacePrototype {
        let mut workspace = Workspace::loading_sessions();
        if let Some(preferences) = load_desktop_preferences() {
            workspace.apply_preferences(preferences.clone());
            pending_workspace_startup_preferences = Some(preferences);
        }
        pending_workspace_startup_load = true;
        DesktopApp::Workspace(workspace)
    } else {
        initial_single_session_app(resume_session_id.as_deref())
    };
    startup_trace.mark("app state initialized");
    window.set_title(&app.status_title());
    let mut reload_startup_handoff = desktop_reload_startup.handoff;
    let mut modifiers = ModifiersState::empty();
    let mut cursor_position = winit::dpi::PhysicalPosition::new(0.0, 0.0);
    let mut selecting_body = false;
    let mut selecting_draft = false;
    let mut scroll_accumulator = ScrollLineAccumulator::default();
    let mut scroll_metrics_cache = SingleSessionScrollMetricsCache::default();
    let mut hot_reloader = DesktopHotReloader::new(process_role.reload_strategy());
    if process_role == DesktopProcessRole::StableHost {
        hot_reloader.start_app_worker_for_current_binary(&app, &window, "stable host startup");
    }
    let preferences_save_tx = spawn_desktop_preferences_saver();
    let mut power_inhibitor = power_inhibit::PowerInhibitor::new();
    let (session_event_tx, session_event_rx) = mpsc::channel();
    spawn_session_event_forwarder(session_event_rx, event_loop_proxy.clone());
    if simulate_stream_requested(&args) && app.is_single_session() {
        // Dev-only: drive the real streaming pipeline with synthetic, bursty
        // TextDelta events so the streaming reveal animation can be observed and
        // recorded live without a backend. Mirrors provider chunk cadence.
        if let DesktopApp::SingleSession(single) = &mut app {
            seed_desktop_stream_simulator_transcript(single);
        }
        window.set_title(&app.status_title());
        spawn_desktop_stream_simulator(session_event_tx.clone());
    }
    let reasoning_effort_queue = spawn_desktop_reasoning_effort_request_queue()?;
    let mut recovery_scan_pending = app.is_single_session() && !desktop_gallery;
    let mut first_frame_presented = false;
    let mut first_content_frame_presented = false;
    let mut interaction_latency = DesktopInteractionLatencyProfiler::new();
    let mut no_paint_watchdog = DesktopNoPaintWatchdog::new();
    let mut last_backend_redraw_request: Option<Instant> = None;
    let mut pending_backend_redraw_since: Option<Instant> = None;
    let mut surface_timeout_backoff = SurfaceTimeoutBackoff::default();
    let mut surface_timeout_redraw_at: Option<Instant> = None;
    // Scheduled time for the next animation-driven redraw. Continuous animations
    // re-arm this each presented frame so the loop paces itself to roughly the
    // display refresh rate instead of busy-spinning the main thread.
    let mut animation_redraw_at: Option<Instant> = None;
    let mut pending_resize: Option<PhysicalSize<u32>> = None;
    let mut space_hold_started_at: Option<Instant> = None;
    let mut space_hold_consumed = false;
    let mut github_issue_sync_running = false;
    let mut desktop_clipboard = DesktopClipboard::default();

    if pending_workspace_startup_load {
        spawn_session_cards_load(
            DesktopSessionCardsPurpose::WorkspaceInitialLoad,
            event_loop_proxy.clone(),
            Duration::ZERO,
        );
    }

    let mut event_loop_entered = false;
    event_loop.run(move |event, target| {
        if !event_loop_entered {
            event_loop_entered = true;
            startup_trace.mark("event loop entered");
        }
        let event_loop_now = Instant::now();
        let surface_renderable = desktop_surface_size_is_renderable(window.inner_size());
        let renderer_ready = renderer.is_gpu_ready();
        let has_background_work = app.has_background_work();
        power_inhibitor.set_active(has_background_work);
        let default_wake = desktop_background_wake(
            event_loop_now,
            surface_renderable,
            app.has_frame_animation(),
        );
        let backend_wake = pending_backend_redraw_since
            .and(last_backend_redraw_request)
            .map(|last| last + BACKEND_REDRAW_FRAME_INTERVAL);
        let hot_reload_wake = hot_reloader.next_wake(event_loop_now);
        let space_hold_wake = space_hold_started_at.and_then(|started_at| match &app {
            DesktopApp::Workspace(workspace) if !space_hold_consumed => {
                Some(started_at + workspace.space_hold_toggle_duration())
            }
            _ => None,
        });
        let wake = [
            default_wake,
            backend_wake,
            hot_reload_wake,
            space_hold_wake,
            surface_timeout_redraw_at,
            animation_redraw_at,
        ]
            .into_iter()
            .flatten()
            .min();
        if let Some(wake) = wake {
            target.set_control_flow(ControlFlow::WaitUntil(wake));
        } else {
            target.set_control_flow(ControlFlow::Wait);
        }

        let pending_interaction_kind = interaction_latency.pending_kind();
        let frame_animation_active = app.has_frame_animation();
        let pending_backend_redraw = pending_backend_redraw_since.is_some();
        let no_paint_active = surface_renderable
            && renderer_ready
            && (!first_frame_presented
                || has_background_work
                || frame_animation_active
                || pending_backend_redraw
                || pending_interaction_kind.is_some());
        if no_paint_watchdog.observe_active_tick(
            event_loop_now,
            NoPaintWatchdogContext {
                active: no_paint_active,
                mode: app.mode(),
                has_background_work,
                frame_animation_active,
                pending_backend_redraw,
                pending_interaction_kind,
            },
        ) {
            window.request_redraw();
        }
        let worker_drain = hot_reloader.drain_app_worker_messages();
        if let Some(scene) = worker_drain.latest_scene {
            // Keep receiving worker scenes so the IPC path stays exercised, but do
            // not make them primary yet. The worker currently emits only the
            // display-list skeleton, while the in-process host renderer still owns
            // the complete desktop UI. Rendering the worker scene here regresses
            // normal launches to a blank/gray window.
            drop(scene);
            window.request_redraw();
        }
        if worker_drain.reload_requested {
            show_desktop_reload_notice(&mut app);
            window.set_title(&app.status_title());
            window.request_redraw();
            if hot_reloader.force_reload(&app, &window) {
                target.exit();
                return;
            }
        }

        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(size) => {
                    pending_resize = Some(size);
                    forward_app_worker_input(
                        &mut hot_reloader,
                        DesktopInputEvent::Window(DesktopWindowEvent::Resized {
                            width: size.width,
                            height: size.height,
                            scale_factor: window.scale_factor() as f32,
                        }),
                    );
                    window.request_redraw();
                }
                WindowEvent::ScaleFactorChanged { .. } => {
                    pending_resize = Some(window.inner_size());
                    let size = window.inner_size();
                    forward_app_worker_input(
                        &mut hot_reloader,
                        DesktopInputEvent::Window(DesktopWindowEvent::Resized {
                            width: size.width,
                            height: size.height,
                            scale_factor: window.scale_factor() as f32,
                        }),
                    );
                    window.request_redraw();
                }
                WindowEvent::Focused(focused) => {
                    forward_app_worker_input(
                        &mut hot_reloader,
                        DesktopInputEvent::Window(DesktopWindowEvent::Focused(focused)),
                    );
                }
                WindowEvent::ModifiersChanged(new_modifiers) => {
                    modifiers = new_modifiers.state();
                }
                WindowEvent::MouseWheel { delta, phase, .. } => {
                    forward_app_worker_input(
                        &mut hot_reloader,
                        DesktopInputEvent::Mouse(desktop_mouse_wheel_event(delta)),
                    );
                    let size = window.inner_size();
                    let now = Instant::now();
                    let previous_smooth_scroll = app.single_session_smooth_scroll_lines(
                        scroll_accumulator.pending_lines(),
                        size,
                        &mut scroll_metrics_cache,
                    );
                    let mut should_redraw = false;
                    if !app.is_single_session() {
                        scroll_accumulator.reset();
                        scroll_metrics_cache.clear();
                    } else if let Some(lines) = scroll_accumulator.scroll_lines(delta, now) {
                        should_redraw |=
                            app.scroll_single_session_body(lines, size, &mut scroll_metrics_cache);
                    }
                    if matches!(phase, TouchPhase::Cancelled) {
                        scroll_accumulator.reset();
                    }
                    let next_smooth_scroll = app.single_session_smooth_scroll_lines(
                        scroll_accumulator.pending_lines(),
                        size,
                        &mut scroll_metrics_cache,
                    );
                    should_redraw |= (next_smooth_scroll - previous_smooth_scroll).abs()
                        >= SCROLL_FRACTIONAL_EPSILON;
                    if should_redraw {
                        interaction_latency.mark("mouse_wheel", now);
                        window.request_redraw();
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    let cursor_started = Instant::now();
                    cursor_position = position;
                    forward_app_worker_input(
                        &mut hot_reloader,
                        DesktopInputEvent::Mouse(DesktopMouseEvent::Move {
                            x: cursor_position.x as f32,
                            y: cursor_position.y as f32,
                        }),
                    );
                    if selecting_draft
                        && app.update_single_session_draft_selection_at(
                            cursor_position.x as f32,
                            cursor_position.y as f32,
                            window.inner_size(),
                        )
                    {
                        interaction_latency.mark("draft_selection_drag", cursor_started);
                        window.request_redraw();
                    } else if selecting_body
                        && app.update_single_session_selection_at(
                            cursor_position.x as f32,
                            cursor_position.y as f32,
                            window.inner_size(),
                        )
                    {
                        interaction_latency.mark("body_selection_drag", cursor_started);
                        window.request_redraw();
                    }
                }
                WindowEvent::MouseInput {
                    state,
                    button: MouseButton::Left,
                    ..
                } => {
                    let mouse_started = Instant::now();
                    forward_app_worker_input(
                        &mut hot_reloader,
                        DesktopInputEvent::Mouse(DesktopMouseEvent::Button {
                            button: DesktopMouseButton::Left,
                            pressed: state == ElementState::Pressed,
                        }),
                    );
                    match state {
                        ElementState::Pressed => {
                        if app.begin_single_session_draft_selection_at(
                            cursor_position.x as f32,
                            cursor_position.y as f32,
                            window.inner_size(),
                        ) {
                            selecting_body = false;
                            selecting_draft = true;
                            window.set_title(&app.status_title());
                            interaction_latency.mark("mouse_press", mouse_started);
                            window.request_redraw();
                            return;
                        }

                        selecting_draft = false;
                        selecting_body = app.begin_single_session_selection_at(
                            cursor_position.x as f32,
                            cursor_position.y as f32,
                            window.inner_size(),
                        );
                        if selecting_body {
                            interaction_latency.mark("mouse_press", mouse_started);
                            window.request_redraw();
                        }
                    }
                    ElementState::Released => {
                        if selecting_draft {
                            app.update_single_session_draft_selection_at(
                                cursor_position.x as f32,
                                cursor_position.y as f32,
                                window.inner_size(),
                            );
                            selecting_draft = false;
                            let selected = app.selected_single_session_draft_text();
                            if let Some(text) = selected {
                                copy_text_to_clipboard(
                                    &mut desktop_clipboard,
                                    &text,
                                    "copied input selection",
                                    &mut app,
                                );
                            }
                            window.set_title(&app.status_title());
                            interaction_latency.mark("mouse_release", mouse_started);
                            window.request_redraw();
                        } else if selecting_body {
                            app.update_single_session_selection_at(
                                cursor_position.x as f32,
                                cursor_position.y as f32,
                                window.inner_size(),
                            );
                            selecting_body = false;
                            let selected = app.selected_single_session_text(window.inner_size());
                            if let Some(text) = selected {
                                copy_text_to_clipboard(
                                    &mut desktop_clipboard,
                                    &text,
                                    "copied selection",
                                    &mut app,
                                );
                            }
                            window.set_title(&app.status_title());
                            interaction_latency.mark("mouse_release", mouse_started);
                            window.request_redraw();
                        }
                    }
                    }
                }
                WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Released => {
                    if app.is_workspace() && is_space_key(&event.logical_key) {
                        if space_hold_started_at.take().is_some()
                            && !space_hold_consumed
                            && matches!(&app, DesktopApp::Workspace(workspace) if workspace.mode == InputMode::Insert)
                            && matches!(app.handle_key(KeyInput::Character(" ".to_string())), KeyOutcome::Redraw)
                        {
                            window.set_title(&app.status_title());
                            window.request_redraw();
                        }
                        space_hold_consumed = false;
                    }
                }
                WindowEvent::KeyboardInput { event, .. }
                    if event.state == ElementState::Pressed =>
                {
                    let keyboard_started = Instant::now();
                    let size = window.inner_size();
                    let had_smooth_scroll = app
                        .single_session_smooth_scroll_lines(
                            scroll_accumulator.pending_lines(),
                            size,
                            &mut scroll_metrics_cache,
                        )
                        .abs()
                        >= SCROLL_FRACTIONAL_EPSILON;
                    scroll_accumulator.reset();
                    if had_smooth_scroll {
                        window.request_redraw();
                    }
                    if app.is_workspace()
                        && is_space_key(&event.logical_key)
                        && modifiers.is_empty()
                    {
                        if space_hold_started_at.is_none() {
                            space_hold_started_at = Some(keyboard_started);
                            space_hold_consumed = false;
                        }
                        window.request_redraw();
                        return;
                    }

                    let key_input = to_key_input(&event.logical_key, modifiers);
                    let key_debug = format!("{key_input:?}");
                    interaction_latency.mark("keyboard_input", keyboard_started);
                    if hot_reloader.has_app_worker() {
                        forward_app_worker_input(
                            &mut hot_reloader,
                            DesktopInputEvent::Key(desktop_key_event_from_winit(
                                &event.logical_key,
                                modifiers,
                                true,
                            )),
                        );
                        window.request_redraw();
                    }
                    if key_input == KeyInput::RefreshSessions && app.is_workspace() {
                        spawn_session_cards_load(
                            DesktopSessionCardsPurpose::WorkspaceRefresh,
                            event_loop_proxy.clone(),
                            Duration::ZERO,
                        );
                        window.request_redraw();
                        return;
                    }

                    match app.handle_key(key_input) {
                        KeyOutcome::Exit => target.exit(),
                        KeyOutcome::Redraw => {
                            if let DesktopApp::Workspace(workspace) = &app {
                                queue_desktop_preferences_save(workspace, &preferences_save_tx);
                            }
                            window.set_title(&app.status_title());
                            window.request_redraw();
                        }
                        KeyOutcome::OpenSession { session_id, title } => {
                            if let DesktopApp::Workspace(workspace) = &app {
                                queue_desktop_preferences_save(workspace, &preferences_save_tx);
                            }
                            if app.promote_focused_workspace_session() {
                                scroll_accumulator = ScrollLineAccumulator::default();
                                scroll_metrics_cache = SingleSessionScrollMetricsCache::default();
                                window.set_title(&app.status_title());
                                window.request_redraw();
                            } else if let Err(error) =
                                session_launch::launch_validated_resume_session(&session_id, &title)
                            {
                                desktop_log::error(format_args!(
                                    "jcode-desktop: failed to open session {session_id}: {error:#}"
                                ));
                            }
                        }
                        KeyOutcome::SpawnSession => {
                            if let DesktopApp::SingleSession(app) = &mut app {
                                app.reset_fresh_session();
                                window.set_title(&app.status_title());
                                window.request_redraw();
                                return;
                            }

                            if let Err(error) = session_launch::launch_new_session() {
                                desktop_log::error(format_args!(
                                    "jcode-desktop: failed to spawn session: {error:#}"
                                ));
                            } else {
                                spawn_session_cards_load(
                                    DesktopSessionCardsPurpose::WorkspaceRefresh,
                                    event_loop_proxy.clone(),
                                    SESSION_SPAWN_REFRESH_DELAY,
                                );
                                window.request_redraw();
                            }
                        }
                        KeyOutcome::SpawnSelfDevSession => {
                            if let Err(error) = session_launch::launch_selfdev_session() {
                                desktop_log::error(format_args!(
                                    "jcode-desktop: failed to spawn self-dev session: {error:#}"
                                ));
                            }
                        }
                        KeyOutcome::SpawnHomeSession => {
                            if let Err(error) = session_launch::launch_home_session() {
                                desktop_log::error(format_args!(
                                    "jcode-desktop: failed to spawn home session: {error:#}"
                                ));
                            }
                        }
                        KeyOutcome::SendDraft {
                            session_id,
                            title,
                            message,
                            images,
                        } => {
                            if app.is_single_session() {
                                match session_launch::spawn_message_to_session(
                                    session_id.clone(),
                                    message,
                                    images,
                                    session_event_tx.clone(),
                                ) {
                                    Ok(handle) => app.set_single_session_handle(handle),
                                    Err(error) => apply_single_session_error(&mut app, error),
                                }
                                window.set_title(&app.status_title());
                                window.request_redraw();
                            } else if !images.is_empty() {
                                match session_launch::spawn_message_to_session(
                                    session_id.clone(),
                                    message,
                                    images,
                                    session_event_tx.clone(),
                                ) {
                                    Ok(_handle) => {
                                        spawn_session_cards_load(
                                            DesktopSessionCardsPurpose::WorkspaceRefresh,
                                            event_loop_proxy.clone(),
                                            SESSION_SPAWN_REFRESH_DELAY,
                                        );
                                        window.request_redraw();
                                    }
                                    Err(error) => desktop_log::error(format_args!(
                                        "jcode-desktop: failed to send image draft to {session_id}: {error:#}"
                                    )),
                                }
                            } else if let Err(error) = session_launch::send_message_to_session(
                                &session_id,
                                &title,
                                &message,
                            ) {
                                desktop_log::error(format_args!(
                                    "jcode-desktop: failed to send draft to {session_id}: {error:#}"
                                ));
                            } else {
                                spawn_session_cards_load(
                                    DesktopSessionCardsPurpose::WorkspaceRefresh,
                                    event_loop_proxy.clone(),
                                    SESSION_SPAWN_REFRESH_DELAY,
                                );
                                window.request_redraw();
                            }
                        }
                        KeyOutcome::StartFreshSession { message, images } => {
                            match session_launch::spawn_fresh_server_session(
                                message,
                                images,
                                session_event_tx.clone(),
                            ) {
                                Ok(handle) => app.set_single_session_handle(handle),
                                Err(error) => apply_single_session_error(&mut app, error),
                            }
                            window.set_title(&app.status_title());
                            window.request_redraw();
                        }
                        KeyOutcome::CancelGeneration => {
                            app.cancel_single_session_generation();
                            window.set_title(&app.status_title());
                            window.request_redraw();
                        }
                        KeyOutcome::CopyLatestResponse(text) => {
                            copy_text_to_clipboard(
                                &mut desktop_clipboard,
                                &text,
                                "copied latest response",
                                &mut app,
                            );
                            window.set_title(&app.status_title());
                            window.request_redraw();
                        }
                        KeyOutcome::CopyText {
                            text,
                            success_notice,
                        } => {
                            copy_text_to_clipboard(
                                &mut desktop_clipboard,
                                &text,
                                success_notice,
                                &mut app,
                            );
                            window.set_title(&app.status_title());
                            window.request_redraw();
                        }
                        KeyOutcome::CutDraftToClipboard(text) => {
                            copy_text_to_clipboard(
                                &mut desktop_clipboard,
                                &text,
                                "cut input line",
                                &mut app,
                            );
                            window.set_title(&app.status_title());
                            window.request_redraw();
                        }
                        KeyOutcome::CycleModel(direction) => {
                            if let Err(error) = session_launch::spawn_cycle_model(
                                direction,
                                app.single_session_live_id(),
                                session_event_tx.clone(),
                            ) {
                                apply_single_session_error(&mut app, error);
                            } else {
                                app.apply_session_event(session_launch::DesktopSessionEvent::Status(
                                    DesktopSessionStatus::SwitchingModel,
                                ));
                            }
                            window.set_title(&app.status_title());
                            window.request_redraw();
                        }
                        KeyOutcome::CycleReasoningEffort(direction) => {
                            let target_session_id = app.single_session_live_id();
                            let outcome = app.preview_single_session_reasoning_effort_cycle(direction);
                            match outcome {
                                ReasoningEffortCycleOutcome::Set(effort) => {
                                    if let Err(error) = reasoning_effort_queue.request(
                                        effort,
                                        target_session_id,
                                        session_event_tx.clone(),
                                    ) {
                                        apply_single_session_error(&mut app, error);
                                    }
                                }
                                ReasoningEffortCycleOutcome::AlreadyAtLimit { .. }
                                | ReasoningEffortCycleOutcome::Unavailable => {}
                            }
                            window.set_title(&app.status_title());
                            window.request_redraw();
                        }
                        KeyOutcome::LoadModelCatalog => {
                            if let Err(error) = session_launch::spawn_load_model_catalog(
                                app.single_session_live_id(),
                                session_event_tx.clone(),
                            ) {
                                apply_single_session_error(&mut app, error);
                            }
                            window.set_title(&app.status_title());
                            window.request_redraw();
                        }
                        KeyOutcome::LoadSessionSwitcher => {
                            let purpose = if app.is_workspace() {
                                DesktopSessionCardsPurpose::WorkspaceRefresh
                            } else {
                                DesktopSessionCardsPurpose::SingleSessionSwitcher
                            };
                            spawn_session_cards_load(
                                purpose,
                                event_loop_proxy.clone(),
                                Duration::ZERO,
                            );
                            window.set_title(&app.status_title());
                            window.request_redraw();
                        }
                        KeyOutcome::RestoreCrashedSessions => {
                            spawn_restore_crashed_sessions(event_loop_proxy.clone());
                            window.set_title(&app.status_title());
                            window.request_redraw();
                        }
                        KeyOutcome::SetModel(model) => {
                            if let Err(error) = session_launch::spawn_set_model(
                                model,
                                app.single_session_live_id(),
                                session_event_tx.clone(),
                            ) {
                                apply_single_session_error(&mut app, error);
                            } else {
                                app.apply_session_event(session_launch::DesktopSessionEvent::Status(
                                    DesktopSessionStatus::SwitchingModel,
                                ));
                            }
                            window.set_title(&app.status_title());
                            window.request_redraw();
                        }
                        KeyOutcome::RefreshModelCatalog => {
                            if let Err(error) = session_launch::spawn_refresh_models(
                                app.single_session_live_id(),
                                session_event_tx.clone(),
                            ) {
                                apply_single_session_error(&mut app, error);
                            }
                            window.set_title(&app.status_title());
                            window.request_redraw();
                        }
                        KeyOutcome::SetReasoningEffort(effort) => {
                            let target_session_id = app.single_session_live_id();
                            match app.preview_single_session_reasoning_effort_set(&effort) {
                                Some(effort) => {
                                    if app
                                        .set_reasoning_effort_via_active_session(effort.clone())
                                        .is_err()
                                        && let Err(error) = reasoning_effort_queue.request(
                                            effort,
                                            target_session_id,
                                            session_event_tx.clone(),
                                        )
                                    {
                                        apply_single_session_error(&mut app, error);
                                    }
                                }
                                None => app.set_single_session_status_label(
                                    "thinking level is not available for this model",
                                ),
                            }
                            window.set_title(&app.status_title());
                            window.request_redraw();
                        }
                        KeyOutcome::SetServiceTier(service_tier) => {
                            if let Err(error) = session_launch::spawn_set_service_tier(
                                service_tier,
                                app.single_session_live_id(),
                                session_event_tx.clone(),
                            ) {
                                apply_single_session_error(&mut app, error);
                            } else {
                                app.apply_session_event(session_launch::DesktopSessionEvent::Status(
                                    DesktopSessionStatus::external("setting fast mode"),
                                ));
                            }
                            window.set_title(&app.status_title());
                            window.request_redraw();
                        }
                        KeyOutcome::SetTransport(transport) => {
                            if let Err(error) = session_launch::spawn_set_transport(
                                transport,
                                app.single_session_live_id(),
                                session_event_tx.clone(),
                            ) {
                                apply_single_session_error(&mut app, error);
                            } else {
                                app.apply_session_event(session_launch::DesktopSessionEvent::Status(
                                    DesktopSessionStatus::external("setting transport"),
                                ));
                            }
                            window.set_title(&app.status_title());
                            window.request_redraw();
                        }
                        KeyOutcome::SetCompactionMode(mode) => {
                            if let Err(error) = session_launch::spawn_set_compaction_mode(
                                mode,
                                app.single_session_live_id(),
                                session_event_tx.clone(),
                            ) {
                                apply_single_session_error(&mut app, error);
                            } else {
                                app.apply_session_event(session_launch::DesktopSessionEvent::Status(
                                    DesktopSessionStatus::external("setting compaction mode"),
                                ));
                            }
                            window.set_title(&app.status_title());
                            window.request_redraw();
                        }
                        KeyOutcome::CompactSession => {
                            if let Err(error) = session_launch::spawn_compact_session(
                                app.single_session_live_id(),
                                session_event_tx.clone(),
                            ) {
                                apply_single_session_error(&mut app, error);
                            } else {
                                app.apply_session_event(session_launch::DesktopSessionEvent::Status(
                                    DesktopSessionStatus::external("requesting compaction"),
                                ));
                            }
                            window.set_title(&app.status_title());
                            window.request_redraw();
                        }
                        KeyOutcome::RenameSession(title) => {
                            if let Err(error) = session_launch::spawn_rename_session(
                                title,
                                app.single_session_live_id(),
                                session_event_tx.clone(),
                            ) {
                                apply_single_session_error(&mut app, error);
                            } else {
                                app.apply_session_event(session_launch::DesktopSessionEvent::Status(
                                    DesktopSessionStatus::external("renaming session"),
                                ));
                            }
                            window.set_title(&app.status_title());
                            window.request_redraw();
                        }
                        KeyOutcome::ClearServerSession => {
                            if let Err(error) = session_launch::spawn_clear_server_session(
                                app.single_session_live_id(),
                                session_event_tx.clone(),
                            ) {
                                apply_single_session_error(&mut app, error);
                            } else {
                                app.apply_session_event(session_launch::DesktopSessionEvent::Status(
                                    DesktopSessionStatus::external("clearing session"),
                                ));
                            }
                            window.set_title(&app.status_title());
                            window.request_redraw();
                        }
                        KeyOutcome::SendStdinResponse { request_id, input } => {
                            if let Err(error) = app.send_single_session_stdin_response(request_id, input)
                            {
                                apply_single_session_error(&mut app, error);
                            }
                            window.set_title(&app.status_title());
                            window.request_redraw();
                        }
                        KeyOutcome::AttachClipboardImage => {
                            match clipboard_image_png_base64(&mut desktop_clipboard) {
                                Ok((media_type, base64_data)) => {
                                    app.attach_clipboard_image(media_type, base64_data);
                                }
                                Err(error) => apply_single_session_error(&mut app, error),
                            }
                            window.set_title(&app.status_title());
                            window.request_redraw();
                        }
                        KeyOutcome::PasteText => {
                            if let Err(error) =
                                paste_clipboard_into_app(&mut desktop_clipboard, &mut app)
                            {
                                apply_single_session_error(&mut app, error);
                            }
                            window.set_title(&app.status_title());
                            window.request_redraw();
                        }
                        KeyOutcome::ForceReload => {
                            if hot_reloader.force_reload(&app, &window) {
                                target.exit();
                            } else {
                                window.set_title(&app.status_title());
                                window.request_redraw();
                            }
                        }
                        KeyOutcome::None => {}
                    }
                    if start_pending_github_issue_sync(
                        &mut app,
                        &mut github_issue_sync_running,
                        event_loop_proxy.clone(),
                    ) {
                        window.set_title(&app.status_title());
                        window.request_redraw();
                    }
                    if start_pending_transcript_hydration(&mut app, event_loop_proxy.clone()) {
                        window.request_redraw();
                    }
                    log_desktop_slow_interaction(
                        "keyboard_input",
                        keyboard_started.elapsed(),
                        serde_json::json!({ "key": key_debug }),
                    );
                }
                WindowEvent::RedrawRequested => {
                    let Some(canvas) = renderer.canvas_mut() else {
                        return;
                    };
                    if let Some(size) = pending_resize.take() {
                        canvas.resize(size);
                    }
                    let window_size = window.inner_size();
                    if !desktop_surface_size_is_renderable(window_size) {
                        canvas.suspend_for_zero_size(window_size);
                        surface_timeout_backoff.reset();
                        surface_timeout_redraw_at = None;
                        return;
                    }
                    let smooth_scroll_lines = app.single_session_smooth_scroll_lines(
                        scroll_accumulator.pending_lines(),
                        window_size,
                        &mut scroll_metrics_cache,
                    );
                    let render_result = canvas.render(
                        &app,
                        window.current_monitor().map(|monitor| monitor.size()),
                        smooth_scroll_lines,
                        workspace_space_hold_progress(
                            &app,
                            space_hold_started_at,
                            space_hold_consumed,
                        ),
                    );
                    match render_result {
                    Ok(frame) => {
                        surface_timeout_backoff.reset();
                        surface_timeout_redraw_at = None;
                        no_paint_watchdog.observe_presented(Instant::now(), &frame);
                        interaction_latency.observe_presented(&frame);
                        if !first_frame_presented {
                            first_frame_presented = true;
                            startup_trace.mark("first frame presented");
                            if startup_benchmark {
                                target.exit();
                                return;
                            }
                            if recovery_scan_pending {
                                recovery_scan_pending = false;
                                spawn_recovery_session_count_scan(
                                    event_loop_proxy.clone(),
                                    startup_trace,
                                );
                            }
                        }
                        if frame.content_ready && !first_content_frame_presented {
                            first_content_frame_presented = true;
                            startup_trace.mark("first content frame presented");
                        }
                        if startup_content_benchmark && frame.content_ready {
                            target.exit();
                            return;
                        }
                        // Pace continuous animations instead of immediately
                        // re-requesting a redraw. An immediate request makes the
                        // event loop render as fast as the CPU allows (the surface
                        // presents without blocking), pinning the main thread near
                        // 100% CPU and starving input/compositor scheduling. The
                        // scheduled wake is serviced in AboutToWait.
                        animation_redraw_at =
                            next_animation_redraw_at(Instant::now(), frame.animation_active);
                    }
                    Err(SurfaceError::Lost | SurfaceError::Outdated) => {
                        surface_timeout_backoff.reset();
                        surface_timeout_redraw_at = None;
                        canvas.resize(window.inner_size());
                        window.request_redraw();
                    }
                    Err(SurfaceError::OutOfMemory) => target.exit(),
                    Err(SurfaceError::Timeout) => {
                        let now = Instant::now();
                        let (delay, consecutive_timeouts) = surface_timeout_backoff.record_timeout();
                        let redraw_at = now + delay;
                        surface_timeout_redraw_at = Some(redraw_at);
                        if consecutive_timeouts == 1 || delay == SURFACE_TIMEOUT_BACKOFF_MAX {
                            desktop_log::warn(format_args!(
                                "jcode-desktop: surface acquire timed out, retrying in {}ms after {} consecutive timeout(s)",
                                delay.as_millis(),
                                consecutive_timeouts
                            ));
                        }
                        target.set_control_flow(ControlFlow::WaitUntil(redraw_at));
                    }
                    }
                }
                _ => {}
            },
            Event::UserEvent(DesktopUserEvent::RecoveryCount(recovery_count)) => {
                if let DesktopApp::SingleSession(single_session) = &mut app {
                    single_session.set_recovery_session_count(recovery_count);
                    window.set_title(&app.status_title());
                    interaction_latency.mark("recovery_count", Instant::now());
                    window.request_redraw();
                }
            }
            Event::UserEvent(DesktopUserEvent::CanvasReady(result)) => {
                let DesktopCanvasInitResult { canvas, elapsed } = *result;
                match canvas {
                    Ok(mut ready_canvas) => {
                        startup_trace.mark(&format!(
                            "canvas ready (async {}ms)",
                            elapsed.as_millis()
                        ));
                        ready_canvas.resize(window.inner_size());
                        renderer = DesktopHostRendererState::GpuReady(Box::new(ready_canvas));
                        if let Some(handoff) = reload_startup_handoff.as_ref() {
                            handoff.signal_ready_and_wait_for_release();
                            window.set_visible(true);
                            startup_trace.mark("reload handoff released");
                        }
                        reload_startup_handoff = None;
                        window.request_redraw();
                    }
                    Err(message) => {
                        desktop_log::error(format_args!(
                            "jcode-desktop: failed to initialize desktop renderer: {message}"
                        ));
                        renderer = DesktopHostRendererState::GpuFailed { _message: message };
                        target.exit();
                    }
                }
            }
            Event::UserEvent(DesktopUserEvent::SessionCardsLoaded {
                purpose,
                cards,
                loaded_in,
            }) => {
                let card_count = cards.len();
                let mut applied = false;
                match purpose {
                    DesktopSessionCardsPurpose::WorkspaceInitialLoad => {
                        if let DesktopApp::Workspace(workspace) = &mut app {
                            workspace.replace_session_cards(cards);
                            if let Some(preferences) = pending_workspace_startup_preferences.take() {
                                workspace.apply_preferences(preferences);
                            }
                            applied = true;
                        }
                    }
                    DesktopSessionCardsPurpose::WorkspaceRefresh => {
                        if let DesktopApp::Workspace(workspace) = &mut app {
                            workspace.replace_session_cards(cards);
                            queue_desktop_preferences_save(workspace, &preferences_save_tx);
                            applied = true;
                        }
                    }
                    DesktopSessionCardsPurpose::SingleSessionSwitcher => {
                        if app.is_single_session() {
                            app.apply_single_session_switcher_cards(cards);
                            applied = true;
                        }
                    }
                }
                log_desktop_session_cards_load_profile(purpose, loaded_in, card_count, applied);
                if applied {
                    window.set_title(&app.status_title());
                    interaction_latency.mark("session_cards_load", Instant::now());
                    window.request_redraw();
                }
            }
            Event::UserEvent(DesktopUserEvent::SessionCardLoaded {
                session_id,
                card,
                loaded_in,
            }) => {
                let card_found = card.is_some();
                let mut applied = false;
                if let DesktopApp::SingleSession(single_session) = &mut app
                    && single_session.live_session_id.as_deref() == Some(session_id.as_str())
                    && let Some(card) = card
                {
                    single_session.replace_session(Some(card));
                    applied = true;
                }
                log_desktop_session_card_refresh_profile(
                    &session_id,
                    loaded_in,
                    card_found,
                    applied,
                );
                if applied {
                    window.set_title(&app.status_title());
                    interaction_latency.mark("session_card_refresh", Instant::now());
                    window.request_redraw();
                }
            }
            Event::UserEvent(DesktopUserEvent::CrashedSessionsRestoreFinished {
                restored,
                errors,
                elapsed,
            }) => {
                log_desktop_crashed_sessions_restore_profile(restored, errors.len(), elapsed);
                if restored == 0 {
                    let message = if errors.is_empty() {
                        "no crashed sessions found".to_string()
                    } else {
                        format!("failed to restore crashed sessions: {}", errors.join("; "))
                    };
                    apply_single_session_error(&mut app, anyhow::anyhow!(message));
                } else if let DesktopApp::SingleSession(single_session) = &mut app {
                    single_session.set_recovery_session_count(0);
                    single_session.set_status_label(format!("restored {restored} crashed session(s)"));
                }
                window.set_title(&app.status_title());
                interaction_latency.mark("restore_crashed_sessions", Instant::now());
                window.request_redraw();
            }
            Event::UserEvent(DesktopUserEvent::GitHubIssuesSyncFinished(result)) => {
                github_issue_sync_running = false;
                app.apply_github_issue_sync_result(result);
                window.set_title(&app.status_title());
                interaction_latency.mark("github_issue_sync", Instant::now());
                window.request_redraw();
            }
            Event::UserEvent(DesktopUserEvent::TranscriptHydrated {
                session_id,
                result,
                loaded_in,
            }) => {
                if app.apply_hydrated_transcript(&session_id, result) {
                    desktop_log::info(format_args!(
                        "jcode-desktop: hydrated resumed transcript for {session_id} in {}ms",
                        loaded_in.as_millis()
                    ));
                    window.set_title(&app.status_title());
                    interaction_latency.mark("transcript_hydration", Instant::now());
                    window.request_redraw();
                }
            }
            Event::UserEvent(DesktopUserEvent::SessionEvents(batch)) => {
                let ui_received_at = Instant::now();
                let accumulated_for = batch.accumulated_for();
                let raw_event_count = batch.raw_event_count;
                let raw_payload_bytes = batch.raw_payload_bytes;
                let forwarded_at = batch.forwarded_at;
                forward_desktop_session_event_batch_to_worker(&mut hot_reloader, &batch);
                let apply_stats = apply_desktop_session_event_batch_with_stats(&mut app, batch.events);
                let ui_queue_delay = ui_received_at.saturating_duration_since(forwarded_at);
                let mut redraw_requested = false;
                let mut redraw_deferred = false;
                let mut session_card_refresh_spawned = false;
                if apply_stats.visible_changed {
                    let now = Instant::now();
                    if apply_stats.session_card_refresh_requested
                        && let Some(session_id) = app.single_session_live_id()
                    {
                        spawn_single_session_card_refresh(session_id, event_loop_proxy.clone());
                        session_card_refresh_spawned = true;
                    }
                    if let Some((message, images)) = app.take_next_queued_single_session_draft() {
                        let result = if let Some(session_id) = app.single_session_live_id() {
                            session_launch::spawn_message_to_session(
                                session_id,
                                message,
                                images,
                                session_event_tx.clone(),
                            )
                        } else {
                            session_launch::spawn_fresh_server_session(
                                message,
                                images,
                                session_event_tx.clone(),
                            )
                        };
                        match result {
                            Ok(handle) => app.set_single_session_handle(handle),
                            Err(error) => apply_single_session_error(&mut app, error),
                        }
                    }
                    window.set_title(&app.status_title());
                    let redraw_due = last_backend_redraw_request.is_none_or(|last| {
                        now.saturating_duration_since(last) >= BACKEND_REDRAW_FRAME_INTERVAL
                    });
                    if redraw_due {
                        let first_pending = pending_backend_redraw_since.take().unwrap_or(now);
                        interaction_latency.mark("backend_events", first_pending);
                        last_backend_redraw_request = Some(now);
                        window.request_redraw();
                        redraw_requested = true;
                    } else {
                        pending_backend_redraw_since.get_or_insert(now);
                        redraw_deferred = true;
                    }
                }
                log_desktop_session_event_batch_profile(
                    raw_event_count,
                    raw_payload_bytes,
                    accumulated_for,
                    ui_queue_delay,
                    &apply_stats,
                    redraw_requested,
                    redraw_deferred,
                    session_card_refresh_spawned,
                );
            }
            Event::AboutToWait => {
                let surface_renderable = desktop_surface_size_is_renderable(window.inner_size());
                if let Some(redraw_at) = surface_timeout_redraw_at {
                    let now = Instant::now();
                    if now >= redraw_at {
                        surface_timeout_redraw_at = None;
                        if surface_renderable {
                            window.request_redraw();
                        }
                    }
                }
                // Service the paced animation redraw scheduled by RedrawRequested.
                // This keeps continuous animations advancing at ~display refresh
                // without busy-spinning the loop between frames.
                if let Some(redraw_at) = animation_redraw_at {
                    let now = Instant::now();
                    if now >= redraw_at {
                        animation_redraw_at = None;
                        if surface_renderable {
                            window.request_redraw();
                        }
                    }
                }
                if surface_renderable && app.is_single_session() {
                    let about_to_wait_started = Instant::now();
                    let size = window.inner_size();
                    let previous_smooth_scroll = app.single_session_smooth_scroll_lines(
                        scroll_accumulator.pending_lines(),
                        size,
                        &mut scroll_metrics_cache,
                    );
                    let frame = scroll_accumulator.frame(Instant::now());
                    if let Some(lines) = frame.scroll_lines
                        && !app.scroll_single_session_body(lines, size, &mut scroll_metrics_cache)
                    {
                        scroll_accumulator.stop();
                    }
                    let next_smooth_scroll = app.single_session_smooth_scroll_lines(
                        scroll_accumulator.pending_lines(),
                        size,
                        &mut scroll_metrics_cache,
                    );
                    if frame.active
                        || (next_smooth_scroll - previous_smooth_scroll).abs()
                            >= SCROLL_FRACTIONAL_EPSILON
                    {
                        interaction_latency.mark("scroll_momentum", about_to_wait_started);
                        window.request_redraw();
                    }
                } else if scroll_accumulator.is_active() {
                    scroll_accumulator.reset();
                    scroll_metrics_cache.clear();
                }
                if let (DesktopApp::Workspace(workspace), Some(started_at)) = (&mut app, space_hold_started_at)
                    && !space_hold_consumed
                {
                    let now = Instant::now();
                    if now.saturating_duration_since(started_at) >= workspace.space_hold_toggle_duration() {
                        space_hold_consumed = true;
                        if matches!(workspace.handle_key(KeyInput::ToggleInputMode), KeyOutcome::Redraw) {
                            window.set_title(&app.status_title());
                        }
                    }
                    if surface_renderable {
                        window.request_redraw();
                    }
                }
                if let Some(first_pending_backend_redraw) = pending_backend_redraw_since {
                    let now = Instant::now();
                    if surface_renderable
                        && last_backend_redraw_request.is_none_or(|last| {
                            now.saturating_duration_since(last) >= BACKEND_REDRAW_FRAME_INTERVAL
                        })
                    {
                        pending_backend_redraw_since = None;
                        interaction_latency.mark("backend_events", first_pending_backend_redraw);
                        last_backend_redraw_request = Some(now);
                        window.request_redraw();
                    }
                }
                if hot_reloader.poll(&app, &window) {
                    target.exit();
                    return;
                }

                if let Some(canvas) = renderer.canvas_mut()
                    && surface_renderable
                    && canvas.needs_initial_frame
                {
                    canvas.needs_initial_frame = false;
                    window.request_redraw();
                } else if surface_renderable
                    && app.has_frame_animation()
                    && animation_redraw_at.is_none()
                {
                    // An animation is active but no paced redraw is scheduled yet
                    // (e.g. it just became active). Schedule one instead of
                    // requesting a redraw on every loop iteration, which would
                    // busy-spin the main thread at 100% CPU.
                    animation_redraw_at = next_animation_redraw_at(Instant::now(), true);
                }
            }
            _ => {}
        }
    })?;

    Ok(())
}

fn load_session_cards_for_desktop() -> Vec<workspace::SessionCard> {
    match session_data::load_recent_session_cards() {
        Ok(cards) => cards,
        Err(error) => {
            desktop_log::error(format_args!(
                "jcode-desktop: failed to load session metadata: {error:#}"
            ));
            Vec::new()
        }
    }
}

fn load_crashed_session_cards_for_desktop() -> Vec<workspace::SessionCard> {
    match session_data::load_crashed_session_cards() {
        Ok(cards) => cards,
        Err(error) => {
            desktop_log::error(format_args!(
                "jcode-desktop: failed to load crashed session metadata: {error:#}"
            ));
            Vec::new()
        }
    }
}

fn spawn_recovery_session_count_scan(
    event_loop_proxy: EventLoopProxy<DesktopUserEvent>,
    startup_trace: DesktopStartupTrace,
) {
    if let Err(error) = spawn_bounded_desktop_async_job("jcode-desktop-recovery-scan", move || {
        startup_trace.mark("recovery scan started");
        let recovery_count = load_crashed_session_cards_for_desktop().len();
        startup_trace.mark(&format!(
            "recovery scan completed ({recovery_count} crashed)"
        ));
        if event_loop_proxy
            .send_event(DesktopUserEvent::RecoveryCount(recovery_count))
            .is_err()
        {
            desktop_log::warn(format_args!(
                "jcode-desktop: failed to deliver recovery count, event loop is closed"
            ));
        }
    }) {
        desktop_log::error(format_args!(
            "jcode-desktop: failed to start recovery scan: {error:#}"
        ));
    }
}

fn spawn_single_session_card_refresh(
    session_id: String,
    event_loop_proxy: EventLoopProxy<DesktopUserEvent>,
) {
    if let Err(error) =
        spawn_bounded_desktop_async_job("jcode-desktop-session-card-refresh", move || {
            let started = Instant::now();
            let card = load_session_cards_for_desktop()
                .into_iter()
                .find(|card| card.session_id == session_id);
            let loaded_in = started.elapsed();
            if event_loop_proxy
                .send_event(DesktopUserEvent::SessionCardLoaded {
                    session_id,
                    card,
                    loaded_in,
                })
                .is_err()
            {
                desktop_log::warn(format_args!(
                    "jcode-desktop: failed to deliver session card refresh, event loop is closed"
                ));
            }
        })
    {
        desktop_log::error(format_args!(
            "jcode-desktop: failed to start session card refresh: {error:#}"
        ));
    }
}

fn spawn_session_cards_load(
    purpose: DesktopSessionCardsPurpose,
    event_loop_proxy: EventLoopProxy<DesktopUserEvent>,
    delay: Duration,
) {
    if let Err(error) = spawn_bounded_desktop_async_job(
        format!("jcode-desktop-session-cards-{purpose:?}"),
        move || {
            if !delay.is_zero() {
                std::thread::sleep(delay);
            }
            let started = Instant::now();
            let cards = load_session_cards_for_desktop();
            let loaded_in = started.elapsed();
            if event_loop_proxy
                .send_event(DesktopUserEvent::SessionCardsLoaded {
                    purpose,
                    cards,
                    loaded_in,
                })
                .is_err()
            {
                desktop_log::warn(format_args!(
                    "jcode-desktop: failed to deliver session cards load, event loop is closed"
                ));
            }
        },
    ) {
        desktop_log::error(format_args!(
            "jcode-desktop: failed to start session card load: {error:#}"
        ));
    }
}

fn spawn_restore_crashed_sessions(event_loop_proxy: EventLoopProxy<DesktopUserEvent>) {
    if let Err(error) = spawn_bounded_desktop_async_job(
        "jcode-desktop-restore-crashed-sessions",
        move || {
            let started = Instant::now();
            let crashed = load_crashed_session_cards_for_desktop();
            let mut restored = 0usize;
            let mut errors = Vec::new();
            for card in crashed {
                match session_launch::launch_validated_resume_session(&card.session_id, &card.title)
                {
                    Ok(()) => restored += 1,
                    Err(error) => errors.push(format!("{}: {error:#}", card.session_id)),
                }
            }
            if event_loop_proxy
                .send_event(DesktopUserEvent::CrashedSessionsRestoreFinished {
                    restored,
                    errors,
                    elapsed: started.elapsed(),
                })
                .is_err()
            {
                desktop_log::warn(format_args!(
                    "jcode-desktop: failed to deliver crashed-session restore result, event loop is closed"
                ));
            }
        },
    ) {
        desktop_log::error(format_args!(
            "jcode-desktop: failed to start crashed-session restore: {error:#}"
        ));
    }
}

fn spawn_github_issue_sync(event_loop_proxy: EventLoopProxy<DesktopUserEvent>) -> Result<()> {
    spawn_bounded_desktop_async_job("jcode-desktop-github-issues-sync", move || {
        let result = desktop_issue_cache::sync_current_repo_issue_cache()
            .map_err(|error| format!("{error:#}"));
        match &result {
            Ok(summary) => desktop_log::info(format_args!(
                "jcode-desktop: synced {} GitHub issue(s) for {} in {}ms to {} (comment_threads={} comment_errors={})",
                summary.issue_count,
                summary.repo,
                summary.elapsed.as_millis(),
                summary.cache_path.display(),
                summary.fetched_comment_threads,
                summary.comment_fetch_errors
            )),
            Err(error) => desktop_log::warn(format_args!(
                "jcode-desktop: GitHub issue sync failed: {error}"
            )),
        }
        if event_loop_proxy
            .send_event(DesktopUserEvent::GitHubIssuesSyncFinished(result))
            .is_err()
        {
            desktop_log::warn(format_args!(
                "jcode-desktop: failed to deliver GitHub issue sync result"
            ));
        }
    })
}

fn start_pending_github_issue_sync(
    app: &mut DesktopApp,
    sync_running: &mut bool,
    event_loop_proxy: EventLoopProxy<DesktopUserEvent>,
) -> bool {
    if !app.take_github_issue_sync_request() {
        return false;
    }
    if *sync_running {
        app.note_github_issue_sync_already_running();
        return true;
    }
    match spawn_github_issue_sync(event_loop_proxy) {
        Ok(()) => {
            *sync_running = true;
            true
        }
        Err(error) => {
            app.apply_github_issue_sync_result(Err(format!("{error:#}")));
            true
        }
    }
}

/// Start an off-thread transcript load for a session resumed from the
/// switcher (or a promoted workspace card). The result is delivered back to
/// the event loop as `DesktopUserEvent::TranscriptHydrated`, so large
/// transcript parses never stall key handling. Falls back to a synchronous
/// load if the job slot or thread spawn fails.
fn start_pending_transcript_hydration(
    app: &mut DesktopApp,
    event_loop_proxy: EventLoopProxy<DesktopUserEvent>,
) -> bool {
    let Some(session_id) = app.take_pending_transcript_hydration() else {
        return false;
    };
    let job_session_id = session_id.clone();
    let spawned =
        spawn_bounded_desktop_async_job("jcode-desktop-transcript-hydration", move || {
            let started = Instant::now();
            let result = session_data::load_session_transcript_by_id(&job_session_id)
                .map_err(|error| format!("{error:#}"));
            if event_loop_proxy
                .send_event(DesktopUserEvent::TranscriptHydrated {
                    session_id: job_session_id,
                    result,
                    loaded_in: started.elapsed(),
                })
                .is_err()
            {
                desktop_log::warn(format_args!(
                    "jcode-desktop: failed to deliver hydrated transcript"
                ));
            }
        });
    if let Err(error) = spawned {
        desktop_log::warn(format_args!(
            "jcode-desktop: transcript hydration fell back to blocking load: {error:#}"
        ));
        let result = session_data::load_session_transcript_by_id(&session_id)
            .map_err(|error| format!("{error:#}"));
        app.apply_hydrated_transcript(&session_id, result);
    }
    true
}

fn spawn_desktop_preferences_saver() -> Option<mpsc::Sender<workspace::DesktopPreferences>> {
    let (tx, rx) = mpsc::channel::<workspace::DesktopPreferences>();
    match std::thread::Builder::new()
        .name("jcode-desktop-preferences-saver".to_string())
        .spawn(move || {
            while let Ok(mut preferences) = rx.recv() {
                let received_at = Instant::now();
                let mut coalesced_saves = 1usize;
                while let Ok(next_preferences) = rx.try_recv() {
                    preferences = next_preferences;
                    coalesced_saves += 1;
                }
                save_desktop_preferences_off_ui_thread(
                    preferences,
                    coalesced_saves,
                    received_at.elapsed(),
                );
            }
        }) {
        Ok(_) => Some(tx),
        Err(error) => {
            desktop_log::error(format_args!(
                "jcode-desktop: failed to start preferences saver: {error:#}"
            ));
            None
        }
    }
}

fn queue_desktop_preferences_save(
    workspace: &Workspace,
    preferences_save_tx: &Option<mpsc::Sender<workspace::DesktopPreferences>>,
) {
    let preferences = workspace.preferences();
    if let Some(tx) = preferences_save_tx
        && tx.send(preferences.clone()).is_ok()
    {
        return;
    }

    if let Err(error) =
        spawn_bounded_desktop_async_job("jcode-desktop-preferences-save-once", move || {
            save_desktop_preferences_off_ui_thread(preferences, 1, Duration::ZERO);
        })
    {
        desktop_log::error(format_args!(
            "jcode-desktop: failed to queue preferences save: {error:#}"
        ));
    }
}

fn save_desktop_preferences_off_ui_thread(
    preferences: workspace::DesktopPreferences,
    coalesced_saves: usize,
    queued_for: Duration,
) {
    let started = Instant::now();
    let error = desktop_prefs::save_preferences(&preferences)
        .err()
        .map(|error| format!("{error:#}"));
    log_desktop_preferences_save_profile(
        started.elapsed(),
        queued_for,
        coalesced_saves,
        error.as_deref(),
    );
}

fn headless_chat_smoke_message(args: &[String]) -> Option<String> {
    args.iter().enumerate().find_map(|(index, arg)| {
        arg.strip_prefix("--headless-chat-smoke=")
            .map(ToOwned::to_owned)
            .or_else(|| {
                (arg == "--headless-chat-smoke")
                    .then(|| args.get(index + 1).cloned())
                    .flatten()
            })
    })
}

/// Dev-only flag: `--simulate-stream` drives the live single-session app with
/// synthetic streaming deltas so the streaming reveal animation can be observed
/// and recorded without a real backend.
fn simulate_stream_requested(args: &[String]) -> bool {
    args.iter()
        .any(|arg| arg == "--simulate-stream" || arg == "--simulate-streaming")
}

const DESKTOP_STREAM_SIMULATOR_SCRIPT: &str = "Sure, let me walk through how the streaming text reveal works in the desktop app. \
When the provider sends tokens, they arrive in bursty chunks rather than a smooth flow, \
so the renderer keeps a `revealed_chars` cursor that eases toward the full response length. \
The trailing characters get a per-character alpha ramp called the tail fade, \
and a soft breathing cursor sits at the very end of the revealed text to signal activity.\n\n\
Here is a short list of the moving parts:\n\
- The reveal motion integrates a rate proportional to the backlog.\n\
- The body text buffer is rebuilt as the reveal advances.\n\
- A separate overlay buffer paints the streaming tail with its own opacity.\n\n\
Once the response finishes, the overlay hands off to the committed transcript message. \
That handoff should be seamless, with no visible jump or flicker as the text settles into place. \
This paragraph is intentionally long so the streaming text wraps across many lines and the \
viewport scrolls while new tokens keep arriving at the bottom of the transcript.";

/// Seed a small prior transcript so the simulated stream appends after existing
/// messages, mirroring the common case of streaming inside an active session.
fn seed_desktop_stream_simulator_transcript(app: &mut SingleSessionApp) {
    app.replace_session(Some(workspace::SessionCard {
        session_id: "simulate-stream".to_string(),
        title: "Streaming simulation".to_string(),
        subtitle: "dev stream harness".to_string(),
        detail: "fixture".to_string(),
        preview_lines: Vec::new(),
        detail_lines: Vec::new(),
        transcript_messages: Vec::new(),
    }));
    app.messages.push(SingleSessionMessage::user(
        "Explain how the desktop streaming text reveal works.",
    ));
    app.messages.push(SingleSessionMessage::assistant(
        "Earlier reply: the desktop renders streamed assistant text with an adaptive reveal so bursty provider chunks flow in smoothly instead of popping.",
    ));
    app.scroll_body_to_bottom();
}

/// Spawn a background thread that emits synthetic streaming events to exercise
/// the real desktop streaming animation pipeline.
fn spawn_desktop_stream_simulator(
    session_event_tx: mpsc::Sender<session_launch::DesktopSessionEvent>,
) {
    std::thread::Builder::new()
        .name("jcode-desktop-stream-simulator".to_string())
        .spawn(move || {
            // Give the window a moment to come up before streaming starts.
            std::thread::sleep(Duration::from_millis(900));
            if session_event_tx
                .send(session_launch::DesktopSessionEvent::SessionStarted {
                    session_id: "simulate-stream".to_string(),
                })
                .is_err()
            {
                return;
            }
            // Emit word-sized deltas, occasionally bursting several words at once
            // to mimic real provider chunking, with brief stalls between bursts.
            let words: Vec<&str> = DESKTOP_STREAM_SIMULATOR_SCRIPT
                .split_inclusive(' ')
                .collect();
            let mut index = 0usize;
            let mut burst_phase = 0usize;
            while index < words.len() {
                let burst = match burst_phase % 4 {
                    0 => 1,
                    1 => 3,
                    2 => 2,
                    _ => 5,
                };
                burst_phase += 1;
                let end = (index + burst).min(words.len());
                let chunk: String = words[index..end].concat();
                index = end;
                if session_event_tx
                    .send(session_launch::DesktopSessionEvent::TextDelta(chunk))
                    .is_err()
                {
                    return;
                }
                let pause = match burst_phase % 5 {
                    0 => Duration::from_millis(220),
                    3 => Duration::from_millis(120),
                    _ => Duration::from_millis(45),
                };
                std::thread::sleep(pause);
            }
            std::thread::sleep(Duration::from_millis(400));
            let _ = session_event_tx.send(session_launch::DesktopSessionEvent::Done);
        })
        .ok();
}

const DESKTOP_HELP_LINES: &[&str] = &[
    "Jcode Desktop",
    "",
    "Usage:",
    "  jcode-desktop [OPTIONS]",
    "",
    "Options:",
    "  --fullscreen                 Start borderless fullscreen",
    "  --workspace                  Open the workspace prototype instead of the single-session chat",
    "  --desktop-process-role ROLE  Internal: standalone, host, or worker",
    "  --desktop-host               Internal alias for --desktop-process-role=host",
    "  --desktop-app-worker         Internal alias for --desktop-process-role=worker",
    "  --startup-log                Print launch timing milestones to stderr",
    "  --startup-benchmark          Print launch timings and exit after the first frame",
    "  --capture-hero-animation DIR Write deterministic hero animation PNG frames and exit",
    "  --capture-gallery-screens DIR Render gallery fixture states to PNGs headlessly and exit",
    "  --capture-keys KEYS          With --capture-gallery-screens: comma-separated keys to replay first",
    "  --capture-size WxH           With --capture-gallery-screens: render size in pixels",
    "  --resize-render-benchmark[N]  Print CPU resize/render benchmark JSON and exit",
    "  --scroll-render-benchmark[N]  Print CPU scroll/render benchmark JSON and exit",
    "  --real-transcript-scroll-benchmark[N]  Profile scrolling against your real on-disk transcripts and exit",
    "  --real-transcript-action-benchmark[N]  Profile mixed user actions (scroll/resize/typing/pickers/selection/streaming) on real transcripts and exit",
    "  --stream-e2e-benchmark[N]     Print stream event-to-paint guardrail JSON and exit",
    "  --headless-chat-smoke <MSG>  Run a hidden backend smoke test and print JSON events",
    "  --headless-chat-smoke=<MSG>  Same as above",
    "  -V, --version                Print version information",
    "  -h, --help                   Print this help",
    "",
];

fn desktop_help_text() -> String {
    DESKTOP_HELP_LINES.join("\n")
}


/// Request for a headless gallery screenshot capture.
///
/// `--capture-gallery-screens DIR` renders every gallery fixture state to a
/// PNG in DIR without opening a window. `--gallery-state STATE` (optional)
/// restricts the capture to a single state, and `--capture-keys KEYSPEC`
/// (optional) replays comma-separated key names against each state before
/// rendering, so arbitrary interaction states can be inspected visually.
struct GalleryScreenshotCaptureRequest {
    output_dir: PathBuf,
    state: Option<String>,
    keys: Vec<String>,
    size: Option<PhysicalSize<u32>>,
}

fn gallery_screenshot_capture_request(args: &[String]) -> Option<GalleryScreenshotCaptureRequest> {
    let output_dir = args.iter().enumerate().find_map(|(index, arg)| {
        arg.strip_prefix("--capture-gallery-screens=")
            .map(PathBuf::from)
            .or_else(|| {
                (arg == "--capture-gallery-screens")
                    .then(|| args.get(index + 1).map(PathBuf::from))
                    .flatten()
            })
    })?;
    let keys = args
        .iter()
        .enumerate()
        .find_map(|(index, arg)| {
            arg.strip_prefix("--capture-keys=")
                .map(str::to_string)
                .or_else(|| {
                    (arg == "--capture-keys")
                        .then(|| args.get(index + 1).cloned())
                        .flatten()
                })
        })
        .map(|spec| {
            spec.split(',')
                .map(str::trim)
                .filter(|part| !part.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let size = args
        .iter()
        .enumerate()
        .find_map(|(index, arg)| {
            arg.strip_prefix("--capture-size=")
                .map(str::to_string)
                .or_else(|| {
                    (arg == "--capture-size")
                        .then(|| args.get(index + 1).cloned())
                        .flatten()
                })
        })
        .and_then(|spec| {
            let (width, height) = spec.split_once('x')?;
            Some(PhysicalSize::new(
                width.trim().parse().ok()?,
                height.trim().parse().ok()?,
            ))
        });
    Some(GalleryScreenshotCaptureRequest {
        output_dir,
        state: desktop_gallery::state_from_args(args),
        keys,
        size,
    })
}

/// Parse a key name from `--capture-keys` into a `KeyInput`.
fn capture_key_input(name: &str) -> Option<KeyInput> {
    Some(match name {
        "escape" => KeyInput::Escape,
        "enter" => KeyInput::Enter,
        "backspace" => KeyInput::Backspace,
        "tab" => KeyInput::Autocomplete,
        "submit" => KeyInput::SubmitDraft,
        "model-picker" => KeyInput::OpenModelPicker,
        "session-switcher" => KeyInput::OpenSessionSwitcher,
        "hotkey-help" => KeyInput::HotkeyHelp,
        "session-info" => KeyInput::ToggleSessionInfo,
        "scroll-up" => KeyInput::ScrollBodyLines(-3),
        "scroll-down" => KeyInput::ScrollBodyLines(3),
        "scroll-top" => KeyInput::ScrollBodyToTop,
        "scroll-bottom" => KeyInput::ScrollBodyToBottom,
        "page-up" => KeyInput::ScrollBodyPages(-1),
        "page-down" => KeyInput::ScrollBodyPages(1),
        "text-bigger" => KeyInput::AdjustTextScale(1),
        "text-smaller" => KeyInput::AdjustTextScale(-1),
        other => {
            let text = other.strip_prefix("char:")?;
            KeyInput::Character(text.to_string())
        }
    })
}

async fn run_gallery_screenshot_capture(request: &GalleryScreenshotCaptureRequest) -> Result<()> {
    std::fs::create_dir_all(&request.output_dir).with_context(|| {
        format!(
            "failed to create gallery screenshot directory {}",
            request.output_dir.display()
        )
    })?;
    let states: Vec<String> = match &request.state {
        Some(state) => vec![state.clone()],
        None => desktop_gallery::gallery_states()
            .iter()
            .map(|state| state.to_string())
            .collect(),
    };
    let keys = request
        .keys
        .iter()
        .map(|name| {
            capture_key_input(name).with_context(|| format!("unknown capture key name {name:?}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let size = request.size.unwrap_or_else(|| {
        PhysicalSize::new(DEFAULT_WINDOW_WIDTH as u32, DEFAULT_WINDOW_HEIGHT as u32)
    });
    let mut manifest = Vec::new();
    for state in &states {
        let mut app = desktop_gallery::temporary_app(state);
        for key in &keys {
            app.handle_key(key.clone());
        }
        let DesktopApp::SingleSession(single) = &mut app else {
            anyhow::bail!("gallery screenshot capture only supports single-session states");
        };
        single.settle_animations_for_capture();
        let single = &*single;
        let rendered_lines = single_session_rendered_body_lines_for_tick(single, size, 4);
        let widget_geometry =
            inline_widget_capture_geometry(single, size, rendered_lines.len()).map(
                |(card, text_top, line_height, visible_text_bottom, visible_text_right)| {
                    serde_json::json!({
                        "card": { "x": card.x, "y": card.y, "width": card.width, "height": card.height },
                        "text_top": text_top,
                        "line_height": line_height,
                        "visible_text_bottom": visible_text_bottom,
                        "visible_text_right": visible_text_right,
                    })
                },
            );
        let (image, vertices) = render_hero_frame_to_image(single, size, 4, 1.0, false).await?;
        let filename = if request.keys.is_empty() {
            format!("gallery-{state}.png")
        } else {
            let key_part = request
                .keys
                .join("+")
                .chars()
                .map(|ch| {
                    if ch.is_ascii_alphanumeric() || matches!(ch, '+' | '-' | '_' | ':') {
                        ch
                    } else {
                        '_'
                    }
                })
                .collect::<String>();
            format!("gallery-{state}+{key_part}.png")
        };
        let path = request.output_dir.join(&filename);
        image
            .save(&path)
            .with_context(|| format!("failed to save {}", path.display()))?;
        manifest.push(serde_json::json!({
            "state": state,
            "file": filename,
            "keys": request.keys,
            "vertices": vertices,
            "inline_widget": widget_geometry,
            "snapshot": serde_json::to_value(app.snapshot())?,
        }));
    }
    println!(
        "{}",
        serde_json::json!({
            "output_dir": request.output_dir,
            "screens": manifest,
        })
    );
    Ok(())
}


enum DesktopUserEvent {
    CanvasReady(Box<DesktopCanvasInitResult>),
    SessionEvents(DesktopSessionEventBatch),
    SessionCardsLoaded {
        purpose: DesktopSessionCardsPurpose,
        cards: Vec<workspace::SessionCard>,
        loaded_in: Duration,
    },
    SessionCardLoaded {
        session_id: String,
        card: Option<workspace::SessionCard>,
        loaded_in: Duration,
    },
    CrashedSessionsRestoreFinished {
        restored: usize,
        errors: Vec<String>,
        elapsed: Duration,
    },
    GitHubIssuesSyncFinished(
        std::result::Result<desktop_issue_cache::GitHubIssueSyncSummary, String>,
    ),
    TranscriptHydrated {
        session_id: String,
        result: std::result::Result<Option<Vec<workspace::SessionTranscriptMessage>>, String>,
        loaded_in: Duration,
    },
    RecoveryCount(usize),
}

struct DesktopCanvasInitResult {
    canvas: std::result::Result<Canvas, String>,
    elapsed: Duration,
}

enum DesktopHostRendererState {
    NoGpuBoot,
    GpuInitializing { _started_at: Instant },
    GpuReady(Box<Canvas>),
    GpuFailed { _message: String },
}

impl DesktopHostRendererState {
    fn start_gpu_init(
        &mut self,
        window: Arc<Window>,
        event_loop_proxy: EventLoopProxy<DesktopUserEvent>,
        startup_trace: DesktopStartupTrace,
    ) -> Result<()> {
        if matches!(self, Self::GpuInitializing { .. } | Self::GpuReady(_)) {
            return Ok(());
        }

        let started_at = Instant::now();
        std::thread::Builder::new()
            .name("jcode-desktop-gpu-init".to_string())
            .spawn(move || {
                startup_trace.mark("canvas init started");
                let canvas = pollster::block_on(Canvas::new(window, startup_trace))
                    .map_err(|error| format!("{error:#}"));
                let result = DesktopCanvasInitResult {
                    canvas,
                    elapsed: started_at.elapsed(),
                };
                if event_loop_proxy
                    .send_event(DesktopUserEvent::CanvasReady(Box::new(result)))
                    .is_err()
                {
                    desktop_log::warn(format_args!(
                        "jcode-desktop: failed to deliver async canvas initialization result"
                    ));
                }
            })
            .context("failed to spawn desktop GPU initialization thread")?;
        *self = Self::GpuInitializing {
            _started_at: started_at,
        };
        Ok(())
    }

    fn is_gpu_ready(&self) -> bool {
        matches!(self, Self::GpuReady(_))
    }

    fn canvas_mut(&mut self) -> Option<&mut Canvas> {
        match self {
            Self::GpuReady(canvas) => Some(canvas.as_mut()),
            Self::NoGpuBoot | Self::GpuInitializing { .. } | Self::GpuFailed { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DesktopSessionCardsPurpose {
    WorkspaceInitialLoad,
    WorkspaceRefresh,
    SingleSessionSwitcher,
}



fn create_desktop_font_system() -> FontSystem {
    let mut font_system = FontSystem::new();
    font_system
        .db_mut()
        .load_font_data(include_bytes!("../assets/fonts/Kalam-Regular.ttf").to_vec());
    font_system
        .db_mut()
        .load_font_data(include_bytes!("../assets/fonts/ShadowsIntoLightTwo-Regular.ttf").to_vec());
    font_system
        .db_mut()
        .load_font_data(include_bytes!("../assets/fonts/HomemadeApple-Regular.ttf").to_vec());
    font_system
        .db_mut()
        .load_font_data(include_bytes!("../assets/fonts/PatrickHand-Regular.ttf").to_vec());
    font_system
        .db_mut()
        .load_font_data(include_bytes!("../assets/fonts/Gaegu-Regular.ttf").to_vec());
    font_system
        .db_mut()
        .load_font_data(include_bytes!("../assets/fonts/Caveat-Regular.ttf").to_vec());
    font_system
        .db_mut()
        .load_font_data(include_bytes!("../assets/fonts/IndieFlower-Regular.ttf").to_vec());
    font_system
        .db_mut()
        .load_font_data(include_bytes!("../assets/fonts/GloriaHallelujah-Regular.ttf").to_vec());
    font_system
        .db_mut()
        .load_font_data(include_bytes!("../assets/fonts/Handlee-Regular.ttf").to_vec());
    font_system
        .db_mut()
        .load_font_data(include_bytes!("../assets/fonts/ReenieBeanie-Regular.ttf").to_vec());
    font_system
}

fn spawn_desktop_font_system_loader() -> JoinHandle<FontSystem> {
    std::thread::spawn(create_desktop_font_system)
}

#[cfg(target_os = "linux")]
fn desktop_wgpu_startup_backends() -> Vec<wgpu::Backends> {
    vec![wgpu::Backends::PRIMARY]
}

#[cfg(not(target_os = "linux"))]
fn desktop_wgpu_startup_backends() -> Vec<wgpu::Backends> {
    vec![wgpu::Backends::PRIMARY]
}

async fn request_startup_adapter(
    window: Arc<Window>,
    backend_candidates: Vec<wgpu::Backends>,
    startup_trace: DesktopStartupTrace,
) -> Result<(wgpu::Surface<'static>, wgpu::Adapter)> {
    let mut last_error = None;
    for backends in backend_candidates {
        let backend_label = format!("{backends:?}");
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::empty().with_env(),
            ..Default::default()
        });
        startup_trace.mark(&format!("wgpu instance created ({backend_label})"));
        let surface = match instance.create_surface(window.clone()) {
            Ok(surface) => surface,
            Err(error) => {
                last_error = Some(format!(
                    "{backend_label}: failed to create surface: {error:#}"
                ));
                continue;
            }
        };
        startup_trace.mark(&format!("wgpu surface created ({backend_label})"));
        if let Some(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
        {
            startup_trace.mark(&format!("wgpu adapter selected ({backend_label})"));
            return Ok((surface, adapter));
        }
        last_error = Some(format!("{backend_label}: no compatible adapter"));
    }

    match last_error {
        Some(error) => anyhow::bail!("failed to find a compatible GPU adapter ({error})"),
        None => anyhow::bail!("failed to find a compatible GPU adapter"),
    }
}


fn load_desktop_preferences() -> Option<workspace::DesktopPreferences> {
    match desktop_prefs::load_preferences() {
        Ok(preferences) => preferences,
        Err(error) => {
            desktop_log::error(format_args!(
                "jcode-desktop: failed to load desktop preferences: {error:#}"
            ));
            None
        }
    }
}

fn fresh_single_session_app() -> DesktopApp {
    DesktopApp::SingleSession(SingleSessionApp::new(None))
}

fn fresh_desktop_app_for_worker_mode(mode: DesktopWorkerMode) -> DesktopApp {
    match mode {
        DesktopWorkerMode::SingleSession => fresh_single_session_app(),
        DesktopWorkerMode::Workspace => DesktopApp::Workspace(Workspace::loading_sessions()),
    }
}

fn initial_single_session_app(resume_session_id: Option<&str>) -> DesktopApp {
    let Some(session_id) = resume_session_id else {
        return fresh_single_session_app();
    };

    let mut app = SingleSessionApp::new(None);
    app.initialize_resumed_session(session_id);
    match session_data::load_session_card_by_id(session_id) {
        Ok(Some(card)) => {
            app.replace_session(Some(card));
            app.hydrate_resumed_session_from_disk(session_id);
        }
        Ok(None) => {
            app.set_status_label(format!("resumed session {session_id}"));
            app.hydrate_resumed_session_from_disk(session_id);
        }
        Err(error) => {
            desktop_log::error(format_args!(
                "jcode-desktop: failed to load resumed session metadata for {session_id}: {error:#}"
            ));
            app.set_status_label(format!("resumed session {session_id}"));
            app.error = Some(format!("failed to load session metadata: {error:#}"));
        }
    }
    DesktopApp::SingleSession(app)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DesktopMode {
    SingleSession,
    WorkspacePrototype,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DesktopProcessRole {
    Standalone,
    StableHost,
    AppWorker,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DesktopReloadStrategy {
    FullProcessHandoff,
    AppWorkerRestart,
}

impl DesktopProcessRole {
    fn as_str(self) -> &'static str {
        match self {
            Self::Standalone => "standalone",
            Self::StableHost => "stable_host",
            Self::AppWorker => "app_worker",
        }
    }

    fn reload_strategy(self) -> DesktopReloadStrategy {
        match self {
            Self::Standalone | Self::AppWorker => DesktopReloadStrategy::FullProcessHandoff,
            Self::StableHost => DesktopReloadStrategy::AppWorkerRestart,
        }
    }
}

impl DesktopMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::SingleSession => "single_session",
            Self::WorkspacePrototype => "workspace",
        }
    }

    fn worker_mode(self) -> DesktopWorkerMode {
        match self {
            Self::SingleSession => DesktopWorkerMode::SingleSession,
            Self::WorkspacePrototype => DesktopWorkerMode::Workspace,
        }
    }
}

fn run_desktop_app_worker_process(desktop_mode: DesktopMode) -> Result<()> {
    desktop_log::info(format_args!(
        "jcode-desktop: app worker process started; pid={}",
        std::process::id()
    ));

    let mut stdout = std::io::stdout().lock();
    let ready = DesktopProtocolEnvelope::new(
        1,
        DesktopWorkerToHostMessage::Ready(DesktopWorkerReady {
            worker_pid: std::process::id(),
            mode: desktop_mode.worker_mode(),
        }),
    );
    write_desktop_ipc_frame(&mut stdout, &ready).context("failed to write worker ready frame")?;

    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());
    let mut runtime: Option<DesktopAppRuntime<DesktopApp>> = None;
    let mut latest_window = DesktopWindowState {
        width: DEFAULT_WINDOW_WIDTH as u32,
        height: DEFAULT_WINDOW_HEIGHT as u32,
        scale_factor: 1.0,
        focused: true,
    };
    let mut next_worker_sequence = 2;
    loop {
        let frame: Option<DesktopHostToWorkerEnvelope> =
            desktop_ipc::read_desktop_ipc_frame(&mut reader)
                .context("failed to read host frame")?;
        let Some(frame) = frame else {
            break;
        };
        frame
            .validate_version()
            .context("host sent incompatible protocol frame")?;
        match frame.payload {
            DesktopHostToWorkerMessage::Initialize(init) => {
                latest_window = init.window.clone();
                let mut app = fresh_desktop_app_for_worker_mode(init.mode);
                if let Some(snapshot) = init.snapshot.clone()
                    && let Err(error) = app.restore_snapshot(snapshot)
                {
                    desktop_log::error(format_args!(
                        "jcode-desktop: app worker failed to restore host snapshot: {error:#}"
                    ));
                }
                let app_runtime = DesktopAppRuntime::new(app);
                let scene = desktop_scene_for_worker_runtime(&app_runtime, &latest_window);
                runtime = Some(app_runtime);
                let scene_update = DesktopProtocolEnvelope::new(
                    next_worker_sequence,
                    DesktopWorkerToHostMessage::Scene(DesktopSceneUpdate {
                        animation_active: scene.metadata.animation_active,
                        scene,
                    }),
                );
                next_worker_sequence += 1;
                write_desktop_ipc_frame(&mut stdout, &scene_update)
                    .context("failed to write worker initial scene")?;
            }
            DesktopHostToWorkerMessage::SnapshotRequest { request_id } => {
                if let Some(runtime) = runtime.as_ref() {
                    let snapshot = DesktopProtocolEnvelope::new(
                        next_worker_sequence,
                        DesktopWorkerToHostMessage::Snapshot(DesktopSnapshotResponse {
                            request_id,
                            snapshot: runtime.snapshot(),
                        }),
                    );
                    next_worker_sequence += 1;
                    write_desktop_ipc_frame(&mut stdout, &snapshot)
                        .context("failed to write worker snapshot response")?;
                } else {
                    desktop_log::info(format_args!(
                        "jcode-desktop: app worker received snapshot request {request_id} before initialization"
                    ));
                }
            }
            DesktopHostToWorkerMessage::Shutdown {
                reason:
                    DesktopWorkerShutdownReason::HostExit
                    | DesktopWorkerShutdownReason::Reload
                    | DesktopWorkerShutdownReason::ProtocolMismatch,
            } => break,
            DesktopHostToWorkerMessage::Input(input) => {
                let mut changed = false;
                match input {
                    DesktopInputEvent::Key(key) => {
                        if key.pressed
                            && let Some(runtime) = runtime.as_mut()
                        {
                            let outcome =
                                runtime.handle_key_input(desktop_key_event_to_key_input(&key));
                            runtime
                                .driver_mut()
                                .service_pending_transcript_hydration_blocking();
                            if matches!(outcome, KeyOutcome::ForceReload) {
                                let reload_requested = DesktopProtocolEnvelope::new(
                                    next_worker_sequence,
                                    DesktopWorkerToHostMessage::ReloadRequested,
                                );
                                next_worker_sequence += 1;
                                write_desktop_ipc_frame(&mut stdout, &reload_requested)
                                    .context("failed to write worker reload request")?;
                            } else {
                                changed = true;
                            }
                        }
                    }
                    DesktopInputEvent::Window(DesktopWindowEvent::Resized {
                        width,
                        height,
                        scale_factor,
                    }) => {
                        latest_window.width = width;
                        latest_window.height = height;
                        latest_window.scale_factor = scale_factor;
                        changed = true;
                    }
                    DesktopInputEvent::Window(DesktopWindowEvent::Focused(focused)) => {
                        latest_window.focused = focused;
                    }
                    DesktopInputEvent::Mouse(_) => {}
                }
                if changed && let Some(runtime) = runtime.as_ref() {
                    write_worker_scene_update(
                        &mut stdout,
                        &mut next_worker_sequence,
                        runtime,
                        &latest_window,
                    )
                    .context("failed to write worker input scene")?;
                }
            }
            DesktopHostToWorkerMessage::SessionEvents(batch) => {
                let mut changed = false;
                if let Some(runtime) = runtime.as_mut() {
                    for event in batch.events {
                        if let Some(session_event) =
                            desktop_wire_session_event_to_runtime_event(event)
                        {
                            runtime.apply_session_event(session_event);
                            changed = true;
                        }
                    }
                }
                if changed && let Some(runtime) = runtime.as_ref() {
                    write_worker_scene_update(
                        &mut stdout,
                        &mut next_worker_sequence,
                        runtime,
                        &latest_window,
                    )
                    .context("failed to write worker session event scene")?;
                }
            }
            DesktopHostToWorkerMessage::MetricsAck { .. } => {}
        }
    }

    Ok(())
}

#[cfg(test)]
fn desktop_scene_for_worker_init(init: &DesktopWorkerInit) -> DesktopScene {
    let mut scene = DesktopScene::new(DesktopSceneViewport::new(
        init.window.width as f32,
        init.window.height as f32,
        init.window.scale_factor,
    ));
    scene.metadata.title = init
        .snapshot
        .as_ref()
        .map(|snapshot| snapshot.title.clone());
    scene.metadata.content_ready = init.snapshot.is_some();
    scene.push(DesktopDisplayCommand::Clear(DesktopColor::rgba(
        0.02, 0.024, 0.03, 1.0,
    )));
    scene
}

fn desktop_scene_for_worker_runtime(
    runtime: &DesktopAppRuntime<DesktopApp>,
    window: &DesktopWindowState,
) -> DesktopScene {
    let mut scene = DesktopScene::new(DesktopSceneViewport::new(
        window.width as f32,
        window.height as f32,
        window.scale_factor,
    ));
    scene.push(DesktopDisplayCommand::Clear(DesktopColor::rgba(
        0.02, 0.024, 0.03, 1.0,
    )));
    runtime.build_scene(scene)
}

fn write_worker_scene_update(
    stdout: &mut impl Write,
    next_worker_sequence: &mut u64,
    runtime: &DesktopAppRuntime<DesktopApp>,
    window: &DesktopWindowState,
) -> Result<()> {
    let scene = desktop_scene_for_worker_runtime(runtime, window);
    let scene_update = DesktopProtocolEnvelope::new(
        *next_worker_sequence,
        DesktopWorkerToHostMessage::Scene(DesktopSceneUpdate {
            animation_active: scene.metadata.animation_active,
            scene,
        }),
    );
    *next_worker_sequence += 1;
    write_desktop_ipc_frame(stdout, &scene_update)?;
    Ok(())
}

fn desktop_mode_from_args<'a>(args: impl IntoIterator<Item = &'a str>) -> DesktopMode {
    if args.into_iter().any(|arg| arg == "--workspace") {
        DesktopMode::WorkspacePrototype
    } else {
        DesktopMode::SingleSession
    }
}

fn desktop_process_role_from_args<'a>(
    args: impl IntoIterator<Item = &'a str>,
) -> DesktopProcessRole {
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let role = arg
            .strip_prefix("--desktop-process-role=")
            .or_else(|| {
                (arg == "--desktop-process-role")
                    .then(|| args.next())
                    .flatten()
            })
            .or_else(|| {
                (arg == "--desktop-host")
                    .then_some("host")
                    .or_else(|| (arg == "--desktop-app-worker").then_some("worker"))
            });
        if let Some(role) = role {
            return match role {
                "host" | "stable-host" | "stable_host" => DesktopProcessRole::StableHost,
                "worker" | "app-worker" | "app_worker" => DesktopProcessRole::AppWorker,
                _ => DesktopProcessRole::Standalone,
            };
        }
    }
    DesktopProcessRole::StableHost
}

fn desktop_resume_session_id_from_args<'a>(
    args: impl IntoIterator<Item = &'a str>,
) -> Option<String> {
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        if arg == "--resume" {
            return args.next().map(str::to_string);
        }
        if let Some(session_id) = arg.strip_prefix("--resume=") {
            return (!session_id.is_empty()).then(|| session_id.to_string());
        }
    }
    None
}


#[allow(clippy::large_enum_variant)]
enum DesktopApp {
    SingleSession(SingleSessionApp),
    Workspace(Workspace),
}

#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
struct DesktopAppDebugSnapshot {
    mode: &'static str,
    title: String,
    live_session_id: Option<String>,
    status: Option<String>,
    is_processing: bool,
    body_text: String,
}

impl DesktopApp {
    fn mode(&self) -> &'static str {
        match self {
            Self::SingleSession(_) => "single_session",
            Self::Workspace(_) => "workspace",
        }
    }

    fn is_single_session(&self) -> bool {
        matches!(self, Self::SingleSession(_))
    }

    fn is_workspace(&self) -> bool {
        matches!(self, Self::Workspace(_))
    }

    fn has_background_work(&self) -> bool {
        matches!(self, Self::SingleSession(app) if app.has_background_work())
    }

    fn has_frame_animation(&self) -> bool {
        matches!(self, Self::SingleSession(app) if app.has_frame_animation())
    }

    fn status_title(&self) -> String {
        match self {
            Self::SingleSession(app) => app.status_title(),
            Self::Workspace(workspace) => workspace.status_title(),
        }
    }

    fn handle_key(&mut self, key: KeyInput) -> KeyOutcome {
        match self {
            Self::SingleSession(app) => app.handle_key(key),
            Self::Workspace(workspace) => workspace.handle_key(key),
        }
    }

    fn promote_focused_workspace_session(&mut self) -> bool {
        let Self::Workspace(workspace) = self else {
            return false;
        };
        let Some(card) = workspace.focused_session_card() else {
            return false;
        };
        let session_id = card.session_id.clone();
        let mut single_session = SingleSessionApp::new(Some(card));
        single_session.initialize_resumed_session(&session_id);
        single_session.request_transcript_hydration(&session_id);
        *self = Self::SingleSession(single_session);
        true
    }

    /// Take the session id queued for off-thread transcript hydration.
    fn take_pending_transcript_hydration(&mut self) -> Option<String> {
        match self {
            Self::SingleSession(app) => app.take_pending_transcript_hydration(),
            Self::Workspace(_) => None,
        }
    }

    /// Apply a transcript that finished loading off the UI thread.
    fn apply_hydrated_transcript(
        &mut self,
        session_id: &str,
        result: std::result::Result<Option<Vec<workspace::SessionTranscriptMessage>>, String>,
    ) -> bool {
        match self {
            Self::SingleSession(app) => app.apply_hydrated_transcript(session_id, result),
            Self::Workspace(_) => false,
        }
    }

    /// Service any queued transcript hydration synchronously. Used by the
    /// app-worker process, which has no event-loop proxy; the disk scan is
    /// bounded so the worst case stays small.
    fn service_pending_transcript_hydration_blocking(&mut self) {
        if let Self::SingleSession(app) = self
            && let Some(session_id) = app.take_pending_transcript_hydration()
        {
            app.hydrate_resumed_session_from_disk(&session_id);
        }
    }

    fn apply_session_event(&mut self, event: session_launch::DesktopSessionEvent) {
        if let Self::SingleSession(app) = self {
            app.apply_session_event(event);
        }
    }

    fn set_single_session_status_label(&mut self, label: impl Into<String>) {
        if let Self::SingleSession(app) = self {
            app.set_status_label(label);
        }
    }

    fn take_github_issue_sync_request(&mut self) -> bool {
        match self {
            Self::SingleSession(app) => app.take_github_issue_sync_request(),
            Self::Workspace(_) => false,
        }
    }

    fn note_github_issue_sync_already_running(&mut self) {
        if let Self::SingleSession(app) = self {
            app.note_github_issue_sync_already_running();
        }
    }

    fn apply_github_issue_sync_result(
        &mut self,
        result: std::result::Result<desktop_issue_cache::GitHubIssueSyncSummary, String>,
    ) {
        if let Self::SingleSession(app) = self {
            app.apply_github_issue_sync_result(result);
        }
    }

    fn preview_single_session_reasoning_effort_cycle(
        &mut self,
        direction: i8,
    ) -> ReasoningEffortCycleOutcome {
        match self {
            Self::SingleSession(app) => app.preview_reasoning_effort_cycle(direction),
            Self::Workspace(_) => ReasoningEffortCycleOutcome::Unavailable,
        }
    }

    fn preview_single_session_reasoning_effort_set(&mut self, effort: &str) -> Option<String> {
        match self {
            Self::SingleSession(app) => app.preview_reasoning_effort_set(effort),
            Self::Workspace(_) => None,
        }
    }

    fn set_reasoning_effort_via_active_session(&mut self, effort: String) -> anyhow::Result<()> {
        match self {
            Self::SingleSession(app) => app.set_reasoning_effort_via_active_session(effort),
            Self::Workspace(_) => {
                anyhow::bail!("reasoning effort changes require single-session mode")
            }
        }
    }

    fn set_single_session_handle(&mut self, handle: session_launch::DesktopSessionHandle) {
        if let Self::SingleSession(app) = self {
            app.set_session_handle(handle);
        }
    }

    fn apply_single_session_switcher_cards(&mut self, cards: Vec<workspace::SessionCard>) {
        if let Self::SingleSession(app) = self {
            app.apply_session_switcher_cards(cards);
        }
    }

    fn cancel_single_session_generation(&mut self) {
        if let Self::SingleSession(app) = self {
            app.cancel_generation();
        }
    }

    fn attach_clipboard_image(&mut self, media_type: String, base64_data: String) {
        match self {
            Self::SingleSession(app) => app.attach_image(media_type, base64_data),
            Self::Workspace(workspace) => {
                workspace.attach_image(media_type, base64_data);
            }
        }
    }

    fn accepts_clipboard_image_paste(&self) -> bool {
        match self {
            Self::SingleSession(app) => app.accepts_clipboard_image_paste(),
            Self::Workspace(workspace) => workspace.mode == InputMode::Insert,
        }
    }

    fn paste_text(&mut self, text: &str) {
        match self {
            Self::SingleSession(app) => app.paste_text(text),
            Self::Workspace(workspace) => {
                workspace.paste_text(text);
            }
        }
    }

    fn send_single_session_stdin_response(
        &mut self,
        request_id: String,
        input: String,
    ) -> anyhow::Result<()> {
        match self {
            Self::SingleSession(app) => app.send_stdin_response(request_id, input),
            Self::Workspace(_) => {
                anyhow::bail!("stdin responses are only supported in single-session mode")
            }
        }
    }

    fn take_next_queued_single_session_draft(&mut self) -> Option<(String, Vec<(String, String)>)> {
        match self {
            Self::SingleSession(app) => app.take_next_queued_draft(),
            Self::Workspace(_) => None,
        }
    }

    fn begin_single_session_selection_at(
        &mut self,
        x: f32,
        y: f32,
        size: PhysicalSize<u32>,
    ) -> bool {
        if let Self::SingleSession(app) = self {
            let lines = single_session_visible_body(app, size);
            if let Some(point) = single_session_body_point_at_position(size, x, y, &lines) {
                app.begin_selection(point);
                return true;
            }
        }
        false
    }

    fn update_single_session_selection_at(
        &mut self,
        x: f32,
        y: f32,
        size: PhysicalSize<u32>,
    ) -> bool {
        if let Self::SingleSession(app) = self {
            let lines = single_session_visible_body(app, size);
            if let Some(point) = single_session_body_point_at_position(size, x, y, &lines) {
                app.update_selection(point);
                return true;
            }
        }
        false
    }

    fn begin_single_session_draft_selection_at(
        &mut self,
        x: f32,
        y: f32,
        size: PhysicalSize<u32>,
    ) -> bool {
        if let Self::SingleSession(app) = self
            && let Some((line, column)) = single_session_draft_line_col_at_position(app, size, x, y)
        {
            app.begin_draft_selection(SelectionPoint { line, column });
            return true;
        }
        false
    }

    fn update_single_session_draft_selection_at(
        &mut self,
        x: f32,
        y: f32,
        size: PhysicalSize<u32>,
    ) -> bool {
        if let Self::SingleSession(app) = self
            && let Some((line, column)) = single_session_draft_line_col_at_position(app, size, x, y)
        {
            app.update_draft_selection(SelectionPoint { line, column });
            return true;
        }
        false
    }

    fn selected_single_session_draft_text(&mut self) -> Option<String> {
        if let Self::SingleSession(app) = self {
            return app.selected_draft_text();
        }
        None
    }

    fn selected_single_session_text(&mut self, size: PhysicalSize<u32>) -> Option<String> {
        if let Self::SingleSession(app) = self {
            let lines = single_session_visible_body(app, size);
            let selected = app.selected_text_from_lines(&lines);
            app.clear_selection();
            return selected;
        }
        None
    }

    fn scroll_single_session_body(
        &mut self,
        lines: impl Into<f64>,
        size: PhysicalSize<u32>,
        metrics_cache: &mut SingleSessionScrollMetricsCache,
    ) -> bool {
        if let Self::SingleSession(app) = self {
            let previous_scroll_lines = app.body_scroll_lines;
            app.scroll_body_lines(lines);
            if let Some(metrics) = metrics_cache.metrics(app, size) {
                app.body_scroll_lines = app.body_scroll_lines.min(metrics.max_scroll_lines as f32);
            } else {
                app.body_scroll_lines = 0.0;
            }
            return (app.body_scroll_lines - previous_scroll_lines).abs()
                >= SCROLL_FRACTIONAL_EPSILON;
        }
        false
    }

    fn single_session_smooth_scroll_lines(
        &self,
        pending_lines: f32,
        size: PhysicalSize<u32>,
        metrics_cache: &mut SingleSessionScrollMetricsCache,
    ) -> f32 {
        let Self::SingleSession(app) = self else {
            return 0.0;
        };
        let Some(metrics) = metrics_cache.metrics(app, size) else {
            return 0.0;
        };
        let base_scroll = app.body_scroll_lines.min(metrics.max_scroll_lines as f32);
        (base_scroll + pending_lines).clamp(0.0, metrics.max_scroll_lines as f32) - base_scroll
    }

    fn single_session_live_id(&self) -> Option<String> {
        match self {
            Self::SingleSession(app) => app.live_session_id.clone(),
            Self::Workspace(_) => None,
        }
    }

    #[cfg(test)]
    fn debug_snapshot(&self) -> DesktopAppDebugSnapshot {
        match self {
            Self::SingleSession(app) => DesktopAppDebugSnapshot {
                mode: "single_session",
                title: app.title(),
                live_session_id: app.live_session_id.clone(),
                status: app.status.clone(),
                is_processing: app.is_processing,
                body_text: app.body_lines().join("\n"),
            },
            Self::Workspace(workspace) => DesktopAppDebugSnapshot {
                mode: "workspace",
                title: workspace.status_title(),
                live_session_id: None,
                status: None,
                is_processing: false,
                body_text: workspace.status_title(),
            },
        }
    }
}


impl DesktopAppDriver for DesktopApp {
    type KeyInput = KeyInput;
    type KeyOutcome = KeyOutcome;

    fn mode(&self) -> &'static str {
        DesktopApp::mode(self)
    }

    fn status_title(&self) -> String {
        DesktopApp::status_title(self)
    }

    fn live_session_id(&self) -> Option<String> {
        DesktopApp::single_session_live_id(self)
    }

    fn has_background_work(&self) -> bool {
        DesktopApp::has_background_work(self)
    }

    fn has_frame_animation(&self) -> bool {
        DesktopApp::has_frame_animation(self)
    }

    fn handle_key_input(&mut self, key: Self::KeyInput) -> Self::KeyOutcome {
        DesktopApp::handle_key(self, key)
    }

    fn apply_session_event(&mut self, event: session_launch::DesktopSessionEvent) {
        DesktopApp::apply_session_event(self, event);
    }

    fn build_scene(&self, context: DesktopSceneBuildContext) -> desktop_scene::DesktopScene {
        desktop_app_scene(self, context.scene)
    }

    fn snapshot(&self) -> DesktopUiSnapshot {
        DesktopUiSnapshot::new(
            DesktopApp::mode(self),
            DesktopApp::status_title(self),
            DesktopApp::single_session_live_id(self),
            desktop_surface_snapshot(self),
        )
    }

    fn restore_snapshot(
        &mut self,
        snapshot: DesktopUiSnapshot,
    ) -> Result<(), DesktopSnapshotRestoreError> {
        if snapshot.version != DESKTOP_UI_SNAPSHOT_VERSION {
            return Err(DesktopSnapshotRestoreError::UnsupportedVersion {
                version: snapshot.version,
            });
        }
        if snapshot.mode != DesktopApp::mode(self) {
            return Err(DesktopSnapshotRestoreError::UnsupportedMode {
                mode: snapshot.mode,
            });
        }
        Ok(())
    }
}

fn desktop_app_scene(app: &DesktopApp, mut scene: DesktopScene) -> DesktopScene {
    scene.metadata.title = Some(app.status_title());
    scene.metadata.animation_active = app.has_frame_animation();
    scene.metadata.content_ready = true;
    if scene.display_list.commands.is_empty() {
        scene.push(DesktopDisplayCommand::Clear(DesktopColor::from_array(
            BACKGROUND_TOP_LEFT,
        )));
    }
    scene
}

fn desktop_surface_snapshot(app: &DesktopApp) -> DesktopSurfaceSnapshot {
    match app {
        DesktopApp::SingleSession(single_session) => {
            DesktopSurfaceSnapshot::SingleSession(DesktopSingleSessionSnapshot {
                session_title: single_session
                    .session
                    .as_ref()
                    .map(|session| session.title.clone()),
                draft: single_session.draft.clone(),
                draft_cursor: single_session.draft_cursor,
                body_scroll_millis: (single_session.body_scroll_lines * 1000.0).round() as i32,
                detail_scroll: single_session.detail_scroll,
                show_help: single_session.show_help,
                show_session_info: single_session.show_session_info,
                pending_image_count: single_session.pending_images.len(),
                model_picker_open: single_session.model_picker.open,
                session_switcher_open: single_session.session_switcher.open,
                stdin_response_active: single_session.stdin_response.is_some(),
            })
        }
        DesktopApp::Workspace(workspace) => {
            let focused_session_id = workspace
                .surfaces
                .iter()
                .find(|surface| surface.id == workspace.focused_id)
                .and_then(|surface| surface.session_id.clone());
            DesktopSurfaceSnapshot::Workspace(DesktopWorkspaceSnapshot {
                input_mode: format!("{:?}", workspace.mode),
                focused_surface_id: workspace.focused_id,
                focused_session_id,
                zoomed: workspace.zoomed,
                detail_scroll: workspace.detail_scroll,
                draft: workspace.draft.clone(),
                draft_cursor: workspace.draft_cursor,
                pending_image_count: workspace.pending_images.len(),
                surfaces: workspace
                    .surfaces
                    .iter()
                    .map(|surface| DesktopWorkspaceSurfaceSnapshot {
                        id: surface.id,
                        kind: format!("{:?}", surface.kind),
                        title: surface.title.clone(),
                        session_id: surface.session_id.clone(),
                        lane: surface.lane,
                        column: surface.column,
                        color_index: surface.color_index,
                    })
                    .collect(),
            })
        }
    }
}

fn apply_single_session_error(app: &mut DesktopApp, error: anyhow::Error) {
    desktop_log::error(format_args!("jcode-desktop: UI action failed: {error:#}"));
    app.apply_session_event(session_launch::DesktopSessionEvent::Error(format!(
        "{error:#}"
    )));
}

#[derive(Default)]
struct DesktopClipboard {
    clipboard: Option<arboard::Clipboard>,
}

impl DesktopClipboard {
    fn clipboard(&mut self) -> Result<&mut arboard::Clipboard> {
        if self.clipboard.is_none() {
            self.clipboard = Some(arboard::Clipboard::new().context("failed to access clipboard")?);
        }
        self.clipboard
            .as_mut()
            .context("failed to retain clipboard handle")
    }

    fn set_text(&mut self, text: &str) -> Result<()> {
        self.with_clipboard_retry("failed to set clipboard text", |clipboard| {
            clipboard.set_text(text.to_string())
        })
    }

    fn get_text(&mut self) -> Result<String> {
        self.with_clipboard_retry("clipboard does not contain text", |clipboard| {
            clipboard.get_text()
        })
    }

    fn get_image(&mut self) -> Result<arboard::ImageData<'static>> {
        self.with_clipboard_retry("clipboard does not contain an image", |clipboard| {
            clipboard.get_image()
        })
    }

    fn with_clipboard_retry<T>(
        &mut self,
        context: &'static str,
        mut operation: impl FnMut(&mut arboard::Clipboard) -> Result<T, arboard::Error>,
    ) -> Result<T> {
        const CLIPBOARD_RETRY_ATTEMPTS: usize = 3;
        const CLIPBOARD_RETRY_DELAY: Duration = Duration::from_millis(20);

        for attempt in 0..CLIPBOARD_RETRY_ATTEMPTS {
            let result = operation(self.clipboard()?);
            match result {
                Ok(value) => return Ok(value),
                Err(error)
                    if matches!(&error, arboard::Error::ClipboardOccupied)
                        && attempt + 1 < CLIPBOARD_RETRY_ATTEMPTS =>
                {
                    std::thread::sleep(CLIPBOARD_RETRY_DELAY);
                }
                Err(error) => {
                    if !matches!(
                        &error,
                        arboard::Error::ContentNotAvailable | arboard::Error::ClipboardOccupied
                    ) {
                        self.clipboard = None;
                    }
                    return Err(error).context(context);
                }
            }
        }

        anyhow::bail!("clipboard remained occupied after retrying")
    }
}

fn copy_text_to_clipboard(
    clipboard: &mut DesktopClipboard,
    text: &str,
    success_notice: &'static str,
    app: &mut DesktopApp,
) {
    match clipboard.set_text(text) {
        Ok(()) => app.set_single_session_status_label(success_notice),
        Err(error) => {
            desktop_log::error(format_args!(
                "jcode-desktop: failed to update clipboard after {success_notice}: {error:#}"
            ));
            app.apply_session_event(session_launch::DesktopSessionEvent::Error(format!(
                "failed to update clipboard after {success_notice}: {error:#}"
            )));
        }
    }
}

fn paste_clipboard_into_app(clipboard: &mut DesktopClipboard, app: &mut DesktopApp) -> Result<()> {
    match clipboard_text(clipboard) {
        Ok(text) => {
            if paste_clipboard_text(app, &text) || !app.accepts_clipboard_image_paste() {
                return Ok(());
            }
            paste_clipboard_image_into_app(clipboard, app)
                .with_context(|| "clipboard text was empty and no pasteable image was available")
        }
        Err(text_error) if app.accepts_clipboard_image_paste() => {
            paste_clipboard_image_into_app(clipboard, app)
                .with_context(|| format!("clipboard did not contain pasteable text: {text_error}"))
        }
        Err(error) => Err(error),
    }
}

fn paste_clipboard_text(app: &mut DesktopApp, text: &str) -> bool {
    let text = normalize_clipboard_text(text);
    if text.is_empty() {
        return false;
    }
    app.paste_text(&text);
    true
}

fn paste_clipboard_image_into_app(
    clipboard: &mut DesktopClipboard,
    app: &mut DesktopApp,
) -> Result<()> {
    let (media_type, base64_data) = clipboard_image_png_base64(clipboard)?;
    app.attach_clipboard_image(media_type, base64_data);
    Ok(())
}

fn normalize_clipboard_text(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

fn clipboard_image_png_base64(clipboard: &mut DesktopClipboard) -> Result<(String, String)> {
    let image = clipboard.get_image()?;
    let width = u32::try_from(image.width).context("clipboard image is too wide")?;
    let height = u32::try_from(image.height).context("clipboard image is too tall")?;
    let rgba = image.bytes.into_owned();
    let buffer = image::RgbaImage::from_raw(width, height, rgba)
        .context("clipboard image data had unexpected dimensions")?;
    let mut cursor = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(buffer)
        .write_to(&mut cursor, image::ImageFormat::Png)
        .context("failed to encode clipboard image as png")?;
    Ok((
        "image/png".to_string(),
        base64::engine::general_purpose::STANDARD.encode(cursor.into_inner()),
    ))
}

fn clipboard_text(clipboard: &mut DesktopClipboard) -> Result<String> {
    clipboard.get_text()
}

#[derive(Clone, Debug, Default)]
struct ScrollLineAccumulator {
    velocity_lines_per_second: f32,
    last_event_at: Option<Instant>,
    last_frame_at: Option<Instant>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ScrollAnimationFrame {
    scroll_lines: Option<f32>,
    active: bool,
}

impl ScrollLineAccumulator {
    fn scroll_lines(&mut self, delta: MouseScrollDelta, now: Instant) -> Option<f32> {
        if self
            .last_event_at
            .is_some_and(|last| now.saturating_duration_since(last) > SCROLL_GESTURE_IDLE_RESET)
        {
            self.stop();
        }
        self.last_event_at = Some(now);
        self.last_frame_at = Some(now);
        self.input_delta(mouse_scroll_delta_lines(delta))
    }

    fn frame(&mut self, now: Instant) -> ScrollAnimationFrame {
        let Some(last_frame_at) = self.last_frame_at else {
            self.last_frame_at = Some(now);
            return ScrollAnimationFrame {
                scroll_lines: None,
                active: self.is_active(),
            };
        };

        let dt = now
            .saturating_duration_since(last_frame_at)
            .as_secs_f32()
            .min(SCROLL_FRAME_MAX_DT_SECONDS);
        self.last_frame_at = Some(now);

        if dt <= 0.0 || !self.is_active() {
            return ScrollAnimationFrame {
                scroll_lines: None,
                active: self.is_active(),
            };
        }

        let scroll_lines = if self.velocity_lines_per_second.abs() >= SCROLL_MOMENTUM_STOP_VELOCITY
        {
            let lines = self.velocity_lines_per_second * dt;
            let decay = (-SCROLL_MOMENTUM_DECAY_PER_SECOND * dt).exp();
            self.velocity_lines_per_second *= decay;
            if self.velocity_lines_per_second.abs() < SCROLL_MOMENTUM_STOP_VELOCITY {
                self.velocity_lines_per_second = 0.0;
            }
            (lines.abs() >= SCROLL_FRACTIONAL_EPSILON).then_some(lines)
        } else {
            self.velocity_lines_per_second = 0.0;
            None
        };

        ScrollAnimationFrame {
            scroll_lines,
            active: self.is_active(),
        }
    }

    fn reset(&mut self) {
        self.stop();
        self.last_event_at = None;
        self.last_frame_at = None;
    }

    fn stop(&mut self) {
        self.velocity_lines_per_second = 0.0;
    }

    fn pending_lines(&self) -> f32 {
        0.0
    }

    fn is_active(&self) -> bool {
        self.velocity_lines_per_second.abs() >= SCROLL_MOMENTUM_STOP_VELOCITY
    }

    fn input_delta(&mut self, lines: f32) -> Option<f32> {
        if !lines.is_finite() || lines.abs() < SCROLL_FRACTIONAL_EPSILON {
            return None;
        }

        let lines = lines.clamp(
            -MAX_MOUSE_SCROLL_LINES_PER_EVENT,
            MAX_MOUSE_SCROLL_LINES_PER_EVENT,
        );
        if self.velocity_lines_per_second.abs() >= SCROLL_MOMENTUM_STOP_VELOCITY
            && self.velocity_lines_per_second.signum() != lines.signum()
        {
            self.stop();
        }

        self.velocity_lines_per_second = (self.velocity_lines_per_second
            + lines * SCROLL_MOMENTUM_GAIN)
            .clamp(-SCROLL_MOMENTUM_MAX_VELOCITY, SCROLL_MOMENTUM_MAX_VELOCITY);
        Some(lines)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SingleSessionScrollMotionFrame {
    visual_scroll_lines: f32,
    smooth_scroll_lines: f32,
    active: bool,
}

#[derive(Clone, Debug, Default)]
struct SingleSessionScrollMotion {
    initialized: bool,
    start_lines: f32,
    current_lines: f32,
    target_lines: f32,
    started_at: Option<Instant>,
}

impl SingleSessionScrollMotion {
    fn frame(&mut self, target_lines: f32, now: Instant) -> SingleSessionScrollMotionFrame {
        let target_lines = if target_lines.is_finite() {
            target_lines.max(0.0)
        } else {
            0.0
        };

        if !self.initialized || animation::desktop_reduced_motion_enabled() {
            self.initialized = true;
            self.start_lines = target_lines;
            self.current_lines = target_lines;
            self.target_lines = target_lines;
            self.started_at = None;
            return SingleSessionScrollMotionFrame {
                visual_scroll_lines: target_lines,
                smooth_scroll_lines: 0.0,
                active: false,
            };
        }

        if (self.target_lines - target_lines).abs() >= SCROLL_FRACTIONAL_EPSILON {
            self.start_lines = self.current_lines;
            self.target_lines = target_lines;
            self.started_at = Some(now);
        }

        if let Some(started_at) = self.started_at {
            let progress = (now.saturating_duration_since(started_at).as_secs_f32()
                / SINGLE_SESSION_SCROLL_ANIMATION_DURATION.as_secs_f32())
            .clamp(0.0, 1.0);
            let eased = animation::ease_out_cubic(progress);
            self.current_lines = animation::lerp(self.start_lines, self.target_lines, eased);
            if progress >= 1.0
                || (self.current_lines - self.target_lines).abs() < SCROLL_FRACTIONAL_EPSILON
            {
                self.current_lines = self.target_lines;
                self.started_at = None;
            }
        }

        SingleSessionScrollMotionFrame {
            visual_scroll_lines: self.current_lines,
            smooth_scroll_lines: self.current_lines - target_lines,
            active: self.is_active(),
        }
    }

    fn is_active(&self) -> bool {
        self.started_at.is_some()
            || (self.current_lines - self.target_lines).abs() >= SCROLL_FRACTIONAL_EPSILON
    }

    fn clear(&mut self) {
        self.initialized = false;
        self.start_lines = 0.0;
        self.current_lines = 0.0;
        self.target_lines = 0.0;
        self.started_at = None;
    }
}

#[cfg(test)]
fn mouse_scroll_lines(delta: MouseScrollDelta) -> Option<f32> {
    ScrollLineAccumulator::default().scroll_lines(delta, Instant::now())
}

fn mouse_scroll_delta_lines(delta: MouseScrollDelta) -> f32 {
    match delta {
        MouseScrollDelta::LineDelta(_, y) => y * MOUSE_WHEEL_LINES_PER_DETENT,
        MouseScrollDelta::PixelDelta(position) => position.y as f32 / body_scroll_line_pixels(),
    }
}

fn body_scroll_line_pixels() -> f32 {
    let typography = single_session_typography();
    typography.body_size * typography.body_line_height
}

fn desktop_spinner_tick(_now: Instant) -> u64 {
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0);
    (millis / DESKTOP_SPINNER_FRAME_MS) as u64
}

/// Continuous wall-clock seconds for smooth (unquantized) pulse animations.
/// Unlike `desktop_spinner_tick`, this is not stepped to 180ms frames, so
/// breathing cues animate fluidly at the paced 16ms redraw interval.
fn desktop_pulse_seconds() -> f32 {
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0);
    // Wrap at a day to keep f32 precision; pulse phases only use fract().
    ((millis % 86_400_000) as f64 / 1000.0) as f32
}

fn single_session_text_buffer_cache_key(
    app: &SingleSessionApp,
    size: PhysicalSize<u32>,
    _tick: u64,
    rendered_body_key: u64,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    rendered_body_key.hash(&mut hasher);
    (size.width, size.height).hash(&mut hasher);
    app.is_welcome_timeline_visible().hash(&mut hasher);
    app.has_activity_indicator().hash(&mut hasher);
    app.text_scale().to_bits().hash(&mut hasher);
    app.header_title().hash(&mut hasher);
    app.welcome_hero_text().hash(&mut hasher);
    // Use the render-time styled lines (which honor the reveal animation) so the
    // text buffer cache key matches the content actually placed into the buffer.
    // Hashing the pre-reveal lines here while the buffer is built from the
    // reveal-aware lines causes a stale buffer: the picker chrome re-renders every
    // frame but the glyph text is never re-prepared as the reveal progresses.
    app.render_inline_widget_styled_lines().hash(&mut hasher);
    // The inline-widget card grows via a reveal animation, and the glyph text is
    // vertically clipped to the animating card bounds. Quantize the reveal
    // progress into the cache key so the text buffer is re-prepared across the
    // whole animation; otherwise the text is prepared once early (while the card
    // is still small and clips everything but the first line) and never refreshed,
    // leaving stale/partial text under fully-rendered chrome.
    if app.render_inline_widget_kind().is_some() {
        let reveal_bucket = (app.render_inline_widget_reveal_progress() * 32.0).round() as u32;
        reveal_bucket.hash(&mut hasher);
    }
    app.composer_text().hash(&mut hasher);
    hasher.finish()
}

fn single_session_body_text_window_bounds(viewport: &SingleSessionBodyViewport) -> (usize, usize) {
    let start = viewport
        .start_line
        .saturating_sub(SINGLE_SESSION_BODY_TEXT_WINDOW_BEFORE_LINES);
    let end = viewport
        .start_line
        .saturating_add(viewport.lines.len())
        .saturating_add(SINGLE_SESSION_BODY_TEXT_WINDOW_AFTER_LINES)
        .min(viewport.total_lines);
    (start, end.max(start))
}

fn single_session_body_text_window_contains(
    window_start: usize,
    window_end: usize,
    viewport: &SingleSessionBodyViewport,
) -> bool {
    let visible_end = viewport.start_line.saturating_add(viewport.lines.len());
    window_start <= viewport.start_line && visible_end <= window_end
}

#[derive(Default)]
struct SingleSessionScrollMetricsCache {
    key: Option<u64>,
    total_lines: usize,
    raw_body_key: Option<u64>,
    raw_body_lines: Vec<SingleSessionStyledLine>,
    streaming_base_key: Option<u64>,
    streaming_base_total_lines: usize,
}

impl SingleSessionScrollMetricsCache {
    fn metrics(
        &mut self,
        app: &SingleSessionApp,
        size: PhysicalSize<u32>,
    ) -> Option<SingleSessionBodyScrollMetrics> {
        let body_layout_size = single_session_body_layout_cache_size(app, size);
        let key = app.rendered_body_cache_key(body_layout_size);
        if self.key != Some(key) {
            if !app.streaming_response.is_empty() {
                let base_key = app.rendered_body_static_cache_key(body_layout_size);
                if self.streaming_base_key != Some(base_key) {
                    if let Some(base_lines) =
                        single_session_rendered_static_body_lines_for_streaming(app, size, 0)
                    {
                        self.streaming_base_total_lines = base_lines.len();
                        self.streaming_base_key = Some(base_key);
                    } else {
                        self.streaming_base_key = None;
                        self.streaming_base_total_lines = 0;
                    }
                }
                if self.streaming_base_key == Some(base_key) {
                    self.total_lines = self.streaming_base_total_lines
                        + single_session_streaming_response_rendered_body_line_count(app, size);
                } else {
                    self.total_lines =
                        single_session_rendered_body_lines_for_tick(app, size, 0).len();
                }
            } else {
                let raw_key = app.rendered_body_cache_key((0, 0));
                if self.raw_body_key != Some(raw_key) {
                    self.raw_body_lines = app.body_styled_lines_for_tick(0);
                    self.raw_body_key = Some(raw_key);
                }
                self.total_lines = single_session_rendered_body_lines_from_raw_ref(
                    app,
                    size,
                    &self.raw_body_lines,
                )
                .len();
                self.streaming_base_key = None;
                self.streaming_base_total_lines = 0;
            }
            self.key = Some(key);
        }
        single_session_body_scroll_metrics_for_total_lines(app, size, self.total_lines)
    }

    fn clear(&mut self) {
        self.key = None;
        self.total_lines = 0;
        self.raw_body_key = None;
        self.raw_body_lines.clear();
        self.streaming_base_key = None;
        self.streaming_base_total_lines = 0;
    }
}





fn desktop_build_hash_label() -> &'static str {
    option_env!("JCODE_DESKTOP_GIT_HASH").unwrap_or("unknown")
}

#[cfg(test)]
#[path = "main_tests.rs"]
mod tests;
