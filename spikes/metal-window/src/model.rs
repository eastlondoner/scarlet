//! The handoff between AppKit's main thread and the VM thread, in plain Rust.
//!
//! Nothing here names an Apple type. The main thread writes into `Pending`
//! (under one mutex) from its event handlers and the display-link callback;
//! the worker drains it in `WindowShared::next_frame`. This is the shape the
//! real `Platform` would hold per window.

use std::collections::BTreeSet;
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

/// A physical key: the macOS virtual keycode (`kVK_*`). Layout-independent,
/// so "W" means the key in W's position on any keyboard layout, which is what
/// movement wants. Typed characters come separately, as `Event::Text`.
pub type Key = u16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloseCause {
    /// The window's close button (or Cmd-W, or `performClose:`).
    CloseButton,
    /// Cmd-Q, the app menu's Quit, the Dock's Quit, or logout.
    Quit,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    KeyDown(Key),
    KeyUp(Key),
    Text(String),
    MouseDown(u8),
    MouseUp(u8),
    FocusLost,
    FocusGained,
    /// Drawable size in pixels (points * backing scale factor).
    Resized {
        width: u32,
        height: u32,
    },
    CloseRequested(CloseCause),
}

#[derive(Debug, Clone, Default)]
pub struct Input {
    pub held: Vec<Key>,
    /// Summed since the last frame, in points (AppKit's `deltaX`/`deltaY`).
    pub mouse_dx: f64,
    pub mouse_dy: f64,
    pub scroll: f64,
    /// How many raw move events were summed into `mouse_dx`/`mouse_dy`.
    /// Measurement only; the real API would not carry it.
    pub mouse_events_summed: u32,
}

#[derive(Debug, Clone)]
pub struct Frame {
    /// Display-link ticks since the previous frame. 1 is on pace; >1 means the
    /// worker missed refreshes; 0 means no tick came (the window is occluded
    /// or minimised and the fallback timer woke us).
    pub ticks: u64,
    pub dt_ms: f64,
    pub width: u32,
    pub height: u32,
    pub scale: f64,
    pub input: Input,
    pub events: Vec<Event>,
    /// Measurement only: time from the display-link callback to this return.
    pub tick_to_return: Option<Duration>,
    /// Measurement only: whether the worker was already waiting when the tick
    /// came (so `tick_to_return` is pure wake-up latency, not backlog).
    pub worker_was_waiting: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppError {
    /// The window was closed by the runtime (not a close *request*).
    WindowGone,
    /// An Objective-C exception was caught at the boundary.
    Exception(String),
    /// A Rust panic was caught at the boundary (a runtime bug).
    Panic,
    NoMetalDevice,
    Unsupported(String),
}

/// Everything the main thread has observed since the worker last took a frame.
#[derive(Debug)]
pub struct Pending {
    pub tick_seq: u64,
    pub tick_at: Option<Instant>,
    /// CADisplayLink's `timestamp` of the latest tick (CACurrentMediaTime base).
    pub link_timestamp: f64,
    pub link_duration: f64,
    pub held: BTreeSet<Key>,
    pub mouse_dx: f64,
    pub mouse_dy: f64,
    pub mouse_events: u32,
    pub scroll: f64,
    pub events: Vec<Event>,
    pub width: u32,
    pub height: u32,
    pub scale: f64,
    pub window_open: bool,
    pub worker_waiting: bool,
    /// Measurement only: callback time minus the link's `timestamp`, seconds.
    pub callback_lag: Vec<f64>,
}

impl Pending {
    fn new(width: u32, height: u32, scale: f64) -> Self {
        Pending {
            tick_seq: 0,
            tick_at: None,
            link_timestamp: 0.0,
            link_duration: 0.0,
            held: BTreeSet::new(),
            mouse_dx: 0.0,
            mouse_dy: 0.0,
            mouse_events: 0,
            scroll: 0.0,
            events: Vec::new(),
            width,
            height,
            scale,
            window_open: true,
            worker_waiting: false,
            callback_lag: Vec::new(),
        }
    }

    pub fn key_down(&mut self, key: Key) {
        // A key already held is an OS auto-repeat (or a desync): not a new press.
        if self.held.insert(key) {
            self.events.push(Event::KeyDown(key));
        }
    }

    pub fn key_up(&mut self, key: Key) {
        if self.held.remove(&key) {
            self.events.push(Event::KeyUp(key));
        }
    }

    /// Focus loss releases every held key: the OS will not send their key-ups
    /// to a window that is no longer key, so without this they stick.
    pub fn release_all(&mut self) {
        let held = std::mem::take(&mut self.held);
        self.events.extend(held.into_iter().map(Event::KeyUp));
    }

    /// Live resize sends dozens of size changes per frame; only the last one
    /// matters, so an earlier pending `Resized` is replaced rather than queued.
    pub fn resized(&mut self, width: u32, height: u32, scale: f64) {
        self.width = width;
        self.height = height;
        self.scale = scale;
        self.events.retain(|e| !matches!(e, Event::Resized { .. }));
        self.events.push(Event::Resized { width, height });
    }
}

pub struct WindowShared {
    pending: Mutex<Pending>,
    wake: Condvar,
}

/// The worker's private position in the tick stream.
pub struct FrameCursor {
    seen_seq: u64,
    last_link_ts: f64,
    last_return: Instant,
}

impl FrameCursor {
    pub fn new() -> Self {
        FrameCursor {
            seen_seq: 0,
            last_link_ts: 0.0,
            last_return: Instant::now(),
        }
    }
}

/// With no tick for this long, `next_frame` returns anyway. A hidden or
/// minimised window's display link stops, and a program that does anything
/// else on the VM thread (a network keep-alive) must not stop with it.
pub const NO_TICK_FALLBACK: Duration = Duration::from_millis(100);

impl WindowShared {
    pub fn new(width: u32, height: u32, scale: f64) -> Self {
        WindowShared {
            pending: Mutex::new(Pending::new(width, height, scale)),
            wake: Condvar::new(),
        }
    }

    /// Poisoning cannot happen unless a holder panicked, and the main thread's
    /// handlers must not panic (a panic unwinding into AppKit is fatal). Taking
    /// the data anyway keeps a worker bug from turning into a main-thread one.
    pub fn lock(&self) -> MutexGuard<'_, Pending> {
        self.pending.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Called by the main thread after changing `Pending`.
    pub fn notify(&self) {
        self.wake.notify_all();
    }

    pub fn tick(&self, link_timestamp: f64, link_duration: f64, now_media_time: f64) {
        let mut p = self.lock();
        p.tick_seq += 1;
        p.tick_at = Some(Instant::now());
        p.link_timestamp = link_timestamp;
        p.link_duration = link_duration;
        if p.callback_lag.len() < 100_000 {
            p.callback_lag.push(now_media_time - link_timestamp);
        }
        drop(p);
        self.wake.notify_all();
    }

    /// Blocks until the next display refresh (or the fallback timeout), then
    /// hands over everything accumulated since the previous call.
    pub fn next_frame(&self, cursor: &mut FrameCursor) -> Result<Frame, AppError> {
        let mut p = self.lock();
        let was_waiting = p.tick_seq == cursor.seen_seq;
        let started = Instant::now();
        while p.tick_seq == cursor.seen_seq && p.window_open {
            let left = NO_TICK_FALLBACK.saturating_sub(started.elapsed());
            if left.is_zero() {
                break;
            }
            p.worker_waiting = true;
            let (guard, _) = self
                .wake
                .wait_timeout(p, left)
                .unwrap_or_else(PoisonError::into_inner);
            p = guard;
        }
        p.worker_waiting = false;
        if !p.window_open {
            return Err(AppError::WindowGone);
        }
        let now = Instant::now();
        let ticks = p.tick_seq - cursor.seen_seq;
        let dt_ms = if ticks > 0 && cursor.last_link_ts > 0.0 {
            (p.link_timestamp - cursor.last_link_ts) * 1000.0
        } else {
            now.duration_since(cursor.last_return).as_secs_f64() * 1000.0
        };
        let frame = Frame {
            ticks,
            dt_ms,
            width: p.width,
            height: p.height,
            scale: p.scale,
            input: Input {
                held: p.held.iter().copied().collect(),
                mouse_dx: std::mem::take(&mut p.mouse_dx),
                mouse_dy: std::mem::take(&mut p.mouse_dy),
                scroll: std::mem::take(&mut p.scroll),
                mouse_events_summed: std::mem::take(&mut p.mouse_events),
            },
            events: std::mem::take(&mut p.events),
            tick_to_return: if ticks > 0 {
                p.tick_at.map(|t| now.duration_since(t))
            } else {
                None
            },
            worker_was_waiting: was_waiting,
        };
        cursor.seen_seq = p.tick_seq;
        if ticks > 0 {
            cursor.last_link_ts = p.link_timestamp;
        }
        cursor.last_return = now;
        Ok(frame)
    }

    pub fn take_callback_lag(&self) -> Vec<f64> {
        std::mem::take(&mut self.lock().callback_lag)
    }
}

/// mean, p50, p99, max over a sample, in the sample's unit.
pub struct Summary {
    pub n: usize,
    pub mean: f64,
    pub p50: f64,
    pub p99: f64,
    pub max: f64,
}

impl Summary {
    pub fn of(samples: &[f64]) -> Option<Summary> {
        if samples.is_empty() {
            return None;
        }
        let mut s = samples.to_vec();
        s.sort_by(f64::total_cmp);
        let pick = |q: f64| s[((s.len() - 1) as f64 * q).round() as usize];
        Some(Summary {
            n: s.len(),
            mean: s.iter().sum::<f64>() / s.len() as f64,
            p50: pick(0.5),
            p99: pick(0.99),
            max: s[s.len() - 1],
        })
    }
}

impl std::fmt::Display for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "n={} mean={:.3} p50={:.3} p99={:.3} max={:.3}",
            self.n, self.mean, self.p50, self.p99, self.max
        )
    }
}
