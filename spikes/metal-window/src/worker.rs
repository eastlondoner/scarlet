//! The VM thread's program: `--auto` (scripted checks and measurements) and
//! interactive mode. It only uses `app::Window` and `render::Renderer`, the
//! way a Scarlet program would only use `scarlet/app` and `scarlet/metal`.

use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use objc2::MainThreadOnly;
use objc2::msg_send;
use objc2::runtime::AnyObject;
use objc2_app_kit::{NSApplication, NSBackingStoreType, NSWindow, NSWindowStyleMask};
use objc2_foundation::{NSArray, NSOperatingSystemVersion, NSPoint, NSProcessInfo, NSRect, NSSize};

use crate::app::{Window, on_main, post_main};
use crate::appkit::{self, LinkThread, OpenSpec, Pacing, with_window};
use crate::model::{AppError, CloseCause, Event, Frame, Key, Summary};
use crate::render::Renderer;
use crate::synth::{self, Route};

const KEY_W: Key = 13;
const KEY_A: Key = 0;
const KEY_H: Key = 4;
const KEY_I: Key = 34;
const KEY_Q: Key = 12;
const KEY_C: Key = 8;
const KEY_T: Key = 17;
const KEY_ESC: Key = 53;
const KEY_F13: Key = 105;

pub struct Options {
    pub auto: bool,
    pub link_thread_dedicated: bool,
    pub metal_display_link: bool,
    pub pacing_frames: usize,
    /// Deliberately calls one AppKit getter from the VM thread, to prove the
    /// Main Thread Checker is loaded when it reports nothing else.
    pub mtc_selftest: bool,
    /// Panics on the VM thread after the window opens (shutdown-path test).
    pub panic_worker: bool,
}

/// Runs the program; the return value is the process exit code.
pub fn run(opts: Options) -> i32 {
    let renderer = match Renderer::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("metal: {e:?}");
            let _ = on_main(appkit::stop_app);
            return 2;
        }
    };
    let link = if opts.link_thread_dedicated {
        LinkThread::Dedicated
    } else {
        LinkThread::Main
    };
    let opened_at = Instant::now();
    let pacing = if opts.metal_display_link {
        Pacing::MetalDisplayLink
    } else {
        Pacing::DisplayLink
    };
    let spec = OpenSpec {
        title: "scarlet/app spike",
        width: 800.0,
        height: 500.0,
        link_thread: link,
        pacing,
    };
    let win = match Window::open(spec, &renderer.device) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("open: {e:?}");
            let _ = on_main(appkit::stop_app);
            return 2;
        }
    };
    let open_ms = opened_at.elapsed().as_secs_f64() * 1000.0;
    if opts.panic_worker {
        panic!("simulated VM-thread panic");
    }
    if opts.auto {
        let mut ctx = Ctx::new(win, renderer);
        ctx.report.info(
            "open",
            format!("app.open round trip (incl. waiting for NSApp.run to start): {open_ms:.1} ms"),
        );
        auto(&mut ctx, &opts);
        let code = if ctx.report.failures == 0 { 0 } else { 1 };
        ctx.report.print();
        let shutdown = Instant::now();
        let _ = ctx.win.close();
        let _ = on_main(appkit::stop_app);
        println!(
            "shutdown: close + stop round trip {:.2} ms; exiting with {code}",
            shutdown.elapsed().as_secs_f64() * 1000.0
        );
        code
    } else {
        let code = interactive(win, renderer);
        let _ = on_main(|mtm| {
            appkit::close_all();
            appkit::stop_app(mtm)
        });
        code
    }
}

// ---------------------------------------------------------------------------

#[derive(Default)]
struct Report {
    lines: Vec<String>,
    failures: u32,
}

impl Report {
    fn check(&mut self, name: &str, ok: bool, detail: impl Into<String>) {
        if !ok {
            self.failures += 1;
        }
        self.lines.push(format!(
            "[{}] {name}: {}",
            if ok { "PASS" } else { "FAIL" },
            detail.into()
        ));
    }
    fn info(&mut self, name: &str, detail: impl Into<String>) {
        self.lines.push(format!("[INFO] {name}: {}", detail.into()));
    }
    fn unrun(&mut self, name: &str, reason: impl Into<String>) {
        self.lines
            .push(format!("[UNRUN] {name}: {}", reason.into()));
    }
    fn print(&self) {
        println!("\n==== metal-window spike report ====");
        for l in &self.lines {
            println!("{l}");
        }
        println!("failures: {}", self.failures);
    }
}

/// What a run of frames observed, summed.
#[derive(Default, Debug)]
struct Acc {
    frames: usize,
    /// (frame index within this run, event)
    events: Vec<(usize, Event)>,
    held_union: BTreeSet<Key>,
    held_last: Vec<Key>,
    dx: f64,
    dy: f64,
    scroll: f64,
    motion_frames: usize,
    motion_events: u32,
    zero_tick_frames: usize,
    ticks: u64,
    size_mismatch: usize,
    last_size: (u32, u32),
}

impl Acc {
    fn has(&self, e: &Event) -> bool {
        self.events.iter().any(|(_, x)| x == e)
    }
    fn frame_of(&self, e: &Event) -> Option<usize> {
        self.events.iter().find(|(_, x)| x == e).map(|(i, _)| *i)
    }
}

struct Ctx {
    win: Window,
    r: Renderer,
    report: Report,
    intervals_ms: Vec<f64>,
    wake_ms: Vec<f64>,
    backlog_ms: Vec<f64>,
    drawable_wait_ms: Vec<f64>,
    dt_ms: Vec<f64>,
    missed_ticks: u64,
    last_return: Option<Instant>,
    render: bool,
}

impl Ctx {
    fn new(win: Window, r: Renderer) -> Self {
        Ctx {
            win,
            r,
            report: Report::default(),
            intervals_ms: Vec::new(),
            wake_ms: Vec::new(),
            backlog_ms: Vec::new(),
            drawable_wait_ms: Vec::new(),
            dt_ms: Vec::new(),
            missed_ticks: 0,
            last_return: None,
            render: true,
        }
    }

    fn reset_stats(&mut self) {
        self.intervals_ms.clear();
        self.wake_ms.clear();
        self.backlog_ms.clear();
        self.drawable_wait_ms.clear();
        self.dt_ms.clear();
        self.missed_ticks = 0;
        self.last_return = None;
        let _ = self.win.shared.take_callback_lag();
    }

    /// One frame: pull, then render (only when a tick came; a hidden window
    /// gets fallback frames with no tick, and nextDrawable there can block ~1 s).
    fn frame(&mut self) -> Result<(Frame, Option<(u32, u32)>), AppError> {
        let f = self.win.next_frame()?;
        let now = Instant::now();
        if let Some(prev) = self.last_return.replace(now) {
            self.intervals_ms
                .push(now.duration_since(prev).as_secs_f64() * 1000.0);
        }
        if let Some(d) = f.tick_to_return {
            let ms = d.as_secs_f64() * 1000.0;
            if f.worker_was_waiting {
                self.wake_ms.push(ms)
            } else {
                self.backlog_ms.push(ms)
            }
        }
        if f.ticks > 1 {
            self.missed_ticks += f.ticks - 1;
        }
        if f.ticks > 0 {
            self.dt_ms.push(f.dt_ms);
        }
        let mut tex = None;
        if self.render && f.ticks > 0 {
            match self.r.draw(self.win.drawable_source()) {
                Ok(Some(d)) => {
                    self.drawable_wait_ms
                        .push(d.drawable_wait.as_secs_f64() * 1000.0);
                    tex = Some(d.texture_size);
                }
                Ok(None) => {}
                Err(e) => eprintln!("draw: {e:?}"),
            }
        }
        Ok((f, tex))
    }

    /// Runs up to `max` frames, stopping early once `done` holds.
    fn pump(&mut self, max: usize, done: impl Fn(&Acc) -> bool) -> Acc {
        let mut acc = Acc::default();
        for i in 0..max {
            let Ok((f, tex)) = self.frame() else { break };
            acc.frames += 1;
            acc.ticks += f.ticks;
            if f.ticks == 0 {
                acc.zero_tick_frames += 1;
            }
            acc.events.extend(f.events.iter().cloned().map(|e| (i, e)));
            acc.held_union.extend(f.input.held.iter().copied());
            acc.held_last = f.input.held.clone();
            acc.dx += f.input.mouse_dx;
            acc.dy += f.input.mouse_dy;
            acc.scroll += f.input.scroll;
            if f.input.mouse_events_summed > 0 {
                acc.motion_frames += 1;
                acc.motion_events += f.input.mouse_events_summed;
            }
            if let Some(t) = tex
                && t != (f.width, f.height)
            {
                acc.size_mismatch += 1;
            }
            acc.last_size = (f.width, f.height);
            if done(&acc) {
                break;
            }
        }
        acc
    }

    /// Runs a closure against this window's entry on the main thread.
    fn main<R: Send>(
        &self,
        f: impl FnOnce(objc2::MainThreadMarker, &appkit::WindowEntry) -> R + Send,
    ) -> Option<R> {
        let id = self.win.id;
        on_main(move |mtm| with_window(id, |w| f(mtm, w)))
            .ok()
            .flatten()
    }
}

fn ms(s: &[f64]) -> String {
    Summary::of(s).map_or("no samples".into(), |s| format!("{s} ms"))
}

// ---------------------------------------------------------------------------

fn auto(ctx: &mut Ctx, opts: &Options) {
    if opts.mtc_selftest {
        // SAFETY: deliberately wrong (off the main thread); a read-only getter,
        // used only to make the Main Thread Checker speak.
        let mtm = unsafe { objc2::MainThreadMarker::new_unchecked() };
        let _ = NSApplication::sharedApplication(mtm).mainMenu();
        let _ = objc2_app_kit::NSView::new(mtm);
    }
    let pi = NSProcessInfo::processInfo();
    let v14 = pi.isOperatingSystemAtLeastVersion(NSOperatingSystemVersion {
        majorVersion: 14,
        minorVersion: 0,
        patchVersion: 0,
    });
    ctx.report.info(
        "os",
        format!(
            "{} ; CADisplayLink on NSView (macOS 14+) available: {v14}",
            pi.operatingSystemVersionString()
        ),
    );
    ctx.report.info(
        "link thread",
        if opts.link_thread_dedicated {
            "dedicated run-loop thread"
        } else {
            "main run loop (common modes)"
        },
    );
    ctx.report.info(
        "pacing",
        if opts.metal_display_link {
            "CAMetalDisplayLink (drawable delivered with the tick)"
        } else {
            "CADisplayLink + nextDrawable"
        },
    );
    ctx.report.info(
        "CGPreflightPostEventAccess",
        format!("{}", objc2_core_graphics::CGPreflightPostEventAccess()),
    );

    // Let the window come up and the first ticks arrive.
    ctx.pump(30, |_| false);
    let (active, key, scale, occl) = ctx
        .main(|mtm, w| {
            (
                appkit::app_is_active(mtm),
                w.window.isKeyWindow(),
                w.window.backingScaleFactor(),
                w.window.occlusionState().0,
            )
        })
        .unwrap_or((false, false, 0.0, 0));
    ctx.report.info(
        "activation",
        format!(
            "initial activationPolicy={} (0 Regular, 2 Prohibited) ; active right after launch={} ; after 30 frames: NSApp.isActive={active} window.isKeyWindow={key} backingScale={scale} occlusionState=0x{occl:x}",
            appkit::INITIAL_POLICY.load(std::sync::atomic::Ordering::Relaxed),
            appkit::ACTIVE_AFTER_LAUNCH.load(std::sync::atomic::Ordering::Relaxed)
        ),
    );
    let app_route = if active && key {
        Route::AppQueue
    } else {
        Route::Window
    };

    pacing(ctx, opts.pacing_frames);
    main_stall(ctx);
    round_trips(ctx);
    keys(ctx, app_route);
    mouse(ctx, app_route);
    cmd_keyup(ctx, active && key);
    resize(ctx);
    focus(ctx);
    capture(ctx);
    cg_post(ctx);
    occlusion(ctx);
    exception(ctx);
    close_and_quit(ctx);
}

fn pacing(ctx: &mut Ctx, n: usize) {
    ctx.reset_stats();
    let t = Instant::now();
    let acc = ctx.pump(n, |_| false);
    let secs = t.elapsed().as_secs_f64();
    let lag: Vec<f64> = ctx
        .win
        .shared
        .take_callback_lag()
        .iter()
        .map(|s| s * 1000.0)
        .collect();
    let dur = ctx.win.shared.lock().link_duration * 1000.0;
    ctx.report.info("pacing", format!("{} frames in {secs:.2} s = {:.1} fps; link.duration={dur:.3} ms; missed ticks={}; zero-tick frames={}", acc.frames, acc.frames as f64 / secs, ctx.missed_ticks, acc.zero_tick_frames));
    ctx.report.info(
        "frame interval (worker, next_frame return to return)",
        ms(&ctx.intervals_ms),
    );
    ctx.report
        .info("dt_ms (link timestamp deltas)", ms(&ctx.dt_ms));
    ctx.report.info(
        "tick -> next_frame return (worker already waiting)",
        ms(&ctx.wake_ms),
    );
    ctx.report.info(
        "tick -> next_frame return (tick arrived while worker busy)",
        ms(&ctx.backlog_ms),
    );
    ctx.report.info(
        "link.timestamp -> tick callback runs (main thread lag)",
        ms(&lag),
    );
    ctx.report
        .info("nextDrawable wait", ms(&ctx.drawable_wait_ms));
    let p99 = Summary::of(&ctx.intervals_ms).map_or(f64::MAX, |s| s.p99);
    ctx.report.check(
        "pacing p99 within 2 refreshes",
        p99 < dur * 2.0 + 1.0,
        format!("p99 {p99:.3} ms vs refresh {dur:.3} ms"),
    );
}

/// Blocks the main thread for 50 ms, three times, while frames run: shows
/// whether display ticks (and so the VM's frames) depend on the main thread.
fn main_stall(ctx: &mut Ctx) {
    ctx.reset_stats();
    ctx.pump(20, |_| false);
    for _ in 0..3 {
        post_main(|_| std::thread::sleep(Duration::from_millis(50)));
        ctx.pump(30, |_| false);
    }
    let max = ctx.intervals_ms.iter().copied().fold(0.0, f64::max);
    ctx.report.info(
        "main-thread 50 ms stalls x3",
        format!(
            "max frame interval {max:.2} ms; missed ticks {}; intervals {}",
            ctx.missed_ticks,
            ms(&ctx.intervals_ms)
        ),
    );
}

fn round_trips(ctx: &mut Ctx) {
    let mut noop = Vec::new();
    for _ in 0..200 {
        let t = Instant::now();
        let _ = on_main(|_| ());
        noop.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let mut title = Vec::new();
    for i in 0..200 {
        let t = Instant::now();
        let _ = ctx.win.set_title(&format!("scarlet/app spike {i}"));
        title.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    ctx.report.info("on_main no-op round trip", ms(&noop));
    ctx.report.info("on_main set_title round trip", ms(&title));
    // Round trips interleaved with frames (the realistic case: one per frame).
    let mut in_frame = Vec::new();
    for _ in 0..60 {
        ctx.pump(1, |_| false);
        let t = Instant::now();
        let _ = ctx.win.set_title("scarlet/app spike");
        in_frame.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    ctx.report
        .info("set_title round trip, once per frame", ms(&in_frame));
}

fn keys(ctx: &mut Ctx, app_route: Route) {
    // 1. Pressed and released between two frames, in one main-thread job, so
    //    no tick can fall between them.
    ctx.main(|_, w| {
        for down in [true, false] {
            if let Some(e) = synth::key(w, KEY_W, down, "w", false) {
                synth::deliver(w.window.mtm(), w, &e, Route::Window);
            }
        }
    });
    let acc = ctx.pump(30, |a| a.has(&Event::KeyUp(KEY_W)));
    let same = acc.frame_of(&Event::KeyDown(KEY_W)).is_some()
        && acc.frame_of(&Event::KeyDown(KEY_W)) == acc.frame_of(&Event::KeyUp(KEY_W));
    ctx.report.check(
        "key tap within one frame is not lost",
        same && !acc.held_union.contains(&KEY_W),
        format!(
            "events {:?}; held ever contained W: {}",
            acc.events,
            acc.held_union.contains(&KEY_W)
        ),
    );
    ctx.report.check(
        "text via interpretKeyEvents -> insertText",
        acc.has(&Event::Text("w".into())),
        "Text(\"w\") from the synthetic W",
    );

    // 2. The same through the app's event queue.
    ctx.main(move |mtm, w| {
        for down in [true, false] {
            if let Some(e) = synth::key(w, KEY_W, down, "w", false) {
                synth::deliver(mtm, w, &e, app_route);
            }
        }
    });
    let acc = ctx.pump(30, |a| a.has(&Event::KeyUp(KEY_W)));
    ctx.report.check(
        &format!("key tap via {app_route:?} arrives"),
        acc.has(&Event::KeyDown(KEY_W))
            && acc.has(&Event::KeyUp(KEY_W))
            && acc.held_last.is_empty(),
        format!("events {:?}", acc.events),
    );

    // 3. Held snapshot.
    ctx.main(|mtm, w| {
        if let Some(e) = synth::key(w, KEY_A, true, "a", false) {
            synth::deliver(mtm, w, &e, Route::Window);
        }
    });
    let acc = ctx.pump(10, |_| false);
    let held_every = acc.frames == 10 && acc.held_last == vec![KEY_A];
    ctx.main(|mtm, w| {
        if let Some(e) = synth::key(w, KEY_A, false, "a", false) {
            synth::deliver(mtm, w, &e, Route::Window);
        }
    });
    let after = ctx.pump(10, |a| a.has(&Event::KeyUp(KEY_A)));
    ctx.report.check(
        "held snapshot",
        held_every
            && acc
                .events
                .iter()
                .filter(|(_, e)| *e == Event::KeyDown(KEY_A))
                .count()
                == 1
            && after.held_last.is_empty(),
        format!(
            "held after 10 frames {:?}, one KeyDown; after release held {:?}",
            acc.held_last, after.held_last
        ),
    );

    // 4. Text for two keys typed in one frame.
    ctx.main(|mtm, w| {
        for (k, c) in [(KEY_H, "h"), (KEY_I, "i")] {
            for down in [true, false] {
                if let Some(e) = synth::key(w, k, down, c, false) {
                    synth::deliver(mtm, w, &e, Route::Window);
                }
            }
        }
    });
    let acc = ctx.pump(30, |a| a.has(&Event::KeyUp(KEY_I)));
    let text: String = acc
        .events
        .iter()
        .filter_map(|(_, e)| {
            if let Event::Text(t) = e {
                Some(t.as_str())
            } else {
                None
            }
        })
        .collect();
    ctx.report.check(
        "text order",
        text == "hi",
        format!("Text events joined = {text:?}"),
    );
}

fn mouse(ctx: &mut Ctx, app_route: Route) {
    ctx.pump(3, |_| false);
    // 100 moves in one main-thread job: must be one frame, summed.
    ctx.main(|mtm, w| {
        for _ in 0..100 {
            if let Some(e) = synth::mouse_move(w, 1, -2) {
                synth::deliver(mtm, w, &e, Route::Window);
            }
        }
    });
    let acc = ctx.pump(10, |a| a.motion_events >= 100);
    ctx.report.check(
        "mouse deltas summed (100 moves, one job)",
        acc.dx == 100.0 && acc.dy == -200.0 && acc.motion_frames == 1,
        format!(
            "dx={} dy={} over {} frame(s), {} raw events",
            acc.dx, acc.dy, acc.motion_frames, acc.motion_events
        ),
    );
    // The same through the event queue: spread over run-loop turns, still summed.
    ctx.main(move |mtm, w| {
        for _ in 0..100 {
            if let Some(e) = synth::mouse_move(w, 1, -2) {
                synth::deliver(mtm, w, &e, app_route);
            }
        }
    });
    let acc = ctx.pump(30, |a| a.motion_events >= 100);
    ctx.report.check(
        &format!("mouse deltas summed (100 moves via {app_route:?})"),
        acc.dx == 100.0 && acc.dy == -200.0,
        format!(
            "dx={} dy={} over {} frame(s), {} raw events",
            acc.dx, acc.dy, acc.motion_frames, acc.motion_events
        ),
    );
    ctx.main(|mtm, w| {
        if let Some(e) = synth::scroll_lines(w, -3) {
            synth::deliver(mtm, w, &e, Route::Window);
        }
    });
    let acc = ctx.pump(10, |a| a.scroll != 0.0);
    ctx.report.check(
        "scroll",
        acc.scroll != 0.0,
        format!("scroll total {} (3 lines down posted)", acc.scroll),
    );
    ctx.main(|mtm, w| {
        for down in [true, false] {
            if let Some(e) = synth::mouse_button(w, down) {
                synth::deliver(mtm, w, &e, Route::Window);
            }
        }
    });
    let acc = ctx.pump(10, |a| a.has(&Event::MouseUp(0)));
    ctx.report.check(
        "mouse button",
        acc.has(&Event::MouseDown(0)) && acc.has(&Event::MouseUp(0)),
        format!("{:?}", acc.events),
    );
}

/// A key released while Command is held: AppKit swallows its keyUp, the
/// local monitor recovers it. Only meaningful through the app queue.
fn cmd_keyup(ctx: &mut Ctx, can: bool) {
    if !can {
        ctx.report.unrun("Cmd+key keyUp recovery", "app is not active/key, so the app-queue route (where the swallowing happens) cannot be exercised");
        return;
    }
    ctx.main(|mtm, w| {
        for down in [true, false] {
            if let Some(e) = synth::key(w, KEY_T, down, "t", true) {
                synth::deliver(mtm, w, &e, Route::AppQueue);
            }
        }
    });
    let acc = ctx.pump(30, |a| a.has(&Event::KeyUp(KEY_T)));
    ctx.report.check(
        "Cmd+T keyUp recovered by local monitor",
        acc.has(&Event::KeyUp(KEY_T)) && acc.held_last.is_empty(),
        format!("{:?}", acc.events),
    );
}

fn resize(ctx: &mut Ctx) {
    let resizes_before = ctx.win.drawable_resizes;
    let scale = ctx.main(|_, w| {
        w.window.setContentSize(NSSize::new(640.0, 360.0));
        w.window.backingScaleFactor()
    });
    let Some(scale) = scale else {
        ctx.report.check("resize", false, "window gone");
        return;
    };
    let want = (
        (640.0 * scale).round() as u32,
        (360.0 * scale).round() as u32,
    );
    let acc = ctx.pump(30, |a| {
        a.events
            .iter()
            .any(|(_, e)| matches!(e, Event::Resized { .. }))
    });
    let after = ctx.pump(10, |_| false);
    let got = acc
        .events
        .iter()
        .filter(|(_, e)| matches!(e, Event::Resized { .. }))
        .count();
    ctx.report.check(
        "resize -> one Resized, pixel size = points x scale, drawables follow",
        acc.has(&Event::Resized { width: want.0, height: want.1 }) && got == 1 && after.size_mismatch == 0 && ctx.win.drawable_resizes == resizes_before + 1,
        format!("scale {scale}; want {want:?}; events {:?}; drawable/frame size mismatches in next 10 frames: {}", acc.events, after.size_mismatch),
    );
    // A burst of 20 size changes inside one main-thread job coalesces.
    ctx.main(|_, w| {
        for i in 0..20 {
            w.window
                .setContentSize(NSSize::new(640.0 + i as f64 * 5.0, 360.0));
        }
    });
    let acc = ctx.pump(10, |a| {
        a.events
            .iter()
            .any(|(_, e)| matches!(e, Event::Resized { .. }))
    });
    let n = acc
        .events
        .iter()
        .filter(|(_, e)| matches!(e, Event::Resized { .. }))
        .count();
    ctx.report.check(
        "20 resizes in one job coalesce to one Resized",
        n == 1,
        format!("{:?}", acc.events),
    );
    ctx.pump(3, |_| false);
}

fn focus(ctx: &mut Ctx) {
    // Hold A, then make another of our windows key: our window resigns key.
    ctx.main(|mtm, w| {
        if let Some(e) = synth::key(w, KEY_A, true, "a", false) {
            synth::deliver(mtm, w, &e, Route::Window);
        }
    });
    ctx.pump(3, |_| false);
    let helper = ctx.main(|mtm, _| {
        // SAFETY: plain initialiser.
        let other = unsafe {
            NSWindow::initWithContentRect_styleMask_backing_defer(
                NSWindow::alloc(mtm),
                NSRect::new(NSPoint::new(50.0, 50.0), NSSize::new(200.0, 100.0)),
                NSWindowStyleMask::Titled,
                NSBackingStoreType::Buffered,
                false,
            )
        };
        // SAFETY: owned via Retained.
        unsafe { other.setReleasedWhenClosed(false) };
        other.makeKeyAndOrderFront(None);
        // MainThreadBound: Send, but only openable (and dropped) on the main thread.
        dispatch2::MainThreadBound::new(other, mtm)
    });
    let lost = ctx.pump(30, |a| a.has(&Event::FocusLost));
    ctx.report.check(
        "FocusLost, and held keys released with it",
        lost.has(&Event::FocusLost) && lost.has(&Event::KeyUp(KEY_A)) && lost.held_last.is_empty(),
        format!("{:?}", lost.events),
    );
    ctx.main(|_, w| w.window.makeKeyAndOrderFront(None));
    let gained = ctx.pump(30, |a| a.has(&Event::FocusGained));
    ctx.report.check(
        "FocusGained",
        gained.has(&Event::FocusGained),
        format!("{:?}", gained.events),
    );
    if let Some(h) = helper {
        let _ = on_main(move |mtm| {
            let h = h;
            h.get(mtm).orderOut(None);
            h.get(mtm).close();
        });
    }
}

fn capture(ctx: &mut Ctx) {
    match ctx.win.capture_cursor(true) {
        Ok(r) => ctx.report.check(
            "capture_cursor(true)",
            r.cg_error == objc2_core_graphics::CGError::Success,
            format!(
                "applied={} CGAssociateMouseAndMouseCursorPosition -> {:?}",
                r.applied, r.cg_error
            ),
        ),
        Err(e) => ctx
            .report
            .check("capture_cursor(true)", false, format!("{e:?}")),
    }
    ctx.main(|mtm, w| {
        for _ in 0..10 {
            if let Some(e) = synth::mouse_move(w, 5, 3) {
                synth::deliver(mtm, w, &e, Route::Window);
            }
        }
    });
    let acc = ctx.pump(10, |a| a.motion_events >= 10);
    ctx.report.info(
        "deltas while captured (synthetic)",
        format!("dx={} dy={}", acc.dx, acc.dy),
    );
    match ctx.win.capture_cursor(false) {
        Ok(r) => ctx.report.check(
            "capture_cursor(false)",
            r.cg_error == objc2_core_graphics::CGError::Success,
            format!("{:?}", r.cg_error),
        ),
        Err(e) => ctx
            .report
            .check("capture_cursor(false)", false, format!("{e:?}")),
    }
}

/// A real CGEvent posted to our own pid: does it arrive without permission?
fn cg_post(ctx: &mut Ctx) {
    let _ = on_main(|_| {
        synth::cg_key_to_self(KEY_F13, true);
        synth::cg_key_to_self(KEY_F13, false);
    });
    let acc = ctx.pump(30, |a| a.has(&Event::KeyUp(KEY_F13)));
    ctx.report.info(
        "CGEventPostToPid(self) F13",
        format!(
            "arrived: down={} up={} (events {:?})",
            acc.has(&Event::KeyDown(KEY_F13)),
            acc.has(&Event::KeyUp(KEY_F13)),
            acc.events
        ),
    );
}

fn occlusion(ctx: &mut Ctx) {
    ctx.main(|_, w| w.window.orderOut(None));
    let t = Instant::now();
    let mut acc = Acc::default();
    while t.elapsed() < Duration::from_millis(1000) {
        let a = ctx.pump(1, |_| false);
        acc.frames += a.frames;
        acc.ticks += a.ticks;
        acc.zero_tick_frames += a.zero_tick_frames;
    }
    ctx.main(|_, w| w.window.makeKeyAndOrderFront(None));
    ctx.pump(10, |_| false);
    ctx.report.info(
        "window ordered out for 1 s",
        format!(
            "{} next_frame returns, {} display ticks, {} fallback (no-tick) frames",
            acc.frames, acc.ticks, acc.zero_tick_frames
        ),
    );
}

fn exception(ctx: &mut Ctx) {
    let r = on_main(|_| {
        let empty = NSArray::<AnyObject>::new();
        // SAFETY (deliberately raising): objectAtIndex: out of range throws
        // NSRangeException; the boundary must turn it into a value.
        let _: *mut AnyObject = unsafe { msg_send![&*empty, objectAtIndex: 5usize] };
    });
    let alive = on_main(|_| true).unwrap_or(false);
    ctx.report.check(
        "ObjC exception on main-queue job becomes a value",
        matches!(r, Err(AppError::Exception(_))) && alive,
        format!("{r:?}; main thread still serving jobs: {alive}"),
    );
}

fn close_and_quit(ctx: &mut Ctx) {
    ctx.main(|_, w| w.window.performClose(None));
    let acc = ctx.pump(30, |a| {
        a.has(&Event::CloseRequested(CloseCause::CloseButton))
    });
    let visible = ctx.main(|_, w| w.window.isVisible()).unwrap_or(false);
    ctx.report.check(
        "close button -> CloseRequested, window stays until the program decides",
        acc.has(&Event::CloseRequested(CloseCause::CloseButton)) && visible,
        format!("events {:?}; still visible {visible}", acc.events),
    );

    // Cmd-Q through the real menu: key-equivalent matching finds Quit, which
    // sends terminate:, which asks the delegate.
    let matched = ctx.main(|mtm, w| {
        let app = NSApplication::sharedApplication(mtm);
        let ev = synth::key(w, KEY_Q, true, "q", true)?;
        Some(app.mainMenu()?.performKeyEquivalent(&ev))
    });
    let acc = ctx.pump(30, |a| a.has(&Event::CloseRequested(CloseCause::Quit)));
    ctx.report.check(
        "Cmd-Q -> CloseRequested(Quit), process keeps running",
        acc.has(&Event::CloseRequested(CloseCause::Quit)),
        format!(
            "menu matched Cmd-Q: {matched:?}; terminate requests seen by delegate: {}; events {:?}",
            appkit::QUIT_REQUESTS.load(std::sync::atomic::Ordering::Relaxed),
            acc.events
        ),
    );
}

// ---------------------------------------------------------------------------

fn interactive(win: Window, r: Renderer) -> i32 {
    let mut ctx = Ctx::new(win, r);
    println!(
        "interactive: C toggles cursor capture, Esc releases it, T sets the title; close or Cmd-Q twice within 2 s to quit."
    );
    let mut captured = false;
    let mut last_close: Option<Instant> = None;
    let mut window_start = Instant::now();
    let mut frames = 0u32;
    let (mut dx, mut dy, mut scroll) = (0.0, 0.0, 0.0);
    loop {
        let (f, _) = match ctx.frame() {
            Ok(x) => x,
            Err(e) => {
                println!("next_frame: {e:?}");
                return 0;
            }
        };
        frames += 1;
        dx += f.input.mouse_dx;
        dy += f.input.mouse_dy;
        scroll += f.input.scroll;
        for e in &f.events {
            println!("event: {e:?}");
            match e {
                Event::KeyDown(KEY_C) => {
                    captured = !captured;
                    println!(
                        "capture_cursor({captured}) -> {:?}",
                        ctx.win.capture_cursor(captured)
                    );
                }
                Event::KeyDown(KEY_ESC) if captured => {
                    captured = false;
                    println!(
                        "capture_cursor(false) -> {:?}",
                        ctx.win.capture_cursor(false)
                    );
                }
                Event::KeyDown(KEY_T) => {
                    let t = Instant::now();
                    let r = ctx.win.set_title(&format!("spike {:?}", Instant::now()));
                    println!("set_title -> {r:?} in {:?}", t.elapsed());
                }
                Event::CloseRequested(cause) => {
                    if last_close.is_some_and(|t| t.elapsed() < Duration::from_secs(2)) {
                        println!("{cause:?} confirmed: exiting");
                        return 0;
                    }
                    println!("{cause:?}: the program declines. Request again within 2 s to quit.");
                    last_close = Some(Instant::now());
                }
                _ => {}
            }
        }
        if window_start.elapsed() >= Duration::from_secs(1) {
            println!(
                "{frames} fps  {}x{} @{}  held {:?}  mouse ({dx:.1}, {dy:.1})  scroll {scroll:.2}  frame p99 {}",
                f.width,
                f.height,
                f.scale,
                f.input.held,
                ms(&ctx.intervals_ms)
            );
            ctx.intervals_ms.clear();
            frames = 0;
            (dx, dy, scroll) = (0.0, 0.0, 0.0);
            window_start = Instant::now();
        }
    }
}
