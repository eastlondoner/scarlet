//! The main thread: NSApplication, the window, its view, and the display link.
//!
//! Everything in this file runs on the OS main thread. Every Objective-C
//! callback here must not panic: a panic unwinding out of a `define_class!`
//! method into AppKit's frames is at best an abort. So handlers only touch
//! the `WindowShared` mutex (poison-tolerant) and plain data.

use std::cell::{Cell, RefCell};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObject, NSObjectProtocol, ProtocolObject, Sel};
use objc2::{
    AnyThread, DefinedClass, MainThreadMarker, MainThreadOnly, define_class, msg_send, sel,
};
use objc2_app_kit::{
    NSApplication, NSApplicationDelegate, NSApplicationTerminateReply, NSBackingStoreType,
    NSCursor, NSEvent, NSEventMask, NSEventModifierFlags, NSMenu, NSMenuItem, NSResponder,
    NSTextInputClient, NSView, NSWindow, NSWindowDelegate, NSWindowStyleMask,
};
use objc2_core_foundation::{CGPoint, CGSize};
use objc2_core_graphics::{
    CGAssociateMouseAndMouseCursorPosition, CGError, CGWarpMouseCursorPosition,
};
use objc2_foundation::{
    NSArray, NSAttributedString, NSAttributedStringKey, NSNotFound, NSNotification, NSPoint,
    NSRange, NSRangePointer, NSRect, NSRunLoop, NSRunLoopCommonModes, NSSize, NSString, NSUInteger,
};
use objc2_quartz_core::{
    CACurrentMediaTime, CADisplayLink, CAMetalDisplayLink, CAMetalDisplayLinkDelegate,
    CAMetalDisplayLinkUpdate, CAMetalDrawable, CAMetalLayer,
};

use crate::model::{CloseCause, Event, WindowShared};

// ---------------------------------------------------------------------------
// Display-link target.
//
// A separate class from the view, and not main-thread-only, so the link can be
// added to either the main run loop or a dedicated thread's run loop. Its only
// state is an `Arc<WindowShared>`, which is `Sync`.

pub struct TickIvars {
    shared: Arc<WindowShared>,
}

define_class!(
    #[unsafe(super(NSObject))]
    #[name = "ScarletSpikeTickTarget"]
    #[ivars = TickIvars]
    pub struct TickTarget;

    impl TickTarget {
        #[unsafe(method(tick:))]
        fn tick(&self, link: &CADisplayLink) {
            let now = CACurrentMediaTime();
            self.ivars().shared.tick(link.timestamp(), link.duration(), now);
        }
    }

    unsafe impl NSObjectProtocol for TickTarget {}
);

impl TickTarget {
    fn new(shared: Arc<WindowShared>) -> Retained<Self> {
        let this = Self::alloc().set_ivars(TickIvars { shared });
        // SAFETY: NSObject's designated initialiser, on a freshly allocated object.
        unsafe { msg_send![super(this), init] }
    }
}

/// The drawable a CAMetalDisplayLink tick handed over, waiting for the worker.
/// A newer tick replaces an unused one (dropping it returns it to the layer).
pub type DrawableSlot =
    Arc<Mutex<Option<AssertSend<Retained<ProtocolObject<dyn CAMetalDrawable>>>>>>;

pub struct MetalLinkIvars {
    shared: Arc<WindowShared>,
    slot: DrawableSlot,
    period: f64,
}

define_class!(
    #[unsafe(super(NSObject))]
    #[name = "ScarletSpikeMetalLinkTarget"]
    #[ivars = MetalLinkIvars]
    pub struct MetalLinkTarget;

    unsafe impl NSObjectProtocol for MetalLinkTarget {}

    unsafe impl CAMetalDisplayLinkDelegate for MetalLinkTarget {
        #[unsafe(method(metalDisplayLink:needsUpdate:))]
        fn needs_update(&self, _link: &CAMetalDisplayLink, update: &CAMetalDisplayLinkUpdate) {
            let now = CACurrentMediaTime();
            let iv = self.ivars();
            *iv.slot.lock().unwrap_or_else(PoisonError::into_inner) =
                Some(AssertSend(update.drawable()));
            // `targetTimestamp` is a deadline in the future (not a past vsync
            // like CADisplayLink's `timestamp`), so the recorded "lag" is
            // negative: it is headroom.
            iv.shared.tick(update.targetTimestamp(), iv.period, now);
        }
    }
);

impl MetalLinkTarget {
    fn new(shared: Arc<WindowShared>, slot: DrawableSlot, fps: f64) -> Retained<Self> {
        let period = if fps > 0.0 { 1.0 / fps } else { 1.0 / 60.0 };
        let this = Self::alloc().set_ivars(MetalLinkIvars {
            shared,
            slot,
            period,
        });
        // SAFETY: NSObject's designated initialiser.
        unsafe { msg_send![super(this), init] }
    }
}

// ---------------------------------------------------------------------------
// The view: input, IME, backing-scale changes. It is also the window delegate
// (focus, close requests), so capture state lives in one place.

pub struct ViewIvars {
    shared: Arc<WindowShared>,
    /// IME composition in progress (NSTextInputClient "marked text").
    marked: RefCell<String>,
    /// The program asked for cursor capture.
    capture_wanted: Cell<bool>,
    /// Capture is currently applied to the OS (it is suspended while unfocused).
    capture_applied: Cell<bool>,
}

define_class!(
    #[unsafe(super(NSView, NSResponder, NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "ScarletSpikeMetalView"]
    #[ivars = ViewIvars]
    pub struct SpikeView;

    impl SpikeView {
        #[unsafe(method(acceptsFirstResponder))]
        fn accepts_first_responder(&self) -> bool {
            true
        }

        #[unsafe(method(acceptsFirstMouse:))]
        fn accepts_first_mouse(&self, _event: Option<&NSEvent>) -> bool {
            true
        }

        #[unsafe(method(keyDown:))]
        fn key_down(&self, event: &NSEvent) {
            if !event.isARepeat() {
                self.ivars().shared.lock().key_down(event.keyCode());
                self.ivars().shared.notify();
            }
            // Route through the input method: this ends in
            // `insertText:replacementRange:` (plain typing) or
            // `setMarkedText:...` (an IME composing), below.
            let events = NSArray::from_slice(&[event]);
            self.interpretKeyEvents(&events);
        }

        #[unsafe(method(keyUp:))]
        fn key_up(&self, event: &NSEvent) {
            self.ivars().shared.lock().key_up(event.keyCode());
            self.ivars().shared.notify();
        }

        // Modifier keys never send keyDown/keyUp, only flagsChanged. Which
        // physical key changed is the event's keyCode; whether it is now down
        // is its device-dependent bit in modifierFlags.
        #[unsafe(method(flagsChanged:))]
        fn flags_changed(&self, event: &NSEvent) {
            let key = event.keyCode();
            let bits = event.modifierFlags().bits();
            let down = match modifier_device_bit(key) {
                Some(bit) => bits & bit != 0,
                // Caps Lock and fn: no device bit; report a press+release.
                None => {
                    let mut p = self.ivars().shared.lock();
                    p.key_down(key);
                    p.key_up(key);
                    drop(p);
                    self.ivars().shared.notify();
                    return;
                }
            };
            let mut p = self.ivars().shared.lock();
            if down { p.key_down(key) } else { p.key_up(key) }
            drop(p);
            self.ivars().shared.notify();
        }

        #[unsafe(method(mouseMoved:))]
        fn mouse_moved(&self, event: &NSEvent) {
            self.add_motion(event);
        }

        #[unsafe(method(mouseDragged:))]
        fn mouse_dragged(&self, event: &NSEvent) {
            self.add_motion(event);
        }

        #[unsafe(method(rightMouseDragged:))]
        fn right_mouse_dragged(&self, event: &NSEvent) {
            self.add_motion(event);
        }

        #[unsafe(method(otherMouseDragged:))]
        fn other_mouse_dragged(&self, event: &NSEvent) {
            self.add_motion(event);
        }

        #[unsafe(method(mouseDown:))]
        fn mouse_down(&self, _event: &NSEvent) {
            self.push(Event::MouseDown(0));
        }

        #[unsafe(method(mouseUp:))]
        fn mouse_up(&self, _event: &NSEvent) {
            self.push(Event::MouseUp(0));
        }

        #[unsafe(method(rightMouseDown:))]
        fn right_mouse_down(&self, _event: &NSEvent) {
            self.push(Event::MouseDown(1));
        }

        #[unsafe(method(rightMouseUp:))]
        fn right_mouse_up(&self, _event: &NSEvent) {
            self.push(Event::MouseUp(1));
        }

        #[unsafe(method(scrollWheel:))]
        fn scroll_wheel(&self, event: &NSEvent) {
            // Trackpads give precise (point) deltas, wheels give lines. The
            // real API has to pick one unit; this spike keeps lines, dividing
            // precise deltas by a nominal 10 points per line.
            let dy = event.scrollingDeltaY();
            let lines = if event.hasPreciseScrollingDeltas() { dy / 10.0 } else { dy };
            self.ivars().shared.lock().scroll += lines;
            self.ivars().shared.notify();
        }

        #[unsafe(method(setFrameSize:))]
        fn set_frame_size(&self, size: NSSize) {
            // SAFETY: forwarding the same message, with the same argument, to NSView.
            let _: () = unsafe { msg_send![super(self), setFrameSize: size] };
            self.publish_size();
        }

        #[unsafe(method(viewDidChangeBackingProperties))]
        fn did_change_backing(&self) {
            // SAFETY: forwarding the same message to NSView.
            let _: () = unsafe { msg_send![super(self), viewDidChangeBackingProperties] };
            self.publish_size();
        }
    }

    unsafe impl NSObjectProtocol for SpikeView {}

    unsafe impl NSWindowDelegate for SpikeView {
        #[unsafe(method(windowShouldClose:))]
        fn window_should_close(&self, _sender: &NSWindow) -> bool {
            // Close is a request. The program decides; the runtime never closes.
            self.push(Event::CloseRequested(CloseCause::CloseButton));
            false
        }

        #[unsafe(method(windowDidBecomeKey:))]
        fn window_did_become_key(&self, _n: &NSNotification) {
            self.push(Event::FocusGained);
            if self.ivars().capture_wanted.get() {
                self.apply_capture(true);
            }
        }

        #[unsafe(method(windowDidResignKey:))]
        fn window_did_resign_key(&self, _n: &NSNotification) {
            let mut p = self.ivars().shared.lock();
            p.release_all();
            p.events.push(Event::FocusLost);
            drop(p);
            self.ivars().shared.notify();
            // Give the user their cursor back while another app has focus.
            if self.ivars().capture_applied.get() {
                self.apply_capture(false);
            }
        }
    }

    // Implementing NSTextInputClient is what makes AppKit create an
    // NSTextInputContext for the view, which is what IMEs (Japanese, Chinese,
    // the accent popup, the emoji picker) talk to. Without it,
    // interpretKeyEvents: still calls insertText: for plain typing, but
    // composition never works.
    unsafe impl NSTextInputClient for SpikeView {
        #[unsafe(method(insertText:replacementRange:))]
        fn insert_text(&self, string: &AnyObject, _replacement: NSRange) {
            self.ivars().marked.borrow_mut().clear();
            if let Some(text) = any_to_string(string)
                && !text.is_empty()
            {
                self.push(Event::Text(text));
            }
        }

        // Arrow keys, Return, Backspace and so on arrive here as selectors.
        // Doing nothing is deliberate: NSResponder's default beeps, and the
        // program already has the key as a KeyDown.
        #[unsafe(method(doCommandBySelector:))]
        fn do_command_by_selector(&self, _selector: Sel) {}

        #[unsafe(method(setMarkedText:selectedRange:replacementRange:))]
        fn set_marked_text(&self, string: &AnyObject, _selected: NSRange, _replacement: NSRange) {
            *self.ivars().marked.borrow_mut() = any_to_string(string).unwrap_or_default();
        }

        #[unsafe(method(unmarkText))]
        fn unmark_text(&self) {
            self.ivars().marked.borrow_mut().clear();
        }

        #[unsafe(method(selectedRange))]
        fn selected_range(&self) -> NSRange {
            NSRange::new(NSNotFound as NSUInteger, 0)
        }

        #[unsafe(method(markedRange))]
        fn marked_range(&self) -> NSRange {
            let len = self.ivars().marked.borrow().encode_utf16().count();
            if len == 0 { NSRange::new(NSNotFound as NSUInteger, 0) } else { NSRange::new(0, len) }
        }

        #[unsafe(method(hasMarkedText))]
        fn has_marked_text(&self) -> bool {
            !self.ivars().marked.borrow().is_empty()
        }

        #[unsafe(method_id(attributedSubstringForProposedRange:actualRange:))]
        fn attributed_substring(&self, _range: NSRange, _actual: NSRangePointer) -> Option<Retained<NSAttributedString>> {
            None
        }

        #[unsafe(method_id(validAttributesForMarkedText))]
        fn valid_attributes(&self) -> Retained<NSArray<NSAttributedStringKey>> {
            NSArray::new()
        }

        // Where the IME's candidate window goes, in screen coordinates. A game
        // would put it at its chat line; the spike uses the window's bottom-left.
        #[unsafe(method(firstRectForCharacterRange:actualRange:))]
        fn first_rect(&self, _range: NSRange, _actual: NSRangePointer) -> NSRect {
            match self.window() {
                Some(w) => {
                    let f = w.frame();
                    NSRect::new(NSPoint::new(f.origin.x + 8.0, f.origin.y + 8.0), NSSize::new(1.0, 20.0))
                }
                None => NSRect::ZERO,
            }
        }

        #[unsafe(method(characterIndexForPoint:))]
        fn character_index(&self, _point: NSPoint) -> NSUInteger {
            NSNotFound as NSUInteger
        }
    }
);

impl SpikeView {
    fn new(mtm: MainThreadMarker, frame: NSRect, shared: Arc<WindowShared>) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(ViewIvars {
            shared,
            marked: RefCell::new(String::new()),
            capture_wanted: Cell::new(false),
            capture_applied: Cell::new(false),
        });
        // SAFETY: NSView's designated initialiser.
        unsafe { msg_send![super(this), initWithFrame: frame] }
    }

    fn push(&self, event: Event) {
        self.ivars().shared.lock().events.push(event);
        self.ivars().shared.notify();
    }

    fn add_motion(&self, event: &NSEvent) {
        let mut p = self.ivars().shared.lock();
        p.mouse_dx += event.deltaX();
        p.mouse_dy += event.deltaY();
        p.mouse_events += 1;
        drop(p);
        self.ivars().shared.notify();
    }

    /// Recomputes the pixel size from the view's bounds and the window's
    /// backing scale, and keeps the layer's contentsScale in step.
    fn publish_size(&self) {
        // Before the view is in a window there is no backing scale to apply,
        // and a size computed at scale 1 would surface as a spurious Resized.
        let Some(window) = self.window() else { return };
        let scale = window.backingScaleFactor();
        let b = self.bounds();
        let (w, h) = (
            (b.size.width * scale).round() as u32,
            (b.size.height * scale).round() as u32,
        );
        if let Some(layer) = self.layer() {
            layer.setContentsScale(scale);
        }
        let mut p = self.ivars().shared.lock();
        if (p.width, p.height, p.scale) != (w, h, scale) {
            p.resized(w, h, scale);
        }
        drop(p);
        self.ivars().shared.notify();
    }

    pub fn set_capture(&self, on: bool) -> CaptureResult {
        self.ivars().capture_wanted.set(on);
        let focused = self.window().is_some_and(|w| w.isKeyWindow());
        if on && !focused {
            // Applied when the window next becomes key.
            return CaptureResult {
                applied: false,
                cg_error: CGError::Success,
            };
        }
        let cg_error = self.apply_capture(on);
        CaptureResult {
            applied: on,
            cg_error,
        }
    }

    fn apply_capture(&self, on: bool) -> CGError {
        if self.ivars().capture_applied.get() == on {
            return CGError::Success;
        }
        self.ivars().capture_applied.set(on);
        // NSCursor hide/unhide is a counter: every hide needs exactly one
        // unhide, which is why `capture_applied` guards both directions.
        if on {
            NSCursor::hide();
            // Park the (invisible) cursor at the window's centre, so a click
            // cannot land outside it if capture is released later.
            if let Some(w) = self.window() {
                let f = w.frame();
                let primary_h = w.screen().map_or(0.0, |s| s.frame().size.height);
                let centre = CGPoint::new(
                    f.origin.x + f.size.width / 2.0,
                    primary_h - (f.origin.y + f.size.height / 2.0),
                );
                let _ = CGWarpMouseCursorPosition(centre);
            }
            CGAssociateMouseAndMouseCursorPosition(false)
        } else {
            NSCursor::unhide();
            CGAssociateMouseAndMouseCursorPosition(true)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CaptureResult {
    pub applied: bool,
    pub cg_error: CGError,
}

/// NX_DEVICE*KEYMASK bits from IOKit's `IOLLEvent.h`: which side's modifier.
fn modifier_device_bit(key: u16) -> Option<usize> {
    Some(match key {
        59 => 0x0001, // left control
        56 => 0x0002, // left shift
        60 => 0x0004, // right shift
        55 => 0x0008, // left command
        54 => 0x0010, // right command
        58 => 0x0020, // left option
        61 => 0x0040, // right option
        62 => 0x2000, // right control
        _ => return None,
    })
}

fn any_to_string(obj: &AnyObject) -> Option<String> {
    if let Some(s) = obj.downcast_ref::<NSAttributedString>() {
        return Some(s.string().to_string());
    }
    obj.downcast_ref::<NSString>().map(|s| s.to_string())
}

// ---------------------------------------------------------------------------
// The application delegate: Quit is a request.

/// Set by the delegate; read for the report.
pub static ACTIVE_AFTER_LAUNCH: AtomicBool = AtomicBool::new(false);
pub static COOPERATIVE_ACTIVATE: AtomicBool = AtomicBool::new(false);

define_class!(
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "ScarletSpikeAppDelegate"]
    pub struct AppDelegate;

    unsafe impl NSObjectProtocol for AppDelegate {}

    unsafe impl NSApplicationDelegate for AppDelegate {
        #[unsafe(method(applicationDidFinishLaunching:))]
        fn did_finish_launching(&self, _n: &NSNotification) {
            let app = NSApplication::sharedApplication(self.mtm());
            // macOS 14's cooperative `activate` only succeeds if the active
            // app yields, which a terminal does not: measured 0/3 activations
            // from a background launch, against 3/3 for the deprecated call.
            if COOPERATIVE_ACTIVATE.load(Ordering::Relaxed) {
                app.activate();
            } else {
                #[allow(deprecated)]
                app.activateIgnoringOtherApps(true);
            }
            ACTIVE_AFTER_LAUNCH.store(app.isActive(), Ordering::Relaxed);
        }

        // Cmd-Q, the menu's Quit, the Dock's Quit. The program decides, so
        // this cancels and tells every window. (Logout also lands here, and a
        // Cancel there interrupts the logout; see README.)
        #[unsafe(method(applicationShouldTerminate:))]
        fn should_terminate(&self, _sender: &NSApplication) -> NSApplicationTerminateReply {
            QUIT_REQUESTS.fetch_add(1, Ordering::Relaxed);
            WINDOWS.with_borrow(|ws| {
                for w in ws {
                    w.view.push(Event::CloseRequested(CloseCause::Quit));
                }
            });
            NSApplicationTerminateReply::TerminateCancel
        }

        #[unsafe(method(applicationShouldTerminateAfterLastWindowClosed:))]
        fn terminate_after_last_window(&self, _sender: &NSApplication) -> bool {
            false
        }
    }
);

/// NSApplicationActivationPolicy before we set it (0 Regular, 1 Accessory, 2 Prohibited).
pub static INITIAL_POLICY: AtomicU64 = AtomicU64::new(99);
pub static QUIT_REQUESTS: AtomicU64 = AtomicU64::new(0);

impl AppDelegate {
    pub fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(());
        // SAFETY: NSObject's designated initialiser.
        unsafe { msg_send![super(this), init] }
    }
}

/// App menu with Quit (Cmd-Q) and a Window menu with Close (Cmd-W). Without a
/// main menu an unbundled binary has no Cmd-Q at all.
pub fn install_menu(mtm: MainThreadMarker, app: &NSApplication) {
    let main = NSMenu::new(mtm);
    let app_item = NSMenuItem::new(mtm);
    let app_menu = NSMenu::new(mtm);
    // SAFETY: `terminate:` is NSApplication's action; target nil sends it up
    // the responder chain to NSApp.
    let quit = unsafe {
        NSMenuItem::initWithTitle_action_keyEquivalent(
            NSMenuItem::alloc(mtm),
            &NSString::from_str("Quit"),
            Some(sel!(terminate:)),
            &NSString::from_str("q"),
        )
    };
    app_menu.addItem(&quit);
    app_item.setSubmenu(Some(&app_menu));
    main.addItem(&app_item);

    let win_item = NSMenuItem::new(mtm);
    let win_menu = NSMenu::initWithTitle(NSMenu::alloc(mtm), &NSString::from_str("Window"));
    // SAFETY: `performClose:` is NSWindow's; nil target reaches the key window.
    let close = unsafe {
        NSMenuItem::initWithTitle_action_keyEquivalent(
            NSMenuItem::alloc(mtm),
            &NSString::from_str("Close"),
            Some(sel!(performClose:)),
            &NSString::from_str("w"),
        )
    };
    win_menu.addItem(&close);
    win_item.setSubmenu(Some(&win_menu));
    main.addItem(&win_item);
    app.setMainMenu(Some(&main));
}

/// AppKit does not deliver keyUp: to the window while Command is held (the
/// key-down went to menu key-equivalent matching first). Without this monitor
/// a key released while Cmd is down stays in `held` forever.
pub fn install_cmd_keyup_monitor() -> Option<Retained<AnyObject>> {
    let block = RcBlock::new(|event: NonNull<NSEvent>| -> *mut NSEvent {
        // SAFETY: AppKit passes a valid event for the duration of the call.
        let e = unsafe { event.as_ref() };
        if e.modifierFlags().contains(NSEventModifierFlags::Command) {
            WINDOWS.with_borrow(|ws| {
                for w in ws {
                    if w.window.isKeyWindow() {
                        w.view.keyUp(e);
                    }
                }
            });
        }
        event.as_ptr()
    });
    // SAFETY: the block returns the event it was given, a valid pointer.
    unsafe { NSEvent::addLocalMonitorForEventsMatchingMask_handler(NSEventMask::KeyUp, &block) }
}

// ---------------------------------------------------------------------------
// The main thread's window table. Worker-side code holds an id, never an
// NSWindow: dropping a main-thread object from the worker would need the main
// thread, which may by then have stopped running (deadlock).

pub struct WindowEntry {
    pub id: u64,
    pub window: Retained<NSWindow>,
    pub view: Retained<SpikeView>,
    link: Link,
}

/// The second field keeps the link's target alive. CADisplayLink retains its
/// target (a cycle, broken by `invalidate`), but CAMetalDisplayLink's delegate
/// is weak: drop the target and the ticks silently stop.
#[allow(dead_code)]
enum Link {
    Display(Retained<CADisplayLink>, Retained<TickTarget>),
    Metal(Retained<CAMetalDisplayLink>, Retained<MetalLinkTarget>),
}

impl Link {
    fn invalidate(&self) {
        match self {
            Link::Display(l, _) => l.invalidate(),
            Link::Metal(l, _) => l.invalidate(),
        }
    }
}

#[derive(Clone, Copy)]
pub enum Pacing {
    DisplayLink,
    MetalDisplayLink,
}

thread_local! {
    pub static WINDOWS: RefCell<Vec<WindowEntry>> = const { RefCell::new(Vec::new()) };
}

static NEXT_WINDOW: AtomicU64 = AtomicU64::new(1);

/// A value that is not `Send` in objc2's types but is, in fact, safe to move:
/// used for CAMetalLayer (see README "Send/Sync") and a CADisplayLink handed
/// to its own run-loop thread.
pub struct AssertSend<T>(pub T);
// SAFETY: only instantiated for CAMetalLayer, whose nextDrawable and
// drawableSize are documented as usable off the main thread, and for a
// CADisplayLink moved once to the thread whose run loop then owns it.
unsafe impl<T> Send for AssertSend<T> {}

#[derive(Clone, Copy)]
pub enum LinkThread {
    Main,
    Dedicated,
}

#[derive(Clone, Copy)]
pub struct OpenSpec<'a> {
    pub title: &'a str,
    pub width: f64,
    pub height: f64,
    pub link_thread: LinkThread,
    pub pacing: Pacing,
}

pub struct Opened {
    pub id: u64,
    pub layer: AssertSend<Retained<CAMetalLayer>>,
    pub width: u32,
    pub height: u32,
    /// Present with `Pacing::MetalDisplayLink`: where each tick's drawable lands.
    pub slot: Option<DrawableSlot>,
}

pub fn open_window(
    mtm: MainThreadMarker,
    spec: &OpenSpec,
    shared_for: impl FnOnce(u32, u32, f64) -> Arc<WindowShared>,
    device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
) -> Opened {
    let OpenSpec {
        title,
        width,
        height,
        link_thread,
        pacing,
    } = *spec;
    let rect = NSRect::new(NSPoint::new(200.0, 200.0), NSSize::new(width, height));
    let style = NSWindowStyleMask::Titled
        | NSWindowStyleMask::Closable
        | NSWindowStyleMask::Miniaturizable
        | NSWindowStyleMask::Resizable;
    // SAFETY: plain initialiser with valid arguments.
    let window = unsafe {
        NSWindow::initWithContentRect_styleMask_backing_defer(
            NSWindow::alloc(mtm),
            rect,
            style,
            NSBackingStoreType::Buffered,
            false,
        )
    };
    // NSWindow defaults to releasing itself on close, which would free it
    // under the `Retained` we hold. objc2 requires turning that off.
    // SAFETY: we own the window through `Retained`.
    unsafe { window.setReleasedWhenClosed(false) };
    window.setTitle(&NSString::from_str(title));
    window.setAcceptsMouseMovedEvents(true);

    let scale = window.backingScaleFactor();
    let (pw, ph) = (
        (width * scale).round() as u32,
        (height * scale).round() as u32,
    );
    let shared = shared_for(pw, ph, scale);

    let view = SpikeView::new(mtm, rect, shared.clone());
    let layer = CAMetalLayer::new();
    layer.setDevice(Some(device));
    layer.setPixelFormat(objc2_metal::MTLPixelFormat::BGRA8Unorm);
    layer.setFramebufferOnly(true);
    layer.setContentsScale(scale);
    layer.setDrawableSize(CGSize::new(pw as f64, ph as f64));
    if let Some(n) = knob("SPIKE_MAX_DRAWABLES") {
        layer.setMaximumDrawableCount(n as usize);
    }
    // Layer-hosting view: setLayer before setWantsLayer.
    view.setLayer(Some(&layer));
    view.setWantsLayer(true);

    window.setContentView(Some(&view));
    window.setDelegate(Some(ProtocolObject::from_ref(&*view)));
    window.center();
    window.makeKeyAndOrderFront(None);
    window.makeFirstResponder(Some(&view));

    // Ask for the screen's full rate. Left at the default, a ProMotion panel
    // was measured alternating between runs at 60 Hz and 120 Hz presentation
    // while the link itself ticked at 120 Hz, with the worker then blocking in
    // nextDrawable (see README). SPIKE_LINK_FPS=0 keeps the default.
    let fps = knob("SPIKE_LINK_FPS").unwrap_or_else(|| {
        window
            .screen()
            .map_or(60.0, |s| s.maximumFramesPerSecond() as f64)
    });
    let range = objc2_quartz_core::CAFrameRateRange {
        minimum: fps as f32,
        maximum: fps as f32,
        preferred: fps as f32,
    };

    let (link, slot) = match pacing {
        Pacing::DisplayLink => {
            // CADisplayLink via NSView (macOS 14+): it follows the view's screen.
            let target = TickTarget::new(shared);
            // SAFETY: `target` implements `tick:` taking a CADisplayLink.
            let link = unsafe { view.displayLinkWithTarget_selector(&target, sel!(tick:)) };
            if fps > 0.0 {
                link.setPreferredFrameRateRange(range);
            }
            let obj: Retained<NSObject> = Retained::into_super(link.clone());
            schedule(obj, link_thread);
            (Link::Display(link, target), None)
        }
        Pacing::MetalDisplayLink => {
            // CAMetalDisplayLink (macOS 14+): the tick *carries* the drawable,
            // so the worker never blocks in nextDrawable.
            let slot: DrawableSlot = Arc::new(Mutex::new(None));
            let target = MetalLinkTarget::new(shared, slot.clone(), fps);
            let link = CAMetalDisplayLink::initWithMetalLayer(CAMetalDisplayLink::alloc(), &layer);
            link.setDelegate(Some(ProtocolObject::from_ref(&*target)));
            if fps > 0.0 {
                link.setPreferredFrameRateRange(range);
            }
            if let Some(n) = knob("SPIKE_FRAME_LATENCY") {
                link.setPreferredFrameLatency(n as f32);
            }
            let obj: Retained<NSObject> = Retained::into_super(link.clone());
            schedule(obj, link_thread);
            (Link::Metal(link, target), Some(slot))
        }
    };

    let id = NEXT_WINDOW.fetch_add(1, Ordering::Relaxed);
    WINDOWS.with_borrow_mut(|ws| {
        ws.push(WindowEntry {
            id,
            window: window.clone(),
            view: view.clone(),
            link,
        })
    });
    Opened {
        id,
        layer: AssertSend(layer),
        width: pw,
        height: ph,
        slot,
    }
}

/// Adds a display link (either kind; both answer `addToRunLoop:forMode:`) to
/// the main run loop or to a dedicated thread's. Common modes, so ticks keep
/// coming during live resize and menu tracking.
fn schedule(link: Retained<NSObject>, thread: LinkThread) {
    fn add(link: &NSObject, rl: &NSRunLoop) {
        // SAFETY: both CADisplayLink and CAMetalDisplayLink implement
        // addToRunLoop:forMode: with this signature; `rl` is the current
        // thread's run loop.
        let _: () = unsafe { msg_send![link, addToRunLoop: rl, forMode: NSRunLoopCommonModes] };
    }
    match thread {
        LinkThread::Main => add(&link, &NSRunLoop::mainRunLoop()),
        LinkThread::Dedicated => {
            let moved = AssertSend(link.clone());
            let spawned = std::thread::Builder::new()
                .name("display-link".into())
                .spawn(move || {
                    let link = moved;
                    let rl = NSRunLoop::currentRunLoop();
                    add(&link.0, &rl);
                    // Returns once the link is invalidated and no sources remain.
                    rl.run();
                });
            if spawned.is_err() {
                eprintln!("could not spawn display-link thread; using the main run loop");
                add(&link, &NSRunLoop::mainRunLoop());
            }
        }
    }
}

/// Experiment knobs, read from the environment (see README).
pub fn knob(name: &str) -> Option<f64> {
    std::env::var(name).ok()?.parse().ok()
}

pub fn with_window<R>(id: u64, f: impl FnOnce(&WindowEntry) -> R) -> Option<R> {
    WINDOWS.with_borrow(|ws| ws.iter().find(|w| w.id == id).map(f))
}

/// Removes a window: stops its display link (which retains its target), drops
/// capture, and closes it. Runs on the main thread.
pub fn close_window(id: u64) {
    let entry = WINDOWS.with_borrow_mut(|ws| {
        let at = ws.iter().position(|w| w.id == id)?;
        Some(ws.remove(at))
    });
    if let Some(e) = entry {
        e.link.invalidate();
        e.view.set_capture(false);
        e.window.setDelegate(None);
        e.window.orderOut(None);
        e.window.close();
        e.view.ivars().shared.lock().window_open = false;
        e.view.ivars().shared.notify();
    }
}

pub fn close_all() {
    let ids: Vec<u64> = WINDOWS.with_borrow(|ws| ws.iter().map(|w| w.id).collect());
    for id in ids {
        close_window(id);
    }
}

/// `-[NSApplication stop:]` only takes effect after the current event is
/// handled, so post an empty event to make `run` return now.
pub fn stop_app(mtm: MainThreadMarker) {
    let app = NSApplication::sharedApplication(mtm);
    app.stop(None);
    if let Some(ev) = NSEvent::otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2(
        objc2_app_kit::NSEventType::ApplicationDefined,
        NSPoint::ZERO,
        NSEventModifierFlags::empty(),
        0.0,
        0,
        None,
        0,
        0,
        0,
    ) {
        app.postEvent_atStart(&ev, true);
    }
}

pub fn app_is_active(mtm: MainThreadMarker) -> bool {
    NSApplication::sharedApplication(mtm).isActive()
}
