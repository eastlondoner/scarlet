//! Synthetic input for `--auto`, built and delivered on the main thread.
//!
//! No accessibility permission is needed for any of this: the events are
//! NSEvents handed to our own app, either queued (`-[NSApplication
//! postEvent:atStart:]`, the full path through `nextEvent`, local monitors,
//! key-equivalent matching and `sendEvent:`) or dispatched synchronously to
//! the window (`-[NSWindow sendEvent:]`, which skips the app-level routing
//! but still reaches the view's handlers and the input method). What neither
//! exercises is the path from real hardware (HID) into the window server.

use objc2::MainThreadMarker;
use objc2::rc::Retained;
use objc2_app_kit::{NSApplication, NSEvent, NSEventModifierFlags, NSEventType};
use objc2_core_foundation::CGPoint;
use objc2_core_graphics::{CGEvent, CGEventField, CGEventType, CGMouseButton, CGScrollEventUnit};
use objc2_foundation::{NSPoint, NSProcessInfo, NSString};

use crate::appkit::WindowEntry;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// `NSApp.postEvent` — queued, processed by the run loop like real input.
    AppQueue,
    /// `window.sendEvent` — synchronous, within the calling main-queue job.
    Window,
}

pub fn deliver(mtm: MainThreadMarker, w: &WindowEntry, ev: &NSEvent, route: Route) {
    match route {
        Route::AppQueue => NSApplication::sharedApplication(mtm).postEvent_atStart(ev, false),
        Route::Window => w.window.sendEvent(ev),
    }
}

pub fn key(
    w: &WindowEntry,
    keycode: u16,
    down: bool,
    chars: &str,
    cmd: bool,
) -> Option<Retained<NSEvent>> {
    let flags = if cmd {
        NSEventModifierFlags::Command
    } else {
        NSEventModifierFlags::empty()
    };
    let s = NSString::from_str(chars);
    NSEvent::keyEventWithType_location_modifierFlags_timestamp_windowNumber_context_characters_charactersIgnoringModifiers_isARepeat_keyCode(
        if down { NSEventType::KeyDown } else { NSEventType::KeyUp },
        NSPoint::ZERO,
        flags,
        NSProcessInfo::processInfo().systemUptime(),
        w.window.windowNumber(),
        None,
        &s,
        &s,
        false,
        keycode,
    )
}

pub fn mouse_button(w: &WindowEntry, down: bool) -> Option<Retained<NSEvent>> {
    let f = w.view.frame();
    NSEvent::mouseEventWithType_location_modifierFlags_timestamp_windowNumber_context_eventNumber_clickCount_pressure(
        if down { NSEventType::LeftMouseDown } else { NSEventType::LeftMouseUp },
        NSPoint::new(f.size.width / 2.0, f.size.height / 2.0),
        NSEventModifierFlags::empty(),
        NSProcessInfo::processInfo().systemUptime(),
        w.window.windowNumber(),
        None,
        0,
        1,
        if down { 1.0 } else { 0.0 },
    )
}

/// The window's centre in CoreGraphics global coordinates (top-left origin).
pub fn window_centre_cg(w: &WindowEntry) -> CGPoint {
    let f = w.window.frame();
    let primary_h = w.window.screen().map_or(0.0, |s| s.frame().size.height);
    CGPoint::new(
        f.origin.x + f.size.width / 2.0,
        primary_h - (f.origin.y + f.size.height / 2.0),
    )
}

/// A mouse-moved event carrying a relative delta, built as a CGEvent (NSEvent
/// has no constructor that sets deltaX/deltaY) and wrapped as an NSEvent.
pub fn mouse_move(w: &WindowEntry, dx: i64, dy: i64) -> Option<Retained<NSEvent>> {
    let cg = CGEvent::new_mouse_event(
        None,
        CGEventType::MouseMoved,
        window_centre_cg(w),
        CGMouseButton::Left,
    )?;
    CGEvent::set_integer_value_field(Some(&cg), CGEventField::MouseEventDeltaX, dx);
    CGEvent::set_integer_value_field(Some(&cg), CGEventField::MouseEventDeltaY, dy);
    NSEvent::eventWithCGEvent(&cg)
}

pub fn scroll_lines(w: &WindowEntry, lines: i32) -> Option<Retained<NSEvent>> {
    let cg = CGEvent::new_scroll_wheel_event2(None, CGScrollEventUnit::Line, 1, lines, 0, 0)?;
    // An in-process CGEvent converts to an NSEvent with window number 0 whose
    // locationInWindow is the flipped global location, and setting the
    // window-under-pointer fields does not change that. `-[NSWindow
    // sendEvent:]` hit-tests scroll events by locationInWindow, so place the
    // global point where that flip lands inside the view.
    let f = w.view.frame();
    let primary_h = w.window.screen().map_or(0.0, |s| s.frame().size.height);
    CGEvent::set_location(
        Some(&cg),
        CGPoint::new(f.size.width / 2.0, primary_h - f.size.height / 2.0),
    );
    NSEvent::eventWithCGEvent(&cg)
}

/// Posts a real CGEvent to this process (`CGEventPostToPid`), the route a
/// test harness without an NSApp handle would use.
pub fn cg_key_to_self(keycode: u16, down: bool) -> bool {
    let Some(cg) = CGEvent::new_keyboard_event(None, keycode, down) else {
        return false;
    };
    CGEvent::post_to_pid(std::process::id() as i32, Some(&cg));
    true
}
