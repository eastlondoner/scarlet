# Spike: the `scarlet/app` window, thread split and input model

This is a throwaway, standalone crate (its own `[workspace]`, not a member of the repo workspace). It tests the design in `docs/metal-design.md` ("The window, and input into the game") on real hardware before `scarlet_metal` and `scarlet/app` are written. It uses objc2 0.6.4, objc2-{foundation,app-kit,metal,quartz-core,core-graphics,core-foundation} 0.3.2, dispatch2 0.3.1 and block2 0.6.2. AppKit is called directly, with no winit.

Measured on an Apple M1 Max with the built-in Liquid Retina XDR display (ProMotion, 120 Hz, backing scale 2), macOS 26.3 (25D125), Rust 1.97.1. The machine was on AC power with `powermode 1` (Low Power Mode) set for AC.

## What was built

| File | Thread | What it is |
|---|---|---|
| `src/model.rs` | both | The handoff, in plain Rust with no Apple types: `Pending` (one mutex) and a `Condvar`, plus `Frame`/`Input`/`Event` and `next_frame`. This is the shape a `Platform` would hold per window. |
| `src/appkit.rs` | main | NSApplication delegate (Quit is a request), the menu, the view (input, `NSTextInputClient`, backing scale, window delegate), the display-link targets, the window table, and cursor capture. |
| `src/app.rs` | VM | The stand-ins for the intrinsics: `on_main` (sync to the main queue, catching exceptions and panics), `Window::{open, next_frame, set_title, capture_cursor, close}`, and `StopAppOnDrop`. |
| `src/render.rs` | VM | Metal: an MSL library, a pipeline, and per frame a clear colour that changes with time plus a rotating triangle, then present and commit. |
| `src/synth.rs` | main | Synthetic NSEvents and CGEvents for `--auto`. |
| `src/worker.rs` | VM | The "program": the `--auto` checks and measurements, and interactive mode. |

The main thread runs `NSApplication` and nothing else. The `vm` thread starts before `[NSApp run]`. Its first request (`Window::open`, a `dispatch_sync` to the main queue) waits until `run` starts draining the queue. From then on it loops on a blocking `next_frame()`, renders, and calls `nextDrawable`/`presentDrawable`/`commit` itself.

## How to run

```sh
cd spikes/metal-window
cargo build --release
./target/release/metal-window-spike --auto     # scripted checks + numbers, exit 0 iff all PASS
./target/release/metal-window-spike            # interactive
```

Flags:

- `--pacing=metal-link` uses `CAMetalDisplayLink` instead of `CADisplayLink` + `nextDrawable`.
- `--link-thread=dedicated` puts the link on its own run-loop thread instead of the main run loop.
- `--cooperative-activate` uses macOS 14's `activate` instead of `activateIgnoringOtherApps:`.
- `--frames=N` sets the length of the pacing measurement.
- `--panic-worker` panics on the VM thread (a shutdown test).
- `--mtc-selftest` makes one deliberate off-main AppKit call, to prove the Main Thread Checker is loaded.

Experiment knobs (environment variables):

- `SPIKE_LINK_FPS` sets the preferred frame rate. The default is the screen's `maximumFramesPerSecond`; `0` leaves the system default.
- `SPIKE_MAX_DRAWABLES` sets `maximumDrawableCount`.
- `SPIKE_PRESENT_MIN_MS` switches to `presentDrawable:afterMinimumDuration:`.
- `SPIKE_FRAME_LATENCY` sets `CAMetalDisplayLink.preferredFrameLatency`.

Interactive keys: C toggles cursor capture, Esc releases it, and T sets the title. Close (or Cmd-Q) is declined once; a second request within 2 s quits. Once a second it prints fps, size, held keys, summed mouse and scroll.

Checkers, run and clean (see "Verified"):

```sh
DYLD_INSERT_LIBRARIES=/Applications/Xcode.app/Contents/Developer/usr/lib/libMainThreadChecker.dylib \
MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=assert ./target/debug/metal-window-spike --auto
```

Launch that directly. Through `/usr/bin/perl` or any other SIP-protected binary, the `DYLD_*` variable is silently stripped and the checker never loads.

`cargo clippy -- -D warnings` (debug and `--release`) is clean.

## Measured numbers

All numbers are from release builds, 600-frame pacing runs, three runs per configuration (15 runs in total), plus a final run of A and E. Times are in ms.

### Pacing

| Config | fps | frame interval p50 / p99 / max | tick → `next_frame` return, worker already waiting | `nextDrawable` wait, mean | 3 × 50 ms main-thread stall: worst frame interval |
|---|---|---|---|---|---|
| A: CADisplayLink, main run loop, rate = screen max (default) | 119.0–119.8 | 8.33 / 10.7–12.8 / 14–22 | p50 0.015–0.03. In 6–466 of 600 frames the worker was already waiting; in the rest the tick came while it was blocked in `nextDrawable`, 1.2–1.7 mean. | 2.4 – 8.1 | 51–54 |
| B: same, no rate set (`SPIKE_LINK_FPS=0`) | 119.6–120 | 8.33 / 11–13 / 13–25 | 0–600 of 600 frames waiting, varying by run | 0.46 – 8.05 | 52 |
| C: CADisplayLink on a dedicated thread | 119.4–119.9 | 8.33 / 10.7–12 / 13–29 | almost never waiting | 8.0 – 8.2 | **9.1 – 18.7** |
| D: CAMetalDisplayLink, main run loop | 120.0 | 8.33 / 8.4–13.3 / 10–15 | 600/600 waiting; mean 0.02–0.2, p50 0.012–0.022 | **0.000** | 51–54 |
| E: CAMetalDisplayLink on a dedicated thread | 117.6–120 | 8.33 / 9.0–16.6 / 11–25 | 600/600 waiting; mean 0.027–0.087, p50 0.012–0.025, p99 0.16–2.5 | **0.000** | **8.4 – 21** |

- **The link callback runs 0.52 ms (p50) after `CADisplayLink.timestamp` on the main run loop, and 0.05 ms on a dedicated thread.** The p99 is under 3 ms in both.
- **CAMetalDisplayLink calls back about 8.25 ms before its `targetTimestamp`.** That is one frame of headroom.
- **Frame pacing needs one of three things to stay at 120 Hz.** In three early runs with no frame-rate range set, the display link ticked at 120 Hz (duration 8.333) while presentation ran at 60 Hz. Every frame then "missed" one tick and waited about 16 ms in `nextDrawable`. With `SPIKE_LINK_FPS=120` it ran at 120 in 3 of 3 runs, and so did `SPIKE_PRESENT_MIN_MS=8.33` and `SPIKE_MAX_DRAWABLES=2`. The 60 Hz case did not come back in six later runs, with or without the range, so its cause is **not established**. Low Power Mode was on for AC, and ProMotion adapting its rate is the likely factor.
- **With CADisplayLink + `nextDrawable`, how long the worker blocks in `nextDrawable` depends on the phase between tick and drawable, and varies by run from 0.5 to 8 ms mean.** The input snapshot is taken before that wait, so the wait is added input latency and lost frame budget. CAMetalDisplayLink removed it completely in 9 of 9 runs.

### Main-queue round trips (`dispatch_sync` from the VM thread)

- no-op: p50 0.005–0.02, p99 0.02–0.13, max ≤ 0.6.
- `setTitle:` back to back: p50 0.17–0.25, p99 0.9–1.8, max ≤ 3.6.
- `setTitle:` once per frame (the realistic rate): p50 0.02–0.07, p99 0.05–0.8.
- `app.open`, including waiting for `NSApp.run` to start: 38–65.
- Shutdown (close window + `stop:`): 7–17.

### Hidden window (`orderOut:` for 1 s)

- **CADisplayLink stops completely: 0 ticks.** `next_frame` returns only through the 100 ms fallback timer (10 returns in 1 s).
- **CAMetalDisplayLink keeps ticking at about 33–35 per second,** plus 7–8 fallback frames.

## Verified (by `--auto` on this machine, all PASS)

- **The window becomes the active app.** It is a plain CLI binary with no bundle. The initial activation policy is **Prohibited (2)**. After `setActivationPolicy(Regular)` + `activateIgnoringOtherApps(true)`, `NSApp.isActive` and `isKeyWindow` are true 30 frames later, in every run.
- **Cmd-Q through the real main menu becomes `CloseRequested(Quit)`.** Key-equivalent matching finds the Quit item, it sends `terminate:`, and `applicationShouldTerminate:` returns Cancel. The process keeps running.
- **The close button (`performClose:`) becomes `CloseRequested(CloseButton)`.** The window stays visible.
- **The thread split holds.** A worker renders with Metal; `nextDrawable`, `setDrawableSize` and present all happen on the worker. The Main Thread Checker reported **nothing** in any configuration (A, C, E), and its self-test proves it was loaded (it did report `-[NSApplication mainMenu]` and `-[NSView init]` off the main thread). Metal API validation in assert mode also produced nothing.
- **A key pressed and released between two frames is not lost.** Both events, delivered inside one main-thread job, arrive in the same frame as `KeyDown`, `Text("w")`, `KeyUp`, and `held` never contains the key. It also works through the app's event queue.
- **`held` is a snapshot.** It shows the key in all 10 frames while held, with exactly one `KeyDown`; auto-repeat and a duplicate key-down are not new presses.
- **Text arrives through `interpretKeyEvents:` → `insertText:replacementRange:`** (`NSTextInputClient`), in typing order ("hi").
- **Mouse deltas are summed.** 100 moves sent in one job arrive as one frame with dx=100 and dy=-200, and `mouse_events_summed`=100. The same moves through the event queue still sum exactly, over 1–2 frames.
- **Scroll arrives** (a 3-line wheel event gives -3 lines). **Mouse buttons arrive** as `MouseDown`/`MouseUp`.
- **A key released while Cmd is held is recovered by a local `NSEvent` monitor.** AppKit swallows that `keyUp:` otherwise (see Findings).
- **Resize works at HiDPI.**
  - `setContentSize(640×360)` at scale 2 gives exactly one `Resized{1280,720}`.
  - The worker then sets `drawableSize`, and the next 10 drawables have 0 size mismatches with the frame.
  - 20 resizes inside one job coalesce to one `Resized`.
- **Focus works.**
  - Making another window key gives `FocusLost`, preceded by synthetic `KeyUp`s for every held key.
  - Making ours key again gives `FocusGained`.
- **Cursor capture calls succeed.** `CGAssociateMouseAndMouseCursorPosition(false)` + `NSCursor.hide` + warp to the centre returns `kCGErrorSuccess`, and so does the release. Synthetic deltas still flow while captured.
- **An Objective-C exception inside a main-queue job becomes a value.** It comes back as `Err(Exception("*** -[__NSArray0 objectAtIndex:]: index 5 beyond bounds…"))`, and the main thread keeps serving jobs.
- **Shutdown works both ways.**
  - When the worker returns, the app stops and the process exits with the worker's code (0 or 1).
  - When the worker panics (`--panic-worker`), the process exits with 101, because the drop guard stops the app.
- **`CGEventPostToPid(self)`** delivered F13 down and up while `CGPreflightPostEventAccess()` was true. In an earlier run, where preflight reported false, it did not arrive.

## Unverified (needs a human)

- **Real hardware input.** Everything above is synthetic NSEvents, dispatched into the app. The path from the HID device through the window server is not exercised. That covers:
  - real key-repeat timing;
  - real mouse acceleration;
  - trackpad momentum scrolling;
  - precise-delta units.
- **IME composition.** Japanese/Chinese input, the accent popup (press and hold), the emoji picker, and where the candidate window appears. `setMarkedText:` and friends are implemented but never called.
- **Mouse-look under real capture.** With a physical mouse: whether the cursor really stays still, whether raw deltas continue at the screen edge, and whether a click lands outside the window. Also whether deltas are accelerated (they are, by `NSEvent.deltaX` semantics; see Findings).
- **Visual output.** Whether the colours, the triangle's rotation and the absence of tearing are correct was never looked at. Only the numbers and a clean validation layer are verified.
- **Menu and Dock behaviour.** The Dock icon's appearance, the Dock's own "Quit" item, a real Cmd-Q keypress, and whether the menu bar is usable on first activation.
- **Live resize by dragging.** This is a different run-loop mode (`NSEventTrackingRunLoopMode`); ticks are scheduled in common modes but that was not observed. Also moving the window to a second display with a different scale or refresh rate.
- **Logout and shutdown while running.** See Risks.
- **Interactive mode** was only started for 3.5 s: it ran at 116–120 fps, reported an initial `FocusGained`, and was then killed.

## Findings and surprises

### Threads and objc2 types

- **What must run on the main thread.** Everything in `NSApplication`, `NSWindow`, `NSView`, `NSMenu`, `NSCursor`, `NSEvent` monitors, `displayLinkWithTarget:selector:` (it is an NSView method), `setContentsScale`, and `CGWarpMouseCursorPosition`/`CGAssociateMouseAndMouseCursorPosition` (called there for ordering with the cursor state).
- **What is safe on the VM thread (no checker complaints).**
  - `MTLCreateSystemDefaultDevice`, the command queue, libraries, pipelines and encoding.
  - `CAMetalLayer.nextDrawable`, `setDrawableSize`, `presentDrawable`/`commit`.
  - Receiving a CAMetalDisplayLink drawable from another thread and rendering into it after the callback returns.
  - No `CoreAnimation: uncommitted CATransaction` warning appeared when the VM thread exited.
- **objc2 Send/Sync.**
  - `ProtocolObject<dyn MTLDevice>` and `MTLCommandQueue` are `Send + Sync`.
  - `MTLCommandBuffer`, encoders, `MTLTexture`, `CAMetalLayer`, `CAMetalDrawable`, `CADisplayLink` and `CAMetalDisplayLink` are **not** Send. The spike moves `CAMetalLayer`, the display link and delivered drawables with an `AssertSend` wrapper: one `unsafe impl Send`, justified per use.
  - `MainThreadOnly` classes (`NSWindow`, `NSView`, the delegates) cannot even be allocated without a `MainThreadMarker`, so misuse is a compile error.
  - `dispatch2::MainThreadBound<T>` is the right tool to hold one from another thread.
- **Edition 2024 disjoint closure captures bypass a Send wrapper.** `move || w.0.foo()` captures `w.0`, which is not Send, not `w`. Rebind the whole wrapper inside the closure (`let w = w;`).
- **Deadlock found by measurement.**
  - `dispatch_sync` to the main queue only completes while the main run loop runs.
  - After `[NSApp run]` returned, `main` sat in `thread::join`. The VM thread's drop guard then did one more `on_main`, and the process hung (seen with `sample`).
  - The fix: after `run` returns, keep turning the main run loop (`runMode:beforeDate:` in 5 ms slices) until the VM thread `is_finished()`, and only then join.
  - The same rule forbids dropping a main-thread object (`MainThreadBound` sends its drop to main) after the run loop stops. The VM side therefore holds **ids into a main-thread table, never AppKit objects**, which matches the design's handle ids.
- **`on_main` guards.**
  - It runs the closure inline when already on main, because `dispatch_sync` onto the queue you are on deadlocks.
  - It wraps the closure in `objc2::exception::catch` and `std::panic::catch_unwind`. A panic that unwinds out of a GCD C callback aborts.
  - `dispatch2::run_on_main` itself has an internal `unwrap`; the spike does not use it.

### Activation (unbundled binary)

- **An unbundled process starts with activation policy Prohibited.** It has no Dock icon, no menu bar and cannot become key until `setActivationPolicy(Regular)`.
- **macOS 14's cooperative `NSApp.activate()` failed 6 of 6 launches from a background process;** the app never became active. The deprecated `activateIgnoringOtherApps(true)` succeeded in every one of about 25 runs. With `activate()` the window is not key, so no keyboard input reaches it. The spike uses the deprecated call; the real driver should too, until Apple removes it (it still works on 26.3).
- **Activation is asynchronous.** `isActive` is false inside `applicationDidFinishLaunching:` and true a few frames later.
- **The Dock icon is the generic executable icon** unless `setApplicationIconImage:` is called.
- **The app menu's title is the process name.** A main menu must be installed or there is no Cmd-Q at all.

### Pacing

- **CADisplayLink on NSView (macOS 14+) works and follows the view's screen.** It is scheduled in `NSRunLoopCommonModes` so it keeps ticking during tracking modes.
  - It **stops entirely when the window is hidden**. A pure pull-on-tick `next_frame` then blocks forever, which would also stall anything else the single-threaded VM does, such as a network keep-alive. The spike's `next_frame` returns after 100 ms with `ticks = 0` if no tick came.
- **CADisplayLink on the main run loop makes frames hostage to main-thread work.** A 50 ms main-thread stall gave a 51–54 ms frame. On a dedicated run-loop thread the same stall cost 9–21 ms. Events still wait for the main thread either way, so they arrive late, but frames keep coming.
- **The display link's tick and drawable availability are not phase-locked** (see numbers). CAMetalDisplayLink (macOS 14+) fixes that by delivering a drawable with each tick.
  - Its delegate is **weak**: drop the target and ticks stop silently. CADisplayLink instead *retains* its target, a cycle broken by `invalidate`.
  - It keeps ticking at about 34 Hz when the window is hidden.
- **For macOS < 14:**
  - Neither `-[NSView displayLinkWithTarget:selector:]` nor `CAMetalDisplayLink` exists.
  - The fallback is `CVDisplayLink`. It is deprecated in macOS 15 but present, fires on its own high-priority thread, and can signal the same condvar.
  - The simplest fallback is no link at all: block in `nextDrawable` with `displaySyncEnabled`, at the cost of the latency above.
  - Not tested here (no older macOS available).
- **`drawableSize` must be set by whoever calls `nextDrawable`.** Doing it on the VM thread, at the moment `next_frame` returns a new size, guarantees that `Frame.width/height` equal the drawables the frame gets: 0 mismatches measured.
  - The main thread sets `contentsScale` (from `viewDidChangeBackingProperties`) and computes pixel size = bounds × `backingScaleFactor`.
  - Setting the layer before the view is in a window produced a spurious startup `Resized`; fixed by ignoring size changes until the view has a window.

### Input

- **Key identity should be the physical keycode (`kVK_*`), not characters.** WASD then means positions on any layout. Characters come separately as `Text`, after the input method.
- **Modifiers never send keyDown/keyUp, only `flagsChanged:`.** Which side changed comes from the event's keycode plus the `NX_DEVICE*KEYMASK` bits in `modifierFlags`. Toggling on keycode alone desyncs.
- **AppKit swallows the `keyUp:` of any key released while Cmd is held.** The key-down went through key-equivalent matching. Without a local `keyUp` monitor that forwards those, such keys stay in `held` forever (the monitor was verified).
- **Focus loss must release all held keys.** The OS never delivers their key-ups to a window that is not key. Cursor capture should also be suspended on resign-key and restored on become-key; the spike does this.
- **Close and Quit are separate causes** in the spike (`CloseRequested(CloseButton | Quit)`). The design's single `CloseRequested` loses which one it was, and a program may want "close window" and "quit app" to behave differently.
- **The design's `Input` has no mouse buttons.** A Minecraft client needs left/right click as both events and held state. The spike adds `MouseDown(n)`/`MouseUp(n)` events.
- **`NSEvent.deltaX/Y` are accelerated values in points.** Truly raw, unaccelerated mouse motion needs `GCMouse` (GameController, macOS 11+) or IOHID. Not tested.
- **Scroll units differ.** Wheels give lines; trackpads give precise point deltas plus momentum phases. The spike divides precise deltas by a nominal 10 to report lines. The API needs a decision here.
- **`doCommandBySelector:` must be implemented as a no-op.** NSResponder's default beeps on every arrow, Return or Backspace.
- **While an IME is active, it consumes keystrokes for composition,** so a game should be able to turn text input off (like SDL's `StartTextInput`/`StopTextInput`). Not measured.

### Synthesising input (for tests)

- **No accessibility permission is needed** to create NSEvents and deliver them through `-[NSApplication postEvent:atStart:]` (the full path: queue, local monitors, key equivalents, `sendEvent:`) or `-[NSWindow sendEvent:]` (synchronous). The queue path only reaches the window when the app is active and the window is key.
- **NSEvent has no constructor that sets `deltaX`.** Build a CGEvent, set `kCGMouseEventDeltaX/Y`, and wrap it with `eventWithCGEvent:`.
- **In-process CGEvents convert with window number 0,** and setting the window-under-pointer fields does not help. Scroll events are hit-tested by `locationInWindow`, so the spike places the global point where AppKit's flip lands inside the view.
- **`CGEventPost`/`CGEventPostToPid` need the "post events" TCC permission** (`CGPreflightPostEventAccess`), which belongs to the responsible process (the terminal). Without it, events are dropped silently.

### Reliability (the "code never crashes" rule)

Places AppKit or Metal can abort or throw, and how the spike guards each:

| Place | Failure | Guard in the spike |
|---|---|---|
| Any AppKit call on a non-main thread | Exception or undefined behaviour (Main Thread Checker) | `MainThreadOnly` types plus `MainThreadMarker`, so it does not compile. All requests go through `on_main`. |
| AppKit method raising (bad index, bad argument, inconsistent state) | `NSException`, abort if uncaught | `objc2::exception::catch` in `on_main` (verified). The real crate should wrap every Objective-C entry point the same way. |
| Rust panic inside a `define_class!` callback or a GCD block | Unwinds into Objective-C/C frames: abort | Handlers touch only a poison-tolerant mutex and plain data. `on_main` adds `catch_unwind`. |
| `NSWindow` released when closed while held in `Retained` | Use after free | `setReleasedWhenClosed(false)`, which objc2 requires. |
| `NSCursor hide`/`unhide` imbalance | Cursor invisible after exit | `capture_applied` guards both directions. |
| `MTLCreateSystemDefaultDevice`, `newCommandQueue`, `commandBuffer`, the encoder, `newFunctionWithName` | `nil` | Mapped to `Err`. |
| `newLibraryWithSource`, `newRenderPipelineState` | `NSError` | Mapped to `Err`. |
| `nextDrawable` | `nil` after about 1 s | The frame is skipped. |
| Pixel format mismatch (pipeline vs layer) | SIGABRT under validation | Formats fixed to `BGRA8Unorm` on both. Validation layer in assert mode: clean. |
| VM thread panics | Window left open, `[NSApp run]` never returns | `StopAppOnDrop` guard (verified: exit 101). |
| Late `dispatch_sync` after `[NSApp run]` returns | Deadlock | Keep the run loop turning until the VM thread finishes (verified). |
| `applicationShouldTerminate:` returning Cancel during logout or shutdown | Logout interrupted | **Open**; see Risks. |

## Recommendations for `scarlet_metal` / `scarlet/app`

1. **Keep the handoff exactly as in `model.rs`.** One `Mutex<Pending>` + `Condvar` per window, written by the main thread, drained by `next_frame`:
   - `held: BTreeSet<Key>`, where a duplicate down is not a new press;
   - summed `mouse_dx/dy` and `scroll`;
   - `events: Vec<Event>`, with `Resized` coalesced (replaced, not appended);
   - the latest size and scale, and a tick counter with its timestamps.

   `next_frame` waits for the counter to change, then swaps everything out in one critical section.
   - The main thread never allocates in a Scarlet heap and never blocks on the VM.
   - The VM never holds an AppKit object, only an id into the main thread's table.
2. **Give `next_frame` a no-tick fallback** (about 100 ms, `ticks = 0`), so a hidden or minimised window does not stop the VM.
3. **Pace with `CAMetalDisplayLink` on a dedicated run-loop thread (macOS 14+)**, with the delivered drawable handed to the VM through a one-slot mailbox.
   - This measured best on every axis: no `nextDrawable` blocking, p50 0.012–0.025 ms tick-to-return, immune to main-thread stalls, and it keeps ticking when hidden.
   - `app.present` then renders into that drawable.
   - Set `preferredFrameRateRange` to the screen's `maximumFramesPerSecond`.
   - Fallback on macOS < 14: `CVDisplayLink` on its own thread + `nextDrawable`.
   - If CADisplayLink is kept instead, put it on a dedicated thread too.
4. **Resize the drawable on the VM thread** when `next_frame` sees a new size, and report pixel sizes (points × `backingScaleFactor`).
5. **Activation:** `setActivationPolicy(Regular)` then `activateIgnoringOtherApps(true)`. Install a main menu with Quit (`terminate:`) and Close (`performClose:`).
6. **Close and Quit stay requests, but carry their cause.** Use `CloseRequested(CloseButton | Quit)`. Add mouse buttons to `Input`/`Event`. Decide the scroll unit. Consider `app.text_input(w, Bool)`.
7. **Driver shutdown order:**
   - The VM thread ends, or panics behind a drop guard.
   - It posts close-all + `stop:`, plus a dummy `ApplicationDefined` event so `run` returns at once.
   - The main thread keeps turning its run loop until the VM thread has finished.
   - Only then does it `join` and `exit(code)`.
8. **`on_main` is the single door to AppKit.** It runs inline on main, and catches exceptions and panics as values.
9. **Put the Main Thread Checker and Metal's validation layer into the real test harness,** as metal-design.md already plans for the validation layer. Launch the child directly, never through a SIP-protected binary (see "How to run").
10. **Tests can drive input without any permission** through `-[NSWindow sendEvent:]` on synthetic NSEvents. The app-queue path additionally needs the app active and key.

## Open risks

- **Logout and shutdown.** Returning `NSTerminateCancel` blocks logout ("app interrupted logout"). The right shape is probably `NSTerminateLater`, then `replyToApplicationShouldTerminate:` once the program answers. But `terminate:` ends in `exit()` with no way to pass the run's exit code other than exiting from `applicationWillTerminate:`. Untested.
- **The 60 Hz presentation episodes** (Pacing) are unexplained. The frame-rate range is a mitigation, not a proven fix.
- **The deprecated activation API** could be removed in a future macOS. There is no documented replacement that works for a terminal-launched binary.
- **Raw mouse input** needs GCMouse or IOHID for unaccelerated deltas, which adds a framework and possibly a permission prompt.
- **IME, key repeat, trackpad momentum and multi-display scale/refresh changes** are unverified.
- **Holding a CAMetalDisplayLink drawable past its callback, on another thread,** passed validation and produced frames. But Apple's samples render inside the callback. If a VM frame takes more than about one frame of headroom (8.25 ms measured), the link hands out the next drawable and the old one is dropped unused, which silently drops a frame.
- **Main-thread-bound events** (keys, mouse) still wait behind main-thread work even with a dedicated link thread. Input latency under a busy main thread was not measured.

## Would winit have avoided any of this?

Partly. winit already does these:

- `Regular` activation;
- the Cmd-keyUp workaround (it overrides `sendEvent:`);
- the modifier left/right tracking;
- `NSTextInputClient`, including IME;
- `CGAssociateMouseAndMouseCursorPosition`-based cursor grab;
- HiDPI scale events.

It does not do the part this design owns:

- **winit wants the main thread to run its event loop and to call the program's handlers from it.** That is the inverse of a VM that pulls frames on its own thread. Using it would mean forwarding every winit event to the VM thread through the same kind of mailbox built here.
- **winit's pacing is not display-link driven.** It offers `request_redraw` on the main thread and has no CAMetalDisplayLink.

No wall was hit with direct objc2. The only real surprises (the deadlock at join, the swallowed Cmd key-ups, and cooperative activation) would have hit a winit-based design as well, or are handled in about a dozen lines here.
