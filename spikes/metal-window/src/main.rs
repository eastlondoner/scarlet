//! Spike: the thread split and input model planned for `scarlet/app`.
//!
//! The OS main thread runs NSApplication and never runs "Scarlet". A second
//! thread stands in for the VM: it opens a window, pulls one frame per display
//! refresh with a blocking `next_frame`, and renders with Metal itself.
//!
//!   metal-window-spike            interactive
//!   metal-window-spike --auto     scripted checks + measurements, exits 0/1
//!   --link-thread=dedicated       display link on its own run-loop thread
//!   --pacing=metal-link           CAMetalDisplayLink instead of CADisplayLink
//!   --cooperative-activate        macOS 14 `activate` instead of activateIgnoringOtherApps:
//!   --frames=N                    frames in the pacing measurement (default 600)

#[cfg(target_os = "macos")]
mod app;
#[cfg(target_os = "macos")]
mod appkit;
mod model;
#[cfg(target_os = "macos")]
mod render;
#[cfg(target_os = "macos")]
mod synth;
#[cfg(target_os = "macos")]
mod worker;

#[cfg(target_os = "macos")]
fn main() {
    use objc2::MainThreadMarker;
    use objc2::runtime::ProtocolObject;
    use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy};

    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts = worker::Options {
        auto: args.iter().any(|a| a == "--auto"),
        link_thread_dedicated: args.iter().any(|a| a == "--link-thread=dedicated"),
        metal_display_link: args.iter().any(|a| a == "--pacing=metal-link"),
        pacing_frames: args
            .iter()
            .find_map(|a| a.strip_prefix("--frames=")?.parse().ok())
            .unwrap_or(600),
        mtc_selftest: args.iter().any(|a| a == "--mtc-selftest"),
        panic_worker: args.iter().any(|a| a == "--panic-worker"),
    };
    appkit::COOPERATIVE_ACTIVATE.store(
        args.iter().any(|a| a == "--cooperative-activate"),
        std::sync::atomic::Ordering::Relaxed,
    );

    let Some(mtm) = MainThreadMarker::new() else {
        eprintln!("not on the main thread");
        std::process::exit(2);
    };
    let app = NSApplication::sharedApplication(mtm);
    appkit::INITIAL_POLICY.store(
        app.activationPolicy().0 as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    // An unbundled binary starts as a background-only ("Prohibited") process
    // with no Dock icon, no menu bar and no key windows. Regular fixes all three.
    app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
    let delegate = appkit::AppDelegate::new(mtm);
    app.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));
    appkit::install_menu(mtm, &app);
    let monitor = appkit::install_cmd_keyup_monitor();

    // The VM thread. It starts at once; its first AppKit request
    // (`Window::open`) waits on the main queue until `run` below drains it.
    let vm = std::thread::Builder::new()
        .name("vm".into())
        .spawn(move || {
            // However the VM thread ends (return or panic), the main thread
            // must be told, or `app.run()` below never returns.
            let _stop = app::StopAppOnDrop;
            worker::run(opts)
        });
    let vm = match vm {
        Ok(h) => h,
        Err(e) => {
            eprintln!("spawn: {e}");
            std::process::exit(2);
        }
    };

    // Returns when the VM thread asks for `stop:` (appkit::stop_app).
    app.run();

    if let Some(m) = monitor {
        // SAFETY: the object addLocalMonitor returned.
        unsafe { objc2_app_kit::NSEvent::removeMonitor(&m) };
    }
    // `run` has returned, but the VM thread may still be finishing, and may
    // yet make a synchronous main-queue call (a drop guard, a last close). A
    // plain `join` here would deadlock against it: the main queue is only
    // drained while the main run loop runs. So keep turning the run loop
    // until the thread is done. (Measured: without this, a late `on_main`
    // from the VM thread's drop guard hung the process in `join`.)
    while !vm.is_finished() {
        let until = objc2_foundation::NSDate::dateWithTimeIntervalSinceNow(0.005);
        // SAFETY: NSDefaultRunLoopMode is a valid mode constant.
        let mode = unsafe { objc2_foundation::NSDefaultRunLoopMode };
        objc2_foundation::NSRunLoop::mainRunLoop().runMode_beforeDate(mode, &until);
    }
    let code = vm.join().unwrap_or(101);
    std::process::exit(code);
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("macOS only");
    std::process::exit(2);
}
