//! What the VM thread sees: the stand-in for `scarlet/app`'s intrinsics.
//!
//! Every call that needs AppKit goes through `on_main`, which runs a closure on
//! the main queue and waits for its answer. Nothing here holds an AppKit
//! object: a window is an id into the main thread's table.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use dispatch2::DispatchQueue;
use objc2::MainThreadMarker;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_core_foundation::CGSize;
use objc2_foundation::NSString;
use objc2_metal::MTLDevice;
use objc2_quartz_core::CAMetalLayer;

use crate::appkit::{self, AssertSend, CaptureResult, DrawableSlot, OpenSpec};
use crate::model::{AppError, Frame, FrameCursor, WindowShared};

/// Runs `f` on the main thread and waits for its result.
///
/// Three guards: an Objective-C exception becomes `Err(Exception)`, a Rust
/// panic becomes `Err(Panic)` (unwinding out of a GCD callback would abort),
/// and a call already on the main thread runs inline, because `dispatch_sync`
/// onto the queue you are on deadlocks.
pub fn on_main<R: Send>(f: impl FnOnce(MainThreadMarker) -> R + Send) -> Result<R, AppError> {
    let run = move |mtm: MainThreadMarker| -> Result<R, AppError> {
        let caught = std::panic::catch_unwind(AssertUnwindSafe(|| {
            objc2::exception::catch(AssertUnwindSafe(|| f(mtm)))
        }));
        match caught {
            Ok(Ok(r)) => Ok(r),
            Ok(Err(Some(e))) => Err(AppError::Exception(e.to_string())),
            Ok(Err(None)) => Err(AppError::Exception("nil exception".into())),
            Err(_) => Err(AppError::Panic),
        }
    };
    if let Some(mtm) = MainThreadMarker::new() {
        return run(mtm);
    }
    let mut out = None;
    DispatchQueue::main().exec_sync(|| {
        // SAFETY: a block submitted to the main queue runs on the main thread.
        let mtm = unsafe { MainThreadMarker::new_unchecked() };
        out = Some(run(mtm));
    });
    out.unwrap_or(Err(AppError::Panic))
}

/// Fire-and-forget onto the main queue (used only to inject main-thread stalls).
pub fn post_main(f: impl FnOnce(MainThreadMarker) + Send + 'static) {
    DispatchQueue::main().exec_async(move || {
        // SAFETY: as above.
        let mtm = unsafe { MainThreadMarker::new_unchecked() };
        let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let _ = objc2::exception::catch(AssertUnwindSafe(|| f(mtm)));
        }));
    });
}

pub enum DrawableSource<'a> {
    Next(&'a CAMetalLayer),
    Delivered(Option<Retained<ProtocolObject<dyn objc2_quartz_core::CAMetalDrawable>>>),
}

pub struct Window {
    pub id: u64,
    pub shared: Arc<WindowShared>,
    cursor: FrameCursor,
    /// Owned by the VM thread from here on: nextDrawable and drawableSize are
    /// only ever touched from this thread.
    layer: AssertSend<Retained<CAMetalLayer>>,
    drawable_size: (u32, u32),
    pub drawable_resizes: u32,
    slot: Option<DrawableSlot>,
}

impl Window {
    pub fn open(
        spec: OpenSpec<'_>,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Result<Window, AppError> {
        let mut shared_out = None;
        let opened = on_main(|mtm| {
            appkit::open_window(
                mtm,
                &spec,
                |w, h, s| {
                    let shared = Arc::new(WindowShared::new(w, h, s));
                    shared_out = Some(shared.clone());
                    shared
                },
                device,
            )
        })?;
        let shared = shared_out.ok_or(AppError::Panic)?;
        Ok(Window {
            id: opened.id,
            shared,
            cursor: FrameCursor::new(),
            layer: opened.layer,
            drawable_size: (opened.width, opened.height),
            drawable_resizes: 0,
            slot: opened.slot,
        })
    }

    /// `app.next_frame`. The drawable size follows the frame's size here, on
    /// the thread that calls nextDrawable, so a frame's width/height always
    /// matches the drawables it will get.
    pub fn next_frame(&mut self) -> Result<Frame, AppError> {
        let frame = self.shared.next_frame(&mut self.cursor)?;
        let size = (frame.width, frame.height);
        if size != self.drawable_size && size.0 > 0 && size.1 > 0 {
            self.layer
                .0
                .setDrawableSize(CGSize::new(size.0 as f64, size.1 as f64));
            self.drawable_size = size;
            self.drawable_resizes += 1;
        }
        Ok(frame)
    }

    /// Where this frame's drawable comes from: the one a CAMetalDisplayLink
    /// tick delivered, or (CADisplayLink pacing) the layer's nextDrawable.
    pub fn drawable_source(&self) -> DrawableSource<'_> {
        match &self.slot {
            Some(slot) => DrawableSource::Delivered(
                slot.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take()
                    .map(|d| d.0),
            ),
            None => DrawableSource::Next(&self.layer.0),
        }
    }

    #[allow(dead_code)]
    pub fn layer(&self) -> &CAMetalLayer {
        &self.layer.0
    }

    pub fn set_title(&self, title: &str) -> Result<(), AppError> {
        let id = self.id;
        on_main(|_| appkit::with_window(id, |w| w.window.setTitle(&NSString::from_str(title))))?
            .ok_or(AppError::WindowGone)
    }

    pub fn capture_cursor(&self, on: bool) -> Result<CaptureResult, AppError> {
        let id = self.id;
        on_main(|_| appkit::with_window(id, |w| w.view.set_capture(on)))?
            .ok_or(AppError::WindowGone)
    }

    pub fn close(self) -> Result<(), AppError> {
        let id = self.id;
        on_main(move |_| appkit::close_window(id))
    }
}

/// Stops NSApplication when dropped. Held by the VM thread for its whole life,
/// so a panic there still ends the process instead of leaving a window with
/// nothing behind it. Stopping twice is harmless.
pub struct StopAppOnDrop;

impl Drop for StopAppOnDrop {
    fn drop(&mut self) {
        let _ = on_main(|mtm| {
            appkit::close_all();
            appkit::stop_app(mtm);
        });
    }
}
