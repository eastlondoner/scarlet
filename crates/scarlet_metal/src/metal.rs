//! Metal's objects, kept under the ids the VM gives them.
//!
//! The table holds each object's one strong reference, so releasing an id
//! releases the object. Its keys are the VM's ids, never an object's address,
//! which a later object can come back at once the first is freed.
//!
//! Metal is loaded the first time a program asks for a device, not when the
//! process starts. Linking it would load Foundation with it, about a
//! millisecond of every `scarlet run`, GPU or not; the driver links with
//! `-dead_strip_dylibs`, and nothing here names a Metal or Foundation symbol,
//! so neither is loaded until [`create_device`] opens Metal. Every other call
//! is an Objective-C message, which the runtime looks up when it is sent.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::ffi::c_void;
use std::panic::AssertUnwindSafe;
use std::ptr::NonNull;
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use scarlet_vm::platform::{
    Buffer, BufferBytes, BufferError, Device, DeviceError, DeviceInfo, Fault, Handle, Id, Platform,
    ReadInto,
};

type MtlDevice = Retained<ProtocolObject<dyn MTLDevice>>;

type MtlBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;

/// A buffer, as the table keeps it.
struct Kept(MtlBuffer);

// SAFETY: Apple documents a buffer, like its device, as safe to use from any
// thread. objc2-metal leaves `MTLBuffer` without `Send` because writing its
// contents while the GPU reads them needs synchronising; this crate only
// reads contents, and no GPU work exists yet to write them.
unsafe impl Send for Kept {}

#[derive(Default)]
struct Objects {
    devices: HashMap<Id<Device>, MtlDevice>,
    buffers: HashMap<Id<Buffer>, Kept>,
}

pub(crate) struct Metal {
    objects: Mutex<Objects>,
}

impl Metal {
    pub(crate) fn new() -> Metal {
        Metal {
            objects: Mutex::default(),
        }
    }

    /// Nothing here panics while holding the lock, so a poisoned lock still
    /// guards a whole table.
    fn objects(&self) -> MutexGuard<'_, Objects> {
        self.objects.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// How many objects the table holds.
    #[cfg(test)]
    pub(crate) fn held(&self) -> usize {
        let objects = self.objects();
        objects.devices.len() + objects.buffers.len()
    }
}

/// `MTLCreateSystemDefaultDevice`, as `MTLDevice.h` declares it.
type CreateDevice = unsafe extern "C-unwind" fn() -> *mut ProtocolObject<dyn MTLDevice>;

/// Metal's `MTLCreateSystemDefaultDevice`, from Metal opened the first time
/// it is asked for, or `None` on a Mac with no Metal to open. Metal stays
/// open for the rest of the process.
fn create_device() -> Option<CreateDevice> {
    static CREATE: OnceLock<Option<CreateDevice>> = OnceLock::new();
    *CREATE.get_or_init(|| {
        let path = c"/System/Library/Frameworks/Metal.framework/Metal";
        // SAFETY: `path` is a NUL-terminated string that outlives the call.
        // Opening Metal runs its initialisers, which is what linking it would
        // have done at launch.
        let metal = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL) };
        if metal.is_null() {
            return None;
        }
        // SAFETY: `metal` is a handle `dlopen` just returned, never closed,
        // and the name is a NUL-terminated string.
        let found = unsafe { libc::dlsym(metal, c"MTLCreateSystemDefaultDevice".as_ptr()) };
        if found.is_null() {
            return None;
        }
        // SAFETY: the symbol is Metal's `id<MTLDevice>
        // MTLCreateSystemDefaultDevice(void)`, which `CreateDevice` matches:
        // no arguments, and an object pointer back. It is "C-unwind" as
        // objc2-metal declares it, so an exception it raises reaches
        // `guarded`.
        Some(unsafe { std::mem::transmute::<*mut c_void, CreateDevice>(found) })
    })
}

/// Run `f`, whose Objective-C calls may raise an exception, catching one as a
/// [`Fault`]. Uncaught, it would unwind into Rust, which aborts the process.
/// `f` only calls Metal and holds nothing a caught exception could leave
/// half-changed.
fn guarded<R>(f: impl FnOnce() -> R) -> Result<R, Fault> {
    objc2::exception::catch(AssertUnwindSafe(f)).map_err(|e| match e {
        Some(e) => Fault::new(format!("Metal raised {e}")),
        None => Fault::new("Metal raised an exception with no object"),
    })
}

fn made_twice(handle: Handle) -> Fault {
    Fault::new(format!("the VM named two objects {handle:?}"))
}

fn not_held(handle: Handle) -> Fault {
    Fault::new(format!("{handle:?}, which Metal does not hold for the VM"))
}

impl Platform for Metal {
    fn device(&self, id: Id<Device>) -> Result<DeviceInfo, DeviceError> {
        let create = create_device().ok_or(DeviceError::Unsupported)?;
        let raw = guarded(|| {
            // SAFETY: `create` is `MTLCreateSystemDefaultDevice`, which takes
            // nothing and may be called from any thread.
            unsafe { create() }
        })
        .map_err(DeviceError::Fault)?;
        // SAFETY: `MTLCreateSystemDefaultDevice` returns a device the caller
        // owns, by Core Foundation's create rule, or nil. `from_raw` takes
        // that one reference.
        let Some(device) = (unsafe { Retained::from_raw(raw) }) else {
            return Err(DeviceError::NoDevice);
        };
        let (name, max) = guarded(|| (device.name().to_string(), device.maxBufferLength()))
            .map_err(DeviceError::Fault)?;
        match self.objects().devices.entry(id) {
            Entry::Vacant(slot) => slot.insert(device),
            Entry::Occupied(_) => return Err(DeviceError::Fault(made_twice(Handle::Device(id)))),
        };
        Ok(DeviceInfo::new(
            name,
            u64::try_from(max).unwrap_or(u64::MAX),
        ))
    }

    fn buffer(&self, id: Id<Buffer>, bytes: BufferBytes<'_>) -> Result<(), BufferError> {
        let on = bytes.device();
        let device = self.objects().devices.get(&on).cloned();
        let device = device.ok_or_else(|| BufferError::Fault(not_held(Handle::Device(on))))?;
        let data = bytes.bytes();
        let pointer = NonNull::from(data).cast::<c_void>();
        let made = guarded(|| {
            // SAFETY: `pointer` is `data`, readable for `data.len()` bytes
            // for the whole call. Metal copies exactly that many into memory
            // of its own before it returns, and keeps no pointer to them.
            unsafe {
                device.newBufferWithBytes_length_options(
                    pointer,
                    data.len(),
                    MTLResourceOptions::StorageModeShared,
                )
            }
        })
        .map_err(BufferError::Fault)?;
        let buffer = made.ok_or(BufferError::OutOfMemory)?;
        match self.objects().buffers.entry(id) {
            Entry::Vacant(slot) => slot.insert(Kept(buffer)),
            Entry::Occupied(_) => return Err(BufferError::Fault(made_twice(Handle::Buffer(id)))),
        };
        Ok(())
    }

    fn read(&self, to: ReadInto<'_>) -> Result<(), Fault> {
        let id = to.buffer();
        let buffer = self.objects().buffers.get(&id).map(|kept| kept.0.clone());
        let buffer = buffer.ok_or_else(|| not_held(Handle::Buffer(id)))?;
        let (contents, length) = guarded(|| (buffer.contents(), buffer.length()))?;
        let into = to.into_slice();
        // The VM made `into` the length it recorded for the buffer. Reading
        // Metal's memory soundly rests on Metal's own length, not on that.
        if into.len() != length {
            return Err(Fault::new(format!(
                "{id:?} holds {length} bytes, and the VM made room for {}",
                into.len()
            )));
        }
        // SAFETY: a buffer made with `StorageModeShared` keeps its bytes in
        // memory the CPU can read, `length` of them at `contents`, for as
        // long as it lives, and `buffer` keeps it alive past the copy. No GPU
        // work exists yet to write them while they are read.
        let held = unsafe { std::slice::from_raw_parts(contents.as_ptr().cast::<u8>(), length) };
        into.copy_from_slice(held);
        Ok(())
    }

    fn release(&self, handle: Handle) {
        let mut objects = self.objects();
        let device = match handle {
            Handle::Device(id) => objects.devices.remove(&id),
            Handle::Buffer(_) => None,
        };
        let buffer = match handle {
            Handle::Buffer(id) => objects.buffers.remove(&id),
            Handle::Device(_) => None,
        };
        drop(objects);
        // Dropped outside the lock, since the last release runs the object's
        // `dealloc`, and inside `guarded`, so an exception there cannot
        // unwind into Rust. Like `Drop`, a release has no one to report a
        // fault to.
        let _ = guarded(move || drop((device, buffer)));
    }
}
