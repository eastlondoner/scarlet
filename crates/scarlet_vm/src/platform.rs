//! What a program reaches of the machine through the runtime rather than the
//! OS: today, a GPU through Metal (`docs/metal-design.md`).
//!
//! The VM sees it only through [`Platform`], a trait in plain Rust types, so
//! the VM stays safe code and knows nothing of Objective-C. The driver hands
//! one in with the [`crate::Host`]; a host without one has no GPU, and
//! `metal.device` answers `Err(Unsupported)`.
//!
//! A platform is installed as a [`Gpu`], which pairs it with the one counter
//! its ids come from. The VM names every object before the platform makes it,
//! with an [`Id`] from that counter, and the platform keeps the object under
//! that id. Every run on the platform, in turn or at once, takes its ids from
//! the same counter, so none is given twice while the platform lives. A
//! platform cannot make an id up, so every id it is handed is one the VM gave
//! it, of the kind the method's type says.
//!
//! What a platform is handed has been checked by the VM, and its type says
//! so: [`BufferBytes`] is at least one byte and fits its device, and
//! [`ReadInto`] takes exactly as many bytes as the buffer it names holds. A
//! platform does not check them again, and every platform answers a program
//! the same.

#![deny(clippy::wildcard_enum_match_arm)]

use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};

/// An object a platform holds, of kind `T`, like `Id<Buffer>`. The kind is
/// only in the type, so a device's id cannot be passed where a buffer's goes.
/// It is the object's identity on its platform for good, never its address,
/// which a later object can come back at.
pub struct Id<T> {
    raw: NonZeroU64,
    kind: PhantomData<fn() -> T>,
}

impl<T> Id<T> {
    pub(crate) fn new(raw: NonZeroU64) -> Id<T> {
        Id {
            raw,
            kind: PhantomData,
        }
    }

    /// The number a program sees, as in `<metal.Buffer #3>`.
    pub(crate) fn raw(self) -> u64 {
        self.raw.get()
    }
}

impl<T> Clone for Id<T> {
    fn clone(&self) -> Id<T> {
        *self
    }
}

impl<T> Copy for Id<T> {}

impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Id<T>) -> bool {
        self.raw == other.raw
    }
}

impl<T> Eq for Id<T> {}

impl<T> Hash for Id<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.raw.hash(state);
    }
}

impl<T> fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.raw)
    }
}

/// A GPU: `metal.Device`. Only a marker for [`Id`]; nothing is one.
pub enum Device {}

/// Memory the CPU and a GPU share: `metal.Buffer`. Only a marker for [`Id`].
pub enum Buffer {}

/// Any object a platform holds, with its kind, as the VM gives it up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Handle {
    Device(Id<Device>),
    Buffer(Id<Buffer>),
}

/// What the VM keeps of a device the platform made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// The GPU's name, like `Apple M1 Max`.
    pub(crate) name: String,
    /// The most bytes one buffer on it can hold.
    max_buffer_bytes: u64,
}

impl DeviceInfo {
    pub fn new(name: String, max_buffer_bytes: u64) -> DeviceInfo {
        DeviceInfo {
            name,
            max_buffer_bytes,
        }
    }
}

/// Bytes for a new buffer on a device: at least one, and no more than the
/// device's largest buffer holds. Only the VM makes one, once it has checked
/// both, and it carries the device it was checked against, so the platform
/// cannot make the buffer on another.
#[derive(Debug, Clone, Copy)]
pub struct BufferBytes<'a> {
    device: Id<Device>,
    bytes: &'a [u8],
}

/// Why bytes are not [`BufferBytes`] for a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Unfit {
    Empty,
    /// More than the device's largest buffer, which holds this many.
    TooLarge(u64),
}

/// A length a buffer on `device` can hold, checked before the bytes are
/// found, so that a request refused costs nothing.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Fits {
    device: Id<Device>,
    len: usize,
}

impl Fits {
    /// `len` bytes on `device`, which the VM holds as `info`.
    pub(crate) fn check(device: Id<Device>, info: &DeviceInfo, len: u64) -> Result<Fits, Unfit> {
        if len == 0 {
            return Err(Unfit::Empty);
        }
        match usize::try_from(len) {
            Ok(n) if len <= info.max_buffer_bytes => Ok(Fits { device, len: n }),
            Ok(_) | Err(_) => Err(Unfit::TooLarge(info.max_buffer_bytes)),
        }
    }

    /// `bytes`, as many as were checked, or `None` for another length.
    pub(crate) fn bytes(self, bytes: &[u8]) -> Option<BufferBytes<'_>> {
        (bytes.len() == self.len).then_some(BufferBytes {
            device: self.device,
            bytes,
        })
    }
}

impl<'a> BufferBytes<'a> {
    /// The device the bytes fit.
    pub fn device(&self) -> Id<Device> {
        self.device
    }

    /// The bytes: never empty, and never past the device's largest buffer.
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }
}

/// Room for all of a buffer's bytes: exactly as many as it holds. Only the
/// VM makes one, from the length it recorded when the buffer was made, and
/// the only way to fill it is whole, so a platform cannot leave part of it
/// holding whatever the memory held before.
#[derive(Debug)]
pub struct ReadInto<'a> {
    buffer: Id<Buffer>,
    into: &'a mut [u8],
}

impl<'a> ReadInto<'a> {
    /// `into`, for `buffer`, which the VM recorded as holding `len` bytes.
    /// `None` when `into` is another length.
    pub(crate) fn check(buffer: Id<Buffer>, len: u64, into: &'a mut [u8]) -> Option<ReadInto<'a>> {
        (u64::try_from(into.len()) == Ok(len)).then_some(ReadInto { buffer, into })
    }

    /// The buffer to read.
    pub fn buffer(&self) -> Id<Buffer> {
        self.buffer
    }

    /// Fill the room with `bytes`, the buffer's. Bytes of any other length
    /// are refused, and nothing is copied.
    pub fn copy_from(self, bytes: &[u8]) -> Result<(), Fault> {
        if bytes.len() != self.into.len() {
            return Err(Fault::new(format!(
                "{:?} read as {} bytes, where it holds {}",
                self.buffer,
                bytes.len(),
                self.into.len()
            )));
        }
        self.into.copy_from_slice(bytes);
        Ok(())
    }
}

/// A platform failing where the runtime should have kept it from failing,
/// like Metal raising an Objective-C exception. A bug in the runtime, never
/// the program's doing, so a run that meets one stops, saying what.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fault(String);

impl Fault {
    pub fn new(what: impl Into<String>) -> Fault {
        Fault(what.into())
    }
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Why [`Platform::device`] made no device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceError {
    /// This machine has no GPU API the platform can open.
    Unsupported,
    /// It has one, but no GPU it can use.
    NoDevice,
    Fault(Fault),
}

/// Why [`Platform::buffer`] made no buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BufferError {
    /// The GPU API found no memory for it.
    OutOfMemory,
    Fault(Fault),
}

/// A GPU, as the VM reaches it. Its methods take `&self`, so an
/// implementation keeps its objects behind a lock, and it is `Send + Sync`,
/// so that processes on other threads can share one when they exist.
pub trait Platform: Send + Sync {
    /// Make the system's default GPU, as `device`.
    fn device(&self, device: Id<Device>) -> Result<DeviceInfo, DeviceError>;

    /// Make `buffer`, holding a copy of `bytes`, on the device they fit.
    fn buffer(&self, buffer: Id<Buffer>, bytes: BufferBytes<'_>) -> Result<(), BufferError>;

    /// Copy all of a buffer's bytes out.
    fn read(&self, to: ReadInto<'_>) -> Result<(), Fault>;

    /// Let go of the object `handle` names. The VM calls this once for each
    /// handle it made, when the last value naming it goes or the run ends,
    /// and never names it again. Like `Drop`, it cannot fail.
    fn release(&self, handle: Handle);
}

/// A platform, installed: `platform`, with the counter every id of an object
/// it holds comes from. The counter goes with the platform, not with a run,
/// so however many runs share it, in turn or at once, no id is given twice,
/// released or not: an id is the identity of one object on this platform.
pub struct Gpu<P: ?Sized = dyn Platform> {
    /// The last id given out. Ids count up from 1, one counter for every
    /// kind.
    last: AtomicU64,
    platform: P,
}

/// Every id a [`Gpu`] can give out has been given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OutOfIds;

impl<P: Platform> Gpu<P> {
    pub fn new(platform: P) -> Gpu<P> {
        Gpu {
            last: AtomicU64::new(0),
            platform,
        }
    }

    /// A platform whose last id given out was `last`, for a test of what
    /// happens when they run out.
    #[cfg(test)]
    pub(crate) fn after(platform: P, last: u64) -> Gpu<P> {
        Gpu {
            last: AtomicU64::new(last),
            platform,
        }
    }
}

impl<P: ?Sized + Platform> Gpu<P> {
    /// A new id, never given out before by this platform.
    pub(crate) fn next<T>(&self) -> Result<Id<T>, OutOfIds> {
        // Relaxed: the counter orders nothing but itself, and a read-modify-
        // write never gives two callers one value.
        let last = self
            .last
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| OutOfIds)?;
        let raw = last
            .checked_add(1)
            .and_then(NonZeroU64::new)
            .ok_or(OutOfIds)?;
        Ok(Id::new(raw))
    }
}

impl<P: ?Sized> Deref for Gpu<P> {
    type Target = P;

    fn deref(&self) -> &P {
        &self.platform
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Room for a read is filled whole or not at all: bytes of another
    /// length are refused, and leave the room as it was.
    #[test]
    fn a_read_of_another_length_is_refused() {
        let id = Id::new(NonZeroU64::MIN);
        let mut room = [0xEE; 4];
        for bytes in [&[1, 2][..], &[1, 2, 3, 4, 5]] {
            let to = ReadInto::check(id, 4, &mut room).expect("room for 4");
            assert!(to.copy_from(bytes).is_err(), "{bytes:?}");
            assert_eq!(room, [0xEE; 4]);
        }
        let to = ReadInto::check(id, 4, &mut room).expect("room for 4");
        assert_eq!(to.copy_from(&[1, 2, 3, 4]), Ok(()));
        assert_eq!(room, [1, 2, 3, 4]);
    }
}
