//! `scarlet/metal`'s built-ins, and the handles a run holds
//! (`docs/metal-design.md`, "Handles").
//!
//! Every request is checked here, before a platform sees it, so that every
//! platform answers a program the same and Metal never meets one it would
//! refuse. What passes is handed on in the types that say it passed
//! (`platform.rs`).
//!
//! The platform's [`Gpu`] gives each object its id, and the run keeps what it
//! knows of the objects it holds in [`Handles`]. A handle cell the heap frees
//! is released to the platform at the next Perceus `Drop`, platform call or
//! `internal.live_handles`, and every handle still held is released when the
//! machine goes, however the run ended.

#![deny(clippy::wildcard_enum_match_arm)]

use std::borrow::Cow;
use std::collections::HashMap;

use super::{Machine, full};
use crate::Stop;
use crate::bigint;
use crate::binary;
use crate::heap::Heap;
use crate::host::Host;
use crate::platform::{
    Buffer, BufferError, Device, DeviceError, DeviceInfo, Fault, Fits, Gpu, Handle, Id, OutOfIds,
    ReadInto, Unfit,
};
use crate::value::Value;

/// What the VM knows of each object a platform holds for this run, by its
/// id.
#[derive(Default)]
pub(super) struct Handles {
    devices: HashMap<Id<Device>, DeviceInfo>,
    /// Each buffer and the bytes it holds, which never changes.
    buffers: HashMap<Id<Buffer>, u64>,
}

impl Handles {
    fn live(&self) -> usize {
        self.devices.len() + self.buffers.len()
    }

    /// Stop holding `handle`: `true` when the run held it.
    fn forget(&mut self, handle: Handle) -> bool {
        match handle {
            Handle::Device(id) => self.devices.remove(&id).is_some(),
            Handle::Buffer(id) => self.buffers.remove(&id).is_some(),
        }
    }

    /// Every handle still held, forgotten: buffers first, as each is made on
    /// a device, then devices, each in the order they were made.
    fn forget_all(&mut self) -> Vec<Handle> {
        let mut buffers: Vec<Id<Buffer>> = self.buffers.drain().map(|(id, _)| id).collect();
        let mut devices: Vec<Id<Device>> = self.devices.drain().map(|(id, _)| id).collect();
        buffers.sort_by_key(|id| id.raw());
        devices.sort_by_key(|id| id.raw());
        let buffers = buffers.into_iter().map(Handle::Buffer);
        buffers
            .chain(devices.into_iter().map(Handle::Device))
            .collect()
    }
}

/// A `scarlet/metal.MetalError`, as the VM makes one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetalError {
    Unsupported,
    NoDevice,
    EmptyBuffer,
    UnalignedBinary,
    TooLarge { max_bytes: u64 },
    OutOfMemory,
}

impl<'h> Machine<'_, 'h, '_> {
    /// The platform, for a call to it, once every handle freed so far is
    /// released: what a program gave up is back before it asks for more.
    /// `None` for a host with no GPU.
    fn gpu(&mut self) -> Option<&'h Gpu> {
        self.release_freed();
        let host: &'h Host = self.host;
        host.gpu()
    }

    /// The platform, for a call about a handle the run holds, which only a
    /// platform can have made.
    fn gpu_of_handle(&mut self) -> Result<&'h Gpu, Stop> {
        self.gpu().ok_or_else(|| {
            Stop::BadProgram("a Metal handle, in a run with no platform to make one".into())
        })
    }

    /// Tell the platform about every handle cell freed since this was last
    /// called.
    pub(super) fn release_freed(&mut self) {
        let host: &'h Host = self.host;
        for handle in self.heap.freed_handles() {
            if self.handles.forget(handle)
                && let Some(gpu) = host.gpu()
            {
                gpu.release(handle);
            }
        }
    }

    pub(super) fn live_handles(&mut self) -> Result<Value, Stop> {
        self.release_freed();
        bigint::value(&mut self.heap, self.handles.live().into()).map_err(full)
    }

    pub(super) fn metal_device(&mut self) -> Result<Value, Stop> {
        let Some(gpu) = self.gpu() else {
            return self.metal_err(MetalError::Unsupported);
        };
        let id = gpu.next()?;
        match gpu.device(id) {
            Ok(info) => {
                self.handles.devices.insert(id, info);
                self.new_handle(Handle::Device(id))
            }
            Err(DeviceError::Unsupported) => self.metal_err(MetalError::Unsupported),
            Err(DeviceError::NoDevice) => self.metal_err(MetalError::NoDevice),
            Err(DeviceError::Fault(f)) => Err(fault(f)),
        }
    }

    pub(super) fn metal_name(&mut self, device: Value) -> Result<Value, Stop> {
        let (_, info) = device_of(&self.heap, &self.handles, device)?;
        let name = info.name.clone();
        Ok(Value::cell(
            self.heap.string(name.as_bytes()).map_err(full)?,
        ))
    }

    /// Every check comes before the bytes are found, so a request refused
    /// neither copies them nor counts a copy.
    pub(super) fn metal_buffer(&mut self, device: Value, bytes: Value) -> Result<Value, Stop> {
        device_of(&self.heap, &self.handles, device)?;
        let bits = self.binary(bytes)?;
        if !bits.len.is_multiple_of(8) {
            return self.metal_err(MetalError::UnalignedBinary);
        }
        let gpu = self.gpu_of_handle()?;
        let (device, info) = device_of(&self.heap, &self.handles, device)?;
        let fits = match Fits::check(device, info, bits.len / 8) {
            Ok(fits) => fits,
            Err(Unfit::Empty) => return self.metal_err(MetalError::EmptyBuffer),
            Err(Unfit::TooLarge(max_bytes)) => {
                return self.metal_err(MetalError::TooLarge { max_bytes });
            }
        };
        let id = gpu.next()?;
        let view = binary::byte_view(&self.heap, bits);
        if let Cow::Owned(copy) = &view {
            self.staged += copy.len() as u64;
        }
        let len = view.len() as u64;
        let made = match fits.bytes(&view) {
            Some(bytes) => gpu.buffer(id, bytes),
            None => {
                return Err(Stop::BadProgram(format!(
                    "a binary of {} bits, read as {len} bytes",
                    bits.len
                )));
            }
        };
        drop(view);
        match made {
            Ok(()) => {
                self.handles.buffers.insert(id, len);
                self.new_handle(Handle::Buffer(id))
            }
            Err(BufferError::OutOfMemory) => self.metal_err(MetalError::OutOfMemory),
            Err(BufferError::Fault(f)) => Err(fault(f)),
        }
    }

    /// The platform copies straight into the new binary's cell.
    pub(super) fn metal_read(&mut self, buffer: Value) -> Result<Value, Stop> {
        let (id, len) = buffer_of(&self.heap, &self.handles, buffer)?;
        let gpu = self.gpu_of_handle()?;
        let bits = len.checked_mul(8).ok_or(Stop::HeapFull)?;
        let read = binary::fill(&mut self.heap, bits, |into| {
            let to = ReadInto::check(id, len, into).ok_or_else(|| {
                Stop::BadProgram(format!("a read of {len} bytes into another length"))
            })?;
            gpu.read(to).map_err(fault)
        });
        let cell = read.map_err(full)??;
        if !binary::BYTES_IN_PLACE {
            self.staged += len;
        }
        self.ok(Value::cell(cell))
    }

    pub(super) fn metal_byte_size(&mut self, buffer: Value) -> Result<Value, Stop> {
        let (_, len) = buffer_of(&self.heap, &self.handles, buffer)?;
        bigint::value(&mut self.heap, len.into()).map_err(full)
    }

    /// `Ok` of a new cell for `handle`, which the run already holds, so that
    /// it is released even when there is no room for the cell.
    fn new_handle(&mut self, handle: Handle) -> Result<Value, Stop> {
        let cell = self.heap.handle(handle).map_err(full)?;
        self.ok(Value::cell(cell))
    }

    fn metal_err(&mut self, e: MetalError) -> Result<Value, Stop> {
        let types = self.code.abi.metal.ok_or_else(|| {
            Stop::BadProgram(
                "a Metal built-in, in a program with no `scarlet/metal.MetalError`".into(),
            )
        })?;
        let error = match e {
            MetalError::Unsupported => Value::nullary(types.unsupported),
            MetalError::NoDevice => Value::nullary(types.no_device),
            MetalError::EmptyBuffer => Value::nullary(types.empty_buffer),
            MetalError::UnalignedBinary => Value::nullary(types.unaligned_binary),
            MetalError::OutOfMemory => Value::nullary(types.out_of_memory),
            MetalError::TooLarge { max_bytes } => {
                let max = bigint::value(&mut self.heap, max_bytes.into()).map_err(full)?;
                Value::cell(self.heap.ctor(types.too_large, &[max]).map_err(full)?)
            }
        };
        self.err(error)
    }
}

/// However a run ended, with its value or a [`Stop`], the platform releases
/// every object the run still holds: in globals, in the last value, or in
/// registers a stop left behind.
impl Drop for Machine<'_, '_, '_> {
    fn drop(&mut self) {
        self.release_freed();
        let host: &Host = self.host;
        for handle in self.handles.forget_all() {
            if let Some(gpu) = host.gpu() {
                gpu.release(handle);
            }
        }
    }
}

/// The device `v` names, which must be one the run holds. The types allow
/// nothing else, so anything else is a bug in the compiler or the VM.
fn device_of<'a>(
    heap: &Heap,
    handles: &'a Handles,
    v: Value,
) -> Result<(Id<Device>, &'a DeviceInfo), Stop> {
    match v.as_cell().and_then(|c| heap.handle_of(c)) {
        Some(Handle::Device(id)) => match handles.devices.get(&id) {
            Some(info) => Ok((id, info)),
            None => Err(not_held(v)),
        },
        Some(Handle::Buffer(_)) | None => Err(wrong_kind(v, "metal.Device")),
    }
}

/// The buffer `v` names and the bytes it holds, as [`device_of`].
fn buffer_of(heap: &Heap, handles: &Handles, v: Value) -> Result<(Id<Buffer>, u64), Stop> {
    match v.as_cell().and_then(|c| heap.handle_of(c)) {
        Some(Handle::Buffer(id)) => match handles.buffers.get(&id) {
            Some(len) => Ok((id, *len)),
            None => Err(not_held(v)),
        },
        Some(Handle::Device(_)) | None => Err(wrong_kind(v, "metal.Buffer")),
    }
}

fn wrong_kind(v: Value, want: &str) -> Stop {
    Stop::BadProgram(format!("{v:?} where a {want} belongs"))
}

fn not_held(v: Value) -> Stop {
    Stop::BadProgram(format!("{v:?}, a handle this run no longer holds"))
}

fn fault(f: Fault) -> Stop {
    Stop::PlatformFault(f.to_string())
}

impl From<OutOfIds> for Stop {
    fn from(_: OutOfIds) -> Stop {
        Stop::OutOfIds
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use super::super::tests::cells_left_after_in;
    use super::*;
    use crate::fake::Fake;

    fn compile(src: &str) -> scarlet_ir::core_ir::Program {
        let mut scanner = scarlet_core::scanner::new_scanner(src.to_string());
        let parsed = scarlet_core::parser::new_parser(&mut scanner).parse_program();
        let expr = scarlet_core::ast::Expression::BlockExpression(parsed.ast);
        let result = scarlet_core::bytecode::compile(&expr, None);
        assert!(result.success(), "{:?}", result.diagnostics);
        result.into_runnable().expect("a clean compile is runnable")
    }

    fn host(platform: Option<Fake>) -> Host {
        let host = Host::new(Vec::new(), Vec::new());
        match platform {
            Some(fake) => host.with_gpu(Arc::new(Gpu::new(fake))),
            None => host,
        }
    }

    /// `src` run on `host` to its end: what it printed, and how it ended.
    fn run_on(host: &Host, src: &str) -> (String, Result<(), Stop>) {
        let mut out = Vec::new();
        let ran = crate::run(&compile(src), host, &mut out);
        (String::from_utf8(out).expect("UTF-8"), ran)
    }

    /// Every error the VM makes, once.
    const ALL: [MetalError; 6] = [
        MetalError::Unsupported,
        MetalError::NoDevice,
        MetalError::EmptyBuffer,
        MetalError::UnalignedBinary,
        MetalError::TooLarge { max_bytes: 16 },
        MetalError::OutOfMemory,
    ];

    impl MetalError {
        fn name(self) -> &'static str {
            match self {
                MetalError::Unsupported => "Unsupported",
                MetalError::NoDevice => "NoDevice",
                MetalError::EmptyBuffer => "EmptyBuffer",
                MetalError::UnalignedBinary => "UnalignedBinary",
                MetalError::TooLarge { .. } => "TooLarge",
                MetalError::OutOfMemory => "OutOfMemory",
            }
        }
    }

    /// The platform, and the call, that make each error: a new variant does
    /// not compile until it has its line here, which the test below runs.
    fn making(e: MetalError) -> (Option<Fake>, &'static str, String) {
        match e {
            MetalError::Unsupported => (None, "metal.device()", "Unsupported".into()),
            MetalError::NoDevice => (
                Some(Fake::new().without_device()),
                "metal.device()",
                "NoDevice".into(),
            ),
            MetalError::EmptyBuffer => (Some(Fake::new()), "on_device(<<>>)", "EmptyBuffer".into()),
            MetalError::UnalignedBinary => (
                Some(Fake::new()),
                "on_device(<<1, 2:4>>)",
                "UnalignedBinary".into(),
            ),
            MetalError::TooLarge { max_bytes } => (
                Some(Fake::new().with_max_buffer_bytes(max_bytes)),
                "on_device(<<'seventeen bytes!!'>>)",
                format!("TooLarge({max_bytes})"),
            ),
            MetalError::OutOfMemory => (
                Some(Fake::new().out_of_memory()),
                "on_device(<<1>>)",
                "OutOfMemory".into(),
            ),
        }
    }

    #[test]
    fn every_metal_error_has_a_program_that_makes_it() {
        for e in ALL {
            let (platform, call, shown) = making(e);
            let src = format!(
                "import scarlet/metal\n\
                 import scarlet/result\n\
                 fn on_device(bytes Binary) Result(metal.Buffer, metal.MetalError) {{\n\
                 \tdevice <- result.then(metal.device())\n\
                 \tmetal.buffer(device, bytes)\n\
                 }}\n\
                 pub fn main() {{\n\
                 \tmatch {call} {{\n\
                 \t\tOk(_) -> println('made')\n\
                 \t\tErr(e) -> println(e)\n\
                 \t}}\n\
                 }}\n"
            );
            let (out, left) = cells_left_after_in(&host(platform), &src);
            assert_eq!(out, format!("{shown}\n"), "{e:?}");
            assert_eq!(left, 0, "{e:?}");
        }
    }

    /// [`ALL`] is every variant of the stdlib's `MetalError`, by name, so a
    /// variant added there fails here until the VM makes it and it has a
    /// test.
    #[test]
    fn the_vm_makes_every_metal_error_the_stdlib_has() {
        let program = compile("import scarlet/metal\npub fn main() {}\n");
        let metal = program.abi.metal.expect("scarlet/metal's types");
        let names = &program.types[&metal.unsupported.type_id];
        assert_eq!(names.name, "MetalError");
        let stdlib: BTreeSet<&str> = names.variants.iter().map(|v| v.name.as_str()).collect();
        let ours: BTreeSet<&str> = ALL.iter().map(|e| e.name()).collect();
        assert_eq!(stdlib, ours);
        assert_eq!(ALL.len(), ours.len(), "a variant listed twice");
    }

    /// A handle of the wrong kind, or one the run no longer holds, never
    /// reaches the platform: the types make both impossible, so reaching
    /// either is a bug in the compiler or the VM, and says so.
    #[test]
    fn a_handle_of_the_wrong_kind_or_not_held_is_a_bad_program() {
        let program = compile("pub fn main() {}\n");
        let code = crate::code::load(&program);
        let host = host(Some(Fake::new()));
        let mut out = Vec::new();
        let mut m = Machine::new(&code, &host, &mut out);
        let device: Id<Device> = host.gpu().expect("a GPU").next().expect("an id");
        let info = DeviceInfo::new("a device".into(), 16);
        let cell = m.heap.handle(Handle::Device(device)).expect("room");
        let v = Value::cell(cell);
        let bad = |r: Result<Value, Stop>| matches!(r, Err(Stop::BadProgram(_)));
        assert!(bad(m.metal_name(v)), "a device the run does not hold");
        m.handles.devices.insert(device, info);
        let name = m.metal_name(v).expect("a name");
        m.release(name);
        assert!(bad(m.metal_byte_size(v)), "a device where a buffer goes");
        assert!(bad(m.metal_read(v)), "a device where a buffer goes");
        assert!(bad(m.metal_name(Value::NIL)), "Nil where a device goes");
        assert!(bad(m.metal_buffer(Value::NIL, Value::NIL)));
        m.handles.devices.clear();
        m.release(v);
    }

    /// A handle cell is freed like any other: none is left in the heap, and
    /// the fake, dropped with the host, finds nothing held.
    #[test]
    fn handle_cells_are_all_freed() {
        let (out, left) = cells_left_after_in(
            &host(Some(Fake::new())),
            "import scarlet/metal\n\
             import scarlet/result\n\
             pub fn main() {\n\
             \tprintln(result.then(metal.device(), fn(d) { metal.buffer(d, <<1, 2>>) }))\n\
             }\n",
        );
        assert_eq!(out, "Ok(<metal.Buffer #2>)\n");
        assert_eq!(left, 0);
    }

    /// A platform that has no GPU API to open says so through the same
    /// value as a host with no platform at all.
    #[test]
    fn a_platform_with_no_gpu_api_is_unsupported() {
        let (out, ran) = run_on(
            &host(Some(Fake::new().without_gpu_api())),
            "import scarlet/metal\n\
             pub fn main() {\n\
             \tprintln(metal.device())\n\
             }\n",
        );
        assert_eq!(ran, Ok(()));
        assert_eq!(out, "Err(Unsupported)\n");
    }

    /// A platform that reads fewer bytes than the buffer holds is refused,
    /// rather than leaving the rest of the new binary holding what its cell
    /// held before.
    struct Short(Fake);

    impl crate::platform::Platform for Short {
        fn device(&self, device: Id<Device>) -> Result<DeviceInfo, DeviceError> {
            self.0.device(device)
        }

        fn buffer(
            &self,
            buffer: Id<Buffer>,
            bytes: crate::platform::BufferBytes<'_>,
        ) -> Result<(), BufferError> {
            self.0.buffer(buffer, bytes)
        }

        fn read(&self, to: ReadInto<'_>) -> Result<(), Fault> {
            to.copy_from(&[1])
        }

        fn release(&self, handle: Handle) {
            self.0.release(handle);
        }
    }

    #[test]
    fn a_short_read_is_refused() {
        let host =
            Host::new(Vec::new(), Vec::new()).with_gpu(Arc::new(Gpu::new(Short(Fake::new()))));
        let (out, ran) = run_on(
            &host,
            "import scarlet/metal\n\
             import scarlet/result\n\
             pub fn main() {\n\
             \tmade = result.then(metal.device(), fn(d) { metal.buffer(d, <<1, 2, 3>>) })\n\
             \tprintln(result.then(made, metal.read))\n\
             }\n",
        );
        assert_eq!(out, "");
        assert_eq!(
            ran,
            Err(Stop::PlatformFault(
                "#2 read as 1 bytes, where it holds 3".into()
            ))
        );
    }

    /// A buffer refused copies nothing, and counts no copy, even from a
    /// binary that starts mid-byte, which one made would have been copied
    /// from.
    #[test]
    fn a_buffer_refused_stages_nothing() {
        let (out, ran) = run_on(
            &host(Some(Fake::new().with_max_buffer_bytes(16))),
            "import scarlet/binary\n\
             import scarlet/internal\n\
             import scarlet/metal\n\
             fn staged(d metal.Device, from Int, bits Int) Nil {\n\
             \tmatch binary.slice_bits(<<'twenty bytes, or so'>>, from, bits) {\n\
             \t\tOk(part) -> {\n\
             \t\t\tbefore = internal.bytes_staged()\n\
             \t\t\tmade = metal.buffer(d, part)\n\
             \t\t\tstaged = internal.bytes_staged() - before\n\
             \t\t\tprintln(staged)\n\
             \t\t\tmatch made {\n\
             \t\t\t\tOk(b) -> println(b)\n\
             \t\t\t\tErr(e) -> println(e)\n\
             \t\t\t}\n\
             \t\t}\n\
             \t\tErr(Nil) -> println('no slice')\n\
             \t}\n\
             }\n\
             pub fn main() {\n\
             \tmatch metal.device() {\n\
             \t\tOk(d) -> {\n\
             \t\t\tstaged(d, 4, 17 * 8)\n\
             \t\t\tstaged(d, 4, 0)\n\
             \t\t\tstaged(d, 4, 12)\n\
             \t\t\tstaged(d, 4, 16 * 8)\n\
             \t\t}\n\
             \t\tErr(e) -> println(e)\n\
             \t}\n\
             }\n",
        );
        assert_eq!(ran, Ok(()));
        assert_eq!(
            out,
            "0\nTooLarge(16)\n\
             0\nEmptyBuffer\n\
             0\nUnalignedBinary\n\
             16\n<metal.Buffer #2>\n"
        );
    }

    /// Ids come from a counter with the platform, and when it has given out
    /// every one it can, the run stops saying so, never reusing one.
    #[test]
    fn ids_running_out_stops_the_run() {
        let gpu = Gpu::after(Fake::new(), u64::MAX - 1);
        let host = Host::new(Vec::new(), Vec::new()).with_gpu(Arc::new(gpu));
        let (out, ran) = run_on(
            &host,
            "import scarlet/metal\n\
             pub fn main() {\n\
             \tprintln(metal.device())\n\
             \tprintln(metal.device())\n\
             }\n",
        );
        assert_eq!(out, format!("Ok(<metal.Device #{}>)\n", u64::MAX));
        assert_eq!(ran, Err(Stop::OutOfIds));
    }
}
