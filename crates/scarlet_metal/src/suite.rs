//! One suite, run against Metal and against `scarlet_vm`'s fake GPU, so the
//! fake cannot drift from what Metal does (`docs/metal-design.md`, "Tests
//! that ratchet").
//!
//! Each test is a Scarlet program the VM runs on a platform wrapped in a
//! [`Ledger`], which writes down every call and checks the VM's side of the
//! contract: each handle made once and released once, and every one released
//! by the end of the run. The harness then checks that the platform's own
//! table is empty, and a test run through [`run_test`] also that
//! `internal.live_handles()` is 0 once its work is done. So every test checks
//! for leaks without asking.
//!
//! On macOS the Metal half needs a GPU, and a Mac without one fails it,
//! saying so. Off macOS the Metal half is skipped, and says so.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::Write;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use scarlet_vm::fake::Fake;
use scarlet_vm::platform::{
    Buffer, BufferBytes, BufferError, Device, DeviceError, DeviceInfo, Fault, Handle, Id, Platform,
    ReadInto,
};
use scarlet_vm::{Host, Stop};

/// Costs, asserted exactly. One that gets worse fails its test, and one that
/// gets better fails too, until its number here is lowered with it: the
/// ratchet.
mod cost {
    /// `internal.cells_made` across one call: the handle's cell, and its `Ok`.
    pub(super) const CELLS_PER_DEVICE: u64 = 2;
    /// The handle's cell, and its `Ok`.
    pub(super) const CELLS_PER_BUFFER: u64 = 2;
    /// The binary's cell, and its `Ok`.
    pub(super) const CELLS_PER_READ: u64 = 2;
    /// The length is the VM's own record.
    pub(super) const CELLS_PER_BYTE_SIZE: u64 = 0;
    /// The name's string.
    pub(super) const CELLS_PER_NAME: u64 = 1;
    /// `internal.bytes_staged` across a buffer made from a whole binary, or a
    /// slice of one at a byte: the platform reads the binary where it is.
    pub(super) const STAGED_PER_BUFFER: u64 = 0;
    /// Across a buffer made from 4 whole bytes that start mid-byte, which
    /// are copied into line first.
    pub(super) const STAGED_PER_BUFFER_OFF_A_BYTE: u64 = 4;
    /// Across a read: the platform writes into the new binary's cell.
    pub(super) const STAGED_PER_READ: u64 = 0;
    /// Handles held at once by a loop that makes a buffer each turn and
    /// gives up the last one: the device and one buffer.
    pub(super) const HANDLES_AT_STEADY_STATE: usize = 2;
}

/// A platform the suite runs on.
trait Subject: Platform + 'static {
    fn make() -> Self;

    /// How many objects its own table holds.
    fn held(&self) -> usize;
}

impl Subject for Fake {
    fn make() -> Fake {
        Fake::new()
    }

    fn held(&self) -> usize {
        Fake::held(self)
    }
}

#[cfg(target_os = "macos")]
impl Subject for crate::metal::Metal {
    fn make() -> crate::metal::Metal {
        crate::metal::Metal::new()
    }

    fn held(&self) -> usize {
        crate::metal::Metal::held(self)
    }
}

/// How often the VM called each method, and the bytes it moved. The same
/// program makes the same calls on every platform.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Calls {
    device: u64,
    buffer: u64,
    read: u64,
    release: u64,
    bytes_in: u64,
    bytes_out: u64,
}

#[derive(Default)]
struct Log {
    /// What happened, in order, with what the program printed among it.
    events: Vec<String>,
    live: HashSet<Handle>,
    made: HashSet<Handle>,
    /// Each live buffer's length, to count what a read moves.
    lengths: HashMap<Id<Buffer>, u64>,
    /// The most handles live at once.
    peak: usize,
    wrong: Vec<String>,
    calls: Calls,
}

/// `inner`, writing down everything the VM asks of it.
struct Ledger<S> {
    inner: S,
    log: Mutex<Log>,
}

impl<S> Ledger<S> {
    fn log(&self) -> MutexGuard<'_, Log> {
        self.log.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn made(&self, handle: Handle, what: String) {
        let mut log = self.log();
        if !log.made.insert(handle) {
            log.wrong.push(format!("{handle:?} made twice"));
        }
        log.live.insert(handle);
        log.peak = log.peak.max(log.live.len());
        log.events.push(what);
    }
}

impl<S: Platform> Platform for Ledger<S> {
    fn device(&self, id: Id<Device>) -> Result<DeviceInfo, DeviceError> {
        self.log().calls.device += 1;
        let made = self.inner.device(id);
        if made.is_ok() {
            self.made(Handle::Device(id), format!("made device {id:?}"));
        }
        made
    }

    fn buffer(&self, id: Id<Buffer>, bytes: BufferBytes<'_>) -> Result<(), BufferError> {
        let len = bytes.bytes().len() as u64;
        {
            let mut log = self.log();
            log.calls.buffer += 1;
            log.calls.bytes_in += len;
        }
        let made = self.inner.buffer(id, bytes);
        if made.is_ok() {
            self.log().lengths.insert(id, len);
            self.made(Handle::Buffer(id), format!("made buffer {id:?}"));
        }
        made
    }

    fn read(&self, to: ReadInto<'_>) -> Result<(), Fault> {
        {
            let mut log = self.log();
            log.calls.read += 1;
            let len = log.lengths.get(&to.buffer()).copied().unwrap_or(0);
            log.calls.bytes_out += len;
        }
        self.inner.read(to)
    }

    fn release(&self, handle: Handle) {
        {
            let mut log = self.log();
            log.calls.release += 1;
            if !log.live.remove(&handle) {
                let wrong = if log.made.contains(&handle) {
                    format!("{handle:?} released twice")
                } else {
                    format!("{handle:?} released, never made")
                };
                log.wrong.push(wrong);
                return;
            }
            let id = match handle {
                Handle::Device(id) => format!("{id:?}"),
                Handle::Buffer(id) => {
                    log.lengths.remove(&id);
                    format!("{id:?}")
                }
            };
            log.events.push(format!("released {id}"));
        }
        self.inner.release(handle);
    }
}

/// What a program prints goes into the ledger among the platform's events,
/// so a test sees where each release falls among its lines.
struct Printer<S>(Arc<Ledger<S>>);

impl<S> Write for Printer<S> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let text = String::from_utf8_lossy(buf);
        let mut log = self.0.log();
        for line in text.lines() {
            log.events.push(format!("print {line}"));
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

struct Ran {
    events: Vec<String>,
    stop: Result<(), Stop>,
    calls: Calls,
    peak: usize,
}

impl Ran {
    /// What the program printed, a line each.
    fn printed(&self) -> Vec<&str> {
        let lines = self.events.iter();
        lines.filter_map(|e| e.strip_prefix("print ")).collect()
    }
}

fn compile(src: &str) -> scarlet_core::core_ir::Program {
    let mut scanner = scarlet_core::scanner::new_scanner(src.to_string());
    let parsed = scarlet_core::parser::new_parser(&mut scanner).parse_program();
    let expr = scarlet_core::ast::Expression::BlockExpression(parsed.ast);
    let result = scarlet_core::bytecode::compile(&expr, None);
    assert!(result.success(), "{:?}\n{src}", result.diagnostics);
    result.into_runnable().expect("a clean compile is runnable")
}

/// Run `src` on a fresh `S`, and check that the run left nothing behind: the
/// VM released every handle it made, once, and the platform's table is
/// empty.
fn run<S: Subject>(src: &str) -> Ran {
    let program = compile(src);
    let ledger = Arc::new(Ledger {
        inner: S::make(),
        log: Mutex::default(),
    });
    let stop = {
        let host = Host::new(Vec::new(), Vec::new()).with_platform(ledger.clone());
        scarlet_vm::run(&program, &host, &mut Printer(ledger.clone()))
    };
    let log = ledger.log();
    let events = log.events.clone();
    assert!(
        !events
            .iter()
            .any(|e| e == "print NoDevice" || e == "print Unsupported"),
        "this machine has no GPU Metal can use, so the Metal tests cannot run here, \
         and they do not pass without running"
    );
    assert_eq!(log.wrong, Vec::<String>::new(), "{events:#?}");
    assert!(log.live.is_empty(), "held after the run: {:?}", log.live);
    assert_eq!(ledger.inner.held(), 0, "the platform's own table");
    Ran {
        events,
        stop,
        calls: log.calls,
        peak: log.peak,
    }
}

/// Run `defs`, which define `fn test(d metal.Device) Nil`, on a device, and
/// check that nothing is left once it returns: `internal.live_handles()` is
/// 0, and then the checks of [`run`]. `test` is called in tail position, so
/// the frame holding the device's `Ok` is gone by then.
fn run_test<S: Subject>(defs: &str) -> Ran {
    let ran = run::<S>(&format!(
        "import scarlet/array\n\
         import scarlet/binary\n\
         import scarlet/crypto\n\
         import scarlet/internal\n\
         import scarlet/map\n\
         import scarlet/metal\n\
         import scarlet/result\n\
         import scarlet/string\n\
         {defs}\
         fn on_a_device() Nil {{\n\
         \tmatch metal.device() {{\n\
         \t\tOk(d) -> test(d)\n\
         \t\tErr(e) -> println(e)\n\
         \t}}\n\
         }}\n\
         pub fn main() {{\n\
         \ton_a_device()\n\
         \tprintln('live ${{internal.live_handles()}}')\n\
         }}\n"
    ));
    assert_eq!(ran.stop, Ok(()), "{:#?}", ran.events);
    assert_eq!(ran.printed().last(), Some(&"live 0"), "{:#?}", ran.events);
    ran
}

fn lines(want: &[&str]) -> Vec<String> {
    want.iter().map(|s| s.to_string()).collect()
}

fn bytes_come_back_as_they_went_in<S: Subject>() {
    let ran = run_test::<S>(
        "fn back(d metal.Device, bytes Binary) Bool {\n\
         \tresult.then(metal.buffer(d, bytes), metal.read) == Ok(bytes)\n\
         }\n\
         fn each_size(d metal.Device, sizes Array(Int)) Nil {\n\
         \tmatch sizes {\n\
         \t\t[] -> Nil\n\
         \t\t[n, ..rest] -> {\n\
         \t\t\tmatch crypto.random_bytes(n) {\n\
         \t\t\t\tOk(bytes) -> println(back(d, bytes))\n\
         \t\t\t\tErr(Nil) -> println('no random bytes')\n\
         \t\t\t}\n\
         \t\t\teach_size(d, rest)\n\
         \t\t}\n\
         \t}\n\
         }\n\
         fn test(d metal.Device) Nil {\n\
         \tprintln(result.then(metal.buffer(d, <<1, 2, 3, 4>>), metal.read))\n\
         \teach_size(d, [1, 7, 8, 9, 4096, 1048576])\n\
         \tbytes = <<1, 2, 3, 4, 5, 6>>\n\
         \tprintln(result.map(binary.slice_bits(bytes, 8, 32), fn(b) { back(d, b) }))\n\
         \tprintln(result.map(binary.slice_bits(bytes, 4, 32), fn(b) { back(d, b) }))\n\
         \tprintln(result.map(metal.buffer(d, bytes), metal.byte_size))\n\
         \tprintln(string.length(metal.name(d)) > 0)\n\
         }\n",
    );
    assert_eq!(
        ran.printed(),
        [
            "Ok(<<1, 2, 3, 4>>)",
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
            "Ok(True)",
            "Ok(True)",
            "Ok(6)",
            "True",
            "live 0",
        ]
    );
}

/// What each call costs, in cells, staged bytes and platform calls, against
/// the numbers in [`cost`].
fn each_call_costs_what_it_did<S: Subject>() {
    // Each delta is taken straight after its call, before anything else
    // can make a cell, and printed after.
    let ran = run_test::<S>(
        "fn test(d metal.Device) Nil {\n\
         \tbytes = <<1, 2, 3, 4, 5, 6, 7, 8>>\n\
         \toff_a_byte = binary.slice_bits(bytes, 4, 32)\n\
         \tcells = internal.cells_made()\n\
         \tstaged = internal.bytes_staged()\n\
         \tmade = metal.buffer(d, bytes)\n\
         \tbuffer_cells = internal.cells_made() - cells\n\
         \tbuffer_staged = internal.bytes_staged() - staged\n\
         \tprintln('buffer ${buffer_cells} ${buffer_staged}')\n\
         \tmatch made {\n\
         \t\tOk(b) -> {\n\
         \t\t\tcells = internal.cells_made()\n\
         \t\t\tstaged = internal.bytes_staged()\n\
         \t\t\tread = metal.read(b)\n\
         \t\t\tread_cells = internal.cells_made() - cells\n\
         \t\t\tread_staged = internal.bytes_staged() - staged\n\
         \t\t\tcells = internal.cells_made()\n\
         \t\t\tsize = metal.byte_size(b)\n\
         \t\t\tsize_cells = internal.cells_made() - cells\n\
         \t\t\tprintln('read ${read_cells} ${read_staged}')\n\
         \t\t\tprintln('byte_size ${size_cells} 0')\n\
         \t\t\tprintln(read == Ok(bytes) && size == 8)\n\
         \t\t}\n\
         \t\tErr(e) -> println(e)\n\
         \t}\n\
         \tcells = internal.cells_made()\n\
         \tname = metal.name(d)\n\
         \tname_cells = internal.cells_made() - cells\n\
         \tprintln('name ${name_cells} 0')\n\
         \tmatch off_a_byte {\n\
         \t\tOk(part) -> {\n\
         \t\t\tcells = internal.cells_made()\n\
         \t\t\tstaged = internal.bytes_staged()\n\
         \t\t\tmade = metal.buffer(d, part)\n\
         \t\t\tpart_cells = internal.cells_made() - cells\n\
         \t\t\tpart_staged = internal.bytes_staged() - staged\n\
         \t\t\tprintln('off a byte ${part_cells} ${part_staged}')\n\
         \t\t\tprintln(result.then(made, metal.read) == Ok(part))\n\
         \t\t}\n\
         \t\tErr(Nil) -> println('no slice')\n\
         \t}\n\
         \tprintln(string.length(name) > 0)\n\
         }\n",
    );
    let staged = |label: &str, cells: u64, staged: u64| format!("{label} {cells} {staged}");
    assert_eq!(
        ran.printed(),
        [
            staged("buffer", cost::CELLS_PER_BUFFER, cost::STAGED_PER_BUFFER),
            staged("read", cost::CELLS_PER_READ, cost::STAGED_PER_READ),
            staged("byte_size", cost::CELLS_PER_BYTE_SIZE, 0),
            "True".into(),
            staged("name", cost::CELLS_PER_NAME, 0),
            staged(
                "off a byte",
                cost::CELLS_PER_BUFFER,
                cost::STAGED_PER_BUFFER_OFF_A_BYTE
            ),
            "True".into(),
            "True".into(),
            "live 0".into(),
        ]
    );
    // One call per operation, and each byte crosses once each way: a
    // buffer's length and a device's name are the VM's own records.
    assert_eq!(
        ran.calls,
        Calls {
            device: 1,
            buffer: 2,
            read: 2,
            release: 3,
            bytes_in: 8 + 4,
            bytes_out: 8 + 4,
        }
    );
}

fn a_device_costs_what_it_did<S: Subject>() {
    let ran = run::<S>(
        "import scarlet/internal\n\
         import scarlet/metal\n\
         pub fn main() {\n\
         \tcells = internal.cells_made()\n\
         \tmade = metal.device()\n\
         \tprintln(internal.cells_made() - cells)\n\
         \tprintln(made)\n\
         }\n",
    );
    let cells = cost::CELLS_PER_DEVICE.to_string();
    assert_eq!(ran.printed(), [cells.as_str(), "Ok(<metal.Device #1>)"]);
}

/// Each handle goes at its last use, where Perceus drops it: the device once
/// the buffer is made from it, the buffer once it is read, each before the
/// next line prints.
fn a_handle_is_released_at_its_last_use<S: Subject>() {
    let ran = run_test::<S>(
        "fn test(d metal.Device) Nil {\n\
         \tmatch metal.buffer(d, <<1, 2, 3>>) {\n\
         \t\tOk(b) -> {\n\
         \t\t\tprintln('made')\n\
         \t\t\tprintln(metal.read(b))\n\
         \t\t\tprintln('read')\n\
         \t\t}\n\
         \t\tErr(e) -> println(e)\n\
         \t}\n\
         }\n",
    );
    assert_eq!(
        ran.events,
        lines(&[
            "made device #1",
            "made buffer #2",
            "released #1",
            "print made",
            "released #2",
            "print Ok(<<1, 2, 3>>)",
            "print read",
            "print live 0",
        ])
    );
}

/// Whichever arm a `match` leaves by, one that uses its buffer, one that
/// does not, or the `Err` of a buffer never made, nothing made in it
/// outlives it.
fn every_arm_releases_what_it_made<S: Subject>() {
    let ran = run_test::<S>(
        "fn size_or(d metal.Device, skip Bool) Int {\n\
         \tmatch metal.buffer(d, <<1, 2>>) {\n\
         \t\tOk(b) -> if skip { 0 } else { metal.byte_size(b) }\n\
         \t\tErr(_) -> -1\n\
         \t}\n\
         }\n\
         fn test(d metal.Device) Nil {\n\
         \tprintln(size_or(d, True))\n\
         \tprintln(internal.live_handles())\n\
         \tprintln(size_or(d, False))\n\
         \tprintln(internal.live_handles())\n\
         \tprintln(metal.buffer(d, <<>>))\n\
         \tprintln(internal.live_handles())\n\
         }\n",
    );
    assert_eq!(
        ran.events,
        lines(&[
            "made device #1",
            "made buffer #2",
            "released #2",
            "print 0",
            "print 1",
            "made buffer #3",
            "released #3",
            "print 2",
            "print 1",
            "released #1",
            "print Err(EmptyBuffer)",
            "print 0",
            "print live 0",
        ])
    );
}

/// A function that leaves early, through `<-` on an `Err`, gives up what it
/// had made before the error.
fn an_early_return_releases_what_it_made<S: Subject>() {
    let ran = run_test::<S>(
        "fn both(d metal.Device, second Binary) Result(Int, metal.MetalError) {\n\
         \ta <- result.then(metal.buffer(d, <<1>>))\n\
         \tb <- result.then(metal.buffer(d, second))\n\
         \tOk(metal.byte_size(a) + metal.byte_size(b))\n\
         }\n\
         fn test(d metal.Device) Nil {\n\
         \tprintln(both(d, <<>>))\n\
         \tprintln(internal.live_handles())\n\
         \tprintln(both(d, <<2, 3>>))\n\
         }\n",
    );
    assert_eq!(
        ran.events,
        lines(&[
            "made device #1",
            "made buffer #2",
            "released #2",
            "print Err(EmptyBuffer)",
            "print 1",
            "made buffer #3",
            "made buffer #4",
            "released #1",
            "released #4",
            "released #3",
            "print Ok(3)",
            "print live 0",
        ])
    );
}

/// A closure holds what it captured, so a buffer lives as long as the
/// closure does, and goes with it.
fn a_captured_handle_lives_as_long_as_its_closure<S: Subject>() {
    let ran = run_test::<S>(
        "fn sizer(b metal.Buffer) fn() Int {\n\
         \tfn() { metal.byte_size(b) }\n\
         }\n\
         fn twice(f fn() Int) Int { f() + f() }\n\
         fn test(d metal.Device) Nil {\n\
         \tmatch metal.buffer(d, <<1, 2, 3>>) {\n\
         \t\tOk(b) -> {\n\
         \t\t\tf = sizer(b)\n\
         \t\t\tprintln(internal.live_handles())\n\
         \t\t\tprintln(twice(f))\n\
         \t\t\tprintln(internal.live_handles())\n\
         \t\t}\n\
         \t\tErr(e) -> println(e)\n\
         \t}\n\
         }\n",
    );
    assert_eq!(
        ran.events,
        lines(&[
            "made device #1",
            "made buffer #2",
            "released #1",
            "print 1",
            "released #2",
            "print 6",
            "print 0",
            "print live 0",
        ])
    );
}

/// A handle held in a record, a tuple and an array lives while any of them
/// does, and goes when the last one does.
fn a_handle_inside_other_values_goes_with_the_last_of_them<S: Subject>() {
    let ran = run_test::<S>(
        "type Holder {\n\
         \tHolder(buf metal.Buffer, n Int)\n\
         }\n\
         fn sizes(held Holder, pair (metal.Buffer, Int), all Array(metal.Buffer)) Int {\n\
         \tmetal.byte_size(held.buf) + metal.byte_size(pair.0) + array.length(all)\n\
         }\n\
         fn test(d metal.Device) Nil {\n\
         \tmatch metal.buffer(d, <<1>>) {\n\
         \t\tOk(a) -> match metal.buffer(d, <<2, 3>>) {\n\
         \t\t\tOk(b) -> {\n\
         \t\t\t\tprintln(internal.live_handles())\n\
         \t\t\t\tprintln(sizes(Holder(buf: a, n: 1), (b, 2), [a, b]))\n\
         \t\t\t\tprintln(internal.live_handles())\n\
         \t\t\t}\n\
         \t\t\tErr(e) -> println(e)\n\
         \t\t}\n\
         \t\tErr(e) -> println(e)\n\
         \t}\n\
         }\n",
    );
    assert_eq!(
        ran.events,
        lines(&[
            "made device #1",
            "made buffer #2",
            "made buffer #3",
            "released #1",
            "print 2",
            "released #3",
            "released #2",
            "print 5",
            "print 0",
            "print live 0",
        ])
    );
}

/// A loop that makes a buffer each turn, and keeps it in a constructor whose
/// cell Perceus reuses, holds one buffer at a time however long it runs: the
/// one each turn gives up is released before the next is made.
fn a_loop_reusing_cells_holds_one_buffer_at_a_time<S: Subject>() {
    let ran = run_test::<S>(
        "type Box {\n\
         \tBox(buf metal.Buffer)\n\
         }\n\
         fn spin(d metal.Device, n Int, box Box) Int {\n\
         \tif n == 0 {\n\
         \t\tinternal.live_handles() * 100 + metal.byte_size(box.buf)\n\
         \t} else {\n\
         \t\tmatch metal.buffer(d, <<n:size(16)>>) {\n\
         \t\t\tOk(b) -> spin(d, n - 1, Box(b))\n\
         \t\t\tErr(_) -> -1\n\
         \t\t}\n\
         \t}\n\
         }\n\
         fn test(d metal.Device) Nil {\n\
         \tmatch metal.buffer(d, <<0>>) {\n\
         \t\tOk(b) -> {\n\
         \t\t\treused = internal.cells_reused()\n\
         \t\t\tprintln(spin(d, 1000, Box(b)))\n\
         \t\t\tprintln(internal.cells_reused() - reused)\n\
         \t\t}\n\
         \t\tErr(e) -> println(e)\n\
         \t}\n\
         }\n",
    );
    // In the last turn only its buffer is held, with its two bytes: the
    // device went with the turn before, its last use. Every `Box`, the
    // first one too, is built in a cell given up just before it.
    assert_eq!(ran.printed(), ["102", "1001", "live 0"]);
    assert_eq!(ran.peak, cost::HANDLES_AT_STEADY_STATE);
    assert_eq!(ran.calls.buffer, 1001);
    assert_eq!(ran.calls.release, 1002);
}

/// A run that stops early still releases every handle it holds.
fn a_run_that_stops_releases_what_it_held<S: Subject>() {
    let ran = run::<S>(
        "import scarlet/crypto\n\
         import scarlet/metal\n\
         fn go(d metal.Device) Nil {\n\
         \tmatch metal.buffer(d, <<1>>) {\n\
         \t\tOk(b) -> {\n\
         \t\t\tprintln(b)\n\
         \t\t\tprintln(crypto.random_bytes(100000000000))\n\
         \t\t\tprintln(metal.byte_size(b))\n\
         \t\t}\n\
         \t\tErr(e) -> println(e)\n\
         \t}\n\
         }\n\
         pub fn main() {\n\
         \tmatch metal.device() {\n\
         \t\tOk(d) -> go(d)\n\
         \t\tErr(e) -> println(e)\n\
         \t}\n\
         }\n",
    );
    assert_eq!(ran.stop, Err(Stop::HeapFull));
    assert_eq!(
        ran.events,
        lines(&[
            "made device #1",
            "made buffer #2",
            "released #1",
            "print <metal.Buffer #2>",
            "released #2",
        ])
    );
}

/// A handle in a global lives until the run ends, and is released then.
fn a_handle_in_a_global_is_released_when_the_run_ends<S: Subject>() {
    let ran = run::<S>(
        "import scarlet/internal\n\
         import scarlet/metal\n\
         const gpu = metal.device()\n\
         fn size(d metal.Device) Int {\n\
         \tmatch metal.buffer(d, <<1, 2>>) {\n\
         \t\tOk(b) -> metal.byte_size(b)\n\
         \t\tErr(_) -> 0\n\
         \t}\n\
         }\n\
         pub fn main() {\n\
         \tmatch gpu {\n\
         \t\tOk(d) -> println(size(d) + size(d))\n\
         \t\tErr(e) -> println(e)\n\
         \t}\n\
         \tprintln(internal.live_handles())\n\
         }\n",
    );
    assert_eq!(ran.stop, Ok(()));
    assert_eq!(
        ran.events,
        lines(&[
            "made device #1",
            "made buffer #2",
            "released #2",
            "made buffer #3",
            "released #3",
            "print 4",
            "print 1",
            "released #1",
        ])
    );
}

/// A handle prints as its type and id, equals only itself, and hashes by
/// its id, so it can be a map key.
fn a_handle_equals_only_itself<S: Subject>() {
    let ran = run::<S>(
        "import scarlet/map\n\
         import scarlet/metal\n\
         fn compare(a metal.Device, b metal.Device) Nil {\n\
         \tprintln(a)\n\
         \tprintln(a == a)\n\
         \tprintln(a == b)\n\
         \tmatch (metal.buffer(a, <<1>>), metal.buffer(a, <<1>>)) {\n\
         \t\t(Ok(x), Ok(y)) -> {\n\
         \t\t\tprintln([x, y])\n\
         \t\t\tprintln(x == y)\n\
         \t\t\tprintln(metal.read(x) == metal.read(y))\n\
         \t\t\tm = map.set(map.set(map.new(), x, 'x'), y, 'y')\n\
         \t\t\tprintln(map.get(m, x))\n\
         \t\t\tprintln(map.get(m, y))\n\
         \t\t}\n\
         \t\t_ -> println('no buffers')\n\
         \t}\n\
         }\n\
         pub fn main() {\n\
         \tmatch (metal.device(), metal.device()) {\n\
         \t\t(Ok(a), Ok(b)) -> compare(a, b)\n\
         \t\t_ -> println('no devices')\n\
         \t}\n\
         }\n",
    );
    assert_eq!(
        ran.printed(),
        [
            "<metal.Device #1>",
            "True",
            "False",
            "[<metal.Buffer #3>, <metal.Buffer #4>]",
            "False",
            "True",
            "Some(x)",
            "Some(y)",
        ]
    );
}

/// A small seeded generator (xorshift64*), so a failing program can be made
/// again from its seed.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as usize % n
    }
}

/// A value a generated program binds, and the buffers it holds.
struct Bound {
    buffers: BTreeSet<usize>,
    /// A `Result(metal.Buffer, _)`, rather than a tuple or an array of them.
    result: bool,
}

/// One statement of a generated program.
#[derive(Default)]
struct Step {
    /// The value it binds, if it binds one, to `text`.
    binds: Option<usize>,
    text: String,
    uses: Vec<usize>,
    /// Whether it makes a buffer, which uses the device.
    makes: bool,
    /// Which check it is, if it prints `internal.live_handles()`.
    check: Option<usize>,
}

/// A straight-line `test` that makes buffers, shares them into tuples and
/// arrays, uses some and drops the rest, and at each check prints
/// `internal.live_handles()`; and, for each check, what it must print. A
/// handle is live exactly while a value bound so far that holds it is used
/// later, since Perceus drops each value at its last use.
fn random_program(seed: u64) -> (String, BTreeMap<usize, usize>) {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut bound: Vec<Bound> = Vec::new();
    let mut steps: Vec<Step> = Vec::new();
    let mut checks = 0;
    let mut buffers = 0;
    for _ in 0..40 {
        let v = bound.len();
        let results: Vec<usize> = (0..v).filter(|&r| bound[r].result).collect();
        match rng.below(10) {
            0..=3 => {
                steps.push(Step {
                    binds: Some(v),
                    text: format!("metal.buffer(d, <<{}>>)", buffers % 256),
                    makes: true,
                    ..Step::default()
                });
                bound.push(Bound {
                    buffers: BTreeSet::from([buffers]),
                    result: true,
                });
                buffers += 1;
            }
            4 if v > 0 => {
                let (a, b) = (rng.below(v), rng.below(v));
                steps.push(Step {
                    binds: Some(v),
                    text: format!("(v{a}, v{b})"),
                    uses: vec![a, b],
                    ..Step::default()
                });
                let held = &bound[a].buffers | &bound[b].buffers;
                bound.push(Bound {
                    buffers: held,
                    result: false,
                });
            }
            5 if !results.is_empty() => {
                let picked: Vec<usize> =
                    (0..3).map(|_| results[rng.below(results.len())]).collect();
                let names: Vec<String> = picked.iter().map(|p| format!("v{p}")).collect();
                let held = picked.iter().flat_map(|p| bound[*p].buffers.clone());
                bound.push(Bound {
                    buffers: held.collect(),
                    result: false,
                });
                steps.push(Step {
                    binds: Some(v),
                    text: format!("[{}]", names.join(", ")),
                    uses: picked,
                    ..Step::default()
                });
            }
            6 if !results.is_empty() => {
                let r = results[rng.below(results.len())];
                steps.push(Step {
                    text: format!("println(result.map(v{r}, metal.byte_size))"),
                    uses: vec![r],
                    ..Step::default()
                });
            }
            7 if v > 0 => {
                let r = rng.below(v);
                steps.push(Step {
                    text: format!("println(v{r} == v{r})"),
                    uses: vec![r],
                    ..Step::default()
                });
            }
            _ => {
                steps.push(Step {
                    text: format!("println('check {checks} ${{internal.live_handles()}}')"),
                    check: Some(checks),
                    ..Step::default()
                });
                checks += 1;
            }
        }
    }
    let mut want = BTreeMap::new();
    for (at, step) in steps.iter().enumerate() {
        let Some(check) = step.check else {
            continue;
        };
        let (before, after) = steps.split_at(at + 1);
        let bound_so_far: BTreeSet<usize> = before.iter().filter_map(|s| s.binds).collect();
        let held: BTreeSet<usize> = after
            .iter()
            .flat_map(|s| s.uses.iter())
            .filter(|v| bound_so_far.contains(v))
            .flat_map(|v| bound[*v].buffers.iter().copied())
            .collect();
        let device = after.iter().any(|s| s.makes);
        want.insert(check, held.len() + usize::from(device));
    }
    // A binding nothing uses is an error unless its name says so.
    let used: BTreeSet<usize> = steps.iter().flat_map(|s| s.uses.clone()).collect();
    let mut body = String::from("fn test(d metal.Device) Nil {\n");
    for step in &steps {
        let line = match step.binds {
            Some(v) if used.contains(&v) => format!("v{v} = {}", step.text),
            Some(v) => format!("_v{v} = {}", step.text),
            None => step.text.clone(),
        };
        body.push_str(&format!("\t{line}\n"));
    }
    body.push_str("\tprintln('end')\n}\n");
    (body, want)
}

/// Random programs making, sharing and dropping handles: each check prints
/// what liveness says it must, and the harness finds every handle released
/// once and nothing held at the end.
fn random_lifetimes_release_each_handle_once_at_its_last_use<S: Subject>() {
    for seed in 1..=40 {
        let (body, want) = random_program(seed);
        let ran = run_test::<S>(&body);
        let got: BTreeMap<usize, usize> = ran
            .printed()
            .iter()
            .filter_map(|line| {
                let mut words = line.strip_prefix("check ")?.split(' ');
                let check = words.next()?.parse().ok()?;
                Some((check, words.next()?.parse().ok()?))
            })
            .collect();
        assert_eq!(got, want, "seed {seed}:\n{body}");
        assert_eq!(ran.calls.release, ran.calls.device + ran.calls.buffer);
    }
}

macro_rules! suite {
    ($subject:ty) => {
        suite!(@each $subject;
            bytes_come_back_as_they_went_in,
            each_call_costs_what_it_did,
            a_device_costs_what_it_did,
            a_handle_is_released_at_its_last_use,
            every_arm_releases_what_it_made,
            an_early_return_releases_what_it_made,
            a_captured_handle_lives_as_long_as_its_closure,
            a_handle_inside_other_values_goes_with_the_last_of_them,
            a_loop_reusing_cells_holds_one_buffer_at_a_time,
            a_run_that_stops_releases_what_it_held,
            a_handle_in_a_global_is_released_when_the_run_ends,
            a_handle_equals_only_itself,
            random_lifetimes_release_each_handle_once_at_its_last_use,
        );
    };
    (@each $subject:ty; $($test:ident),* $(,)?) => {
        $(
            #[test]
            fn $test() {
                super::$test::<$subject>();
            }
        )*
    };
}

mod on_fake {
    suite!(scarlet_vm::fake::Fake);
}

#[cfg(target_os = "macos")]
mod on_metal {
    suite!(crate::metal::Metal);

    /// The suite above, again, under Metal's validation layer in assert
    /// mode, where a request Metal would refuse stops the whole process
    /// (`docs/metal-design.md`, "The rule, applied to the GPU"). An abort
    /// there is a check the VM is missing, and fails this test.
    #[test]
    fn the_suite_passes_under_the_validation_layer() {
        if std::env::var_os("MTL_DEBUG_LAYER").is_some() {
            // This process is the child, or already runs under the layer.
            return;
        }
        let exe = std::env::current_exe().expect("this test binary");
        let out = std::process::Command::new(exe)
            .arg("suite::on_metal::")
            .args(["--skip", "the_suite_passes_under_the_validation_layer"])
            .env("MTL_DEBUG_LAYER", "1")
            .env("MTL_DEBUG_LAYER_ERROR_MODE", "assert")
            .output()
            .expect("the suite runs");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success(),
            "under the validation layer: {:?}\n{stdout}\n{stderr}",
            out.status
        );
        assert!(
            stderr.contains("Metal API Validation Enabled"),
            "the validation layer did not come on: {stderr}"
        );
        let passed = stdout.lines().find_map(|l| {
            l.strip_prefix("test result: ok. ")?
                .split(' ')
                .next()?
                .parse()
                .ok()
        });
        assert_eq!(passed, Some(13usize), "{stdout}");
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
#[ignore = "Metal exists only on macOS: this machine ran the suite on the fake alone"]
fn on_metal() {}
