//! Metal hands some objects back autoreleased, like a device's name, and
//! `scarlet run`'s thread has no autorelease pool of its own. Each `Platform`
//! method drains a pool of its own, so a program asking for devices over and
//! over holds no more memory at the end than after its first few. Its own
//! test binary, so nothing else in the process moves its memory.

#![cfg(target_os = "macos")]

use scarlet_vm::Host;

/// The most this process may grow over [`MANY`] devices and their names.
/// With a pool it grows by under 200 KB; with none, what Metal autoreleases
/// for each device is kept, about 300 bytes a device, and it grows by 8.9 MB.
const MOST_GROWTH: i64 = 2 << 20;

const MANY: u32 = 30_000;

fn devices(n: u32) -> String {
    format!(
        "import scarlet/metal\n\
         import scarlet/string\n\
         fn go(n Int, total Int) Int {{\n\
         \tif n == 0 {{\n\
         \t\ttotal\n\
         \t}} else {{\n\
         \t\tmatch metal.device() {{\n\
         \t\t\tOk(d) -> go(n - 1, total + string.length(metal.name(d)))\n\
         \t\t\tErr(e) -> {{\n\
         \t\t\t\tprintln(e)\n\
         \t\t\t\t-1\n\
         \t\t\t}}\n\
         \t\t}}\n\
         \t}}\n\
         }}\n\
         pub fn main() {{\n\
         \tprintln(go({n}, 0) > 0)\n\
         }}\n"
    )
}

fn run(host: &Host, src: &str) {
    let mut scanner = scarlet_core::scanner::new_scanner(src.to_string());
    let parsed = scarlet_core::parser::new_parser(&mut scanner).parse_program();
    let expr = scarlet_core::ast::Expression::BlockExpression(parsed.ast);
    let result = scarlet_core::bytecode::compile(&expr, None);
    assert!(result.success(), "{:?}", result.diagnostics);
    let program = result.into_runnable().expect("a clean compile is runnable");
    let mut out = Vec::new();
    assert_eq!(scarlet_vm::run(&program, host, &mut out), Ok(()));
    assert_eq!(String::from_utf8_lossy(&out), "True\n");
}

/// The most memory this process has held at once, in bytes.
fn peak_rss() -> i64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `usage` is valid for writes of one `rusage`, which is all
    // `getrusage` writes.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    assert_eq!(rc, 0, "getrusage");
    // SAFETY: `getrusage` returned 0, so it filled `usage`.
    let usage = unsafe { usage.assume_init() };
    // In bytes on macOS.
    usage.ru_maxrss
}

#[test]
fn asking_for_devices_over_and_over_holds_no_more_memory() {
    let gpu = scarlet_metal::platform().expect("macOS has a platform");
    let host = Host::new(Vec::new(), Vec::new()).with_gpu(gpu);
    run(&host, &devices(1_000));
    let before = peak_rss();
    run(&host, &devices(MANY));
    let grew = peak_rss() - before;
    assert!(
        grew < MOST_GROWTH,
        "{MANY} devices grew the process by {grew} bytes"
    );
}
