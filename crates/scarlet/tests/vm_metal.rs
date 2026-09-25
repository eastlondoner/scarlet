//! `scarlet run` reaching Metal through the real platform: the driver
//! installs `scarlet_metal`'s into every run. `examples/metal.scrl` prints
//! one thing on a Mac and another anywhere else, so it is checked here, per
//! platform, rather than against one golden.

use std::path::PathBuf;
use std::process::Command;

fn example() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/metal.scrl")
}

/// `scarlet run examples/metal.scrl` with `env` set: its stdout, after
/// checking it exited cleanly, saying on stderr exactly the lines that each
/// contain one of `said`, in order.
fn run(env: &[(&str, &str)], said: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_scarlet"))
        .arg("run")
        .arg(example())
        .envs(env.iter().copied())
        .output()
        .expect("scarlet runs");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{:?}\n{stderr}", out.status);
    let lines: Vec<&str> = stderr.lines().collect();
    assert_eq!(lines.len(), said.len(), "stderr: {stderr}");
    for (line, want) in lines.iter().zip(said) {
        assert!(
            line.contains(want),
            "{line:?} does not say {want:?}: {stderr}"
        );
    }
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// A Mac without a GPU Metal can use fails here, saying so, rather than
/// passing without having reached Metal.
#[cfg(target_os = "macos")]
fn assert_round_trip(stdout: &str) {
    assert_ne!(
        stdout, "Err(NoDevice)\n",
        "this Mac has no Metal device, so the round trip cannot be tested here"
    );
    assert_eq!(stdout, "Ok(<<1, 2, 3, 4>>)\n");
}

#[cfg(target_os = "macos")]
#[test]
fn bytes_go_through_metal_and_back() {
    assert_round_trip(&run(&[], &[]));
}

/// The same run under Metal's validation layer in assert mode, where a
/// request Metal would refuse stops the whole process. The layer says it is
/// on, so a run where it did not come on fails here.
#[cfg(target_os = "macos")]
#[test]
fn the_round_trip_passes_the_validation_layer() {
    let env = [
        ("MTL_DEBUG_LAYER", "1"),
        ("MTL_DEBUG_LAYER_ERROR_MODE", "assert"),
    ];
    assert_round_trip(&run(&env, &["Metal API Validation Enabled"]));
}

/// The binary does not load Metal, or the Foundation it brings, when it
/// starts: `scarlet_metal` opens Metal when a program first asks for a
/// device, so a program that never does skips the 1.5 ms loading them costs.
/// Only an optimised build: unoptimised, objc2-foundation's dead code still
/// names a Foundation symbol, so the linker keeps it. The Release workflow
/// runs this with `cargo test --release`.
#[cfg(all(target_os = "macos", not(debug_assertions)))]
#[test]
fn the_binary_does_not_load_metal_at_launch() {
    let out = Command::new("otool")
        .arg("-L")
        .arg(env!("CARGO_BIN_EXE_scarlet"))
        .output()
        .expect("otool runs");
    let libraries = String::from_utf8_lossy(&out.stdout);
    assert!(libraries.contains("libSystem"), "{libraries}");
    assert!(!libraries.contains("Metal.framework"), "{libraries}");
    assert!(!libraries.contains("Foundation.framework"), "{libraries}");
}

#[cfg(not(target_os = "macos"))]
#[test]
fn off_macos_metal_is_unsupported() {
    assert_eq!(run(&[], &[]), "Err(Unsupported)\n");
}
