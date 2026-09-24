#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable,
        clippy::todo,
        clippy::unimplemented,
    )
)]
// The CLI/REPL/LSP driver has no use for unsafe code.
#![forbid(unsafe_code)]

pub use scarlet_core::*;

pub mod cli;
pub mod dis;
pub mod lsp;
pub mod repl;
pub mod stop;

/// The world a program run from here sees: this OS process's, with `argv`,
/// and this machine's GPU when it has one Scarlet can reach.
pub fn host(argv: Vec<String>) -> scarlet_vm::Host {
    let host = scarlet_vm::Host::of_this_process(argv);
    match scarlet_metal::platform() {
        Some(platform) => host.with_platform(platform),
        None => host,
    }
}
