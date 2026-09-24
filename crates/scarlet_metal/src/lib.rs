//! `scarlet/metal`'s platform: `scarlet_vm`'s [`Platform`] over Apple's
//! Metal (`docs/metal-design.md`).
//!
//! This is the one crate in the workspace allowed `unsafe`. Each `unsafe`
//! block holds one operation and says why it is sound. Two lines of defence
//! keep Metal from ending the process: the VM checks every request before it
//! gets here, and hands it over in types that say so; and every Objective-C
//! call runs inside `objc2::exception::catch`, so an exception is a
//! [`scarlet_vm::platform::Fault`] rather than an abort.
//!
//! Off macOS the crate is [`platform`] alone, and there is no platform.

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
#![deny(
    unsafe_op_in_unsafe_fn,
    clippy::undocumented_unsafe_blocks,
    clippy::missing_safety_doc,
    clippy::multiple_unsafe_ops_per_block,
    clippy::cast_possible_truncation,
    clippy::wildcard_enum_match_arm
)]

use std::sync::Arc;

use scarlet_vm::platform::Platform;

#[cfg(target_os = "macos")]
mod metal;
#[cfg(test)]
mod suite;

/// This machine's GPU, for a [`scarlet_vm::Host`]: Metal on macOS, and
/// `None` anywhere else, where `metal.device` is `Err(Unsupported)`. Making
/// one asks Metal nothing, so a program that never calls `metal.device`
/// never loads Metal.
pub fn platform() -> Option<Arc<dyn Platform>> {
    #[cfg(target_os = "macos")]
    {
        Some(Arc::new(metal::Metal::new()))
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}
