//! Stamp `SCARLET_VERSION` for `scarlet --version`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    emit_version();
    // Drop the load of any library nothing in the binary calls into. Metal
    // and the Foundation it brings are opened by `scarlet_metal` when a
    // program first asks for a device, so a run that never does skips loading
    // them: `scarlet run examples/hello.scrl` takes 3.6 ms rather than 4.6 ms
    // on an M1 Max (`crates/scarlet_metal/src/metal.rs`).
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg-bins=-Wl,-dead_strip_dylibs");
    }
}

/// `SCARLET_VERSION` for `scarlet --version`.
///
/// Canary CI rewrites `VERSION` to `0.0.1-canary.YYYYMMDD.HHMM` before
/// `cargo build`, so a file that already carries `-` or `+` is the published
/// stamp and is used as-is. A bare `VERSION` is a source tree: append
/// `+git.<sha>` so a local binary can name the commit it was built from.
fn emit_version() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let repo = manifest.join("../..");
    let version_path = repo.join("VERSION");
    println!("cargo:rerun-if-changed={}", version_path.display());
    watch_git_head(&repo);

    let base = std::fs::read_to_string(&version_path)
        .unwrap_or_else(|e| panic!("read VERSION: {e}"))
        .trim()
        .to_string();
    let stamp = if base.contains('-') || base.contains('+') {
        base
    } else {
        match git_stamp(&repo) {
            Some(git) => format!("{base}+git.{git}"),
            None => base,
        }
    };
    println!("cargo:rustc-env=SCARLET_VERSION={stamp}");
}

fn watch_git_head(repo: &Path) {
    let git = repo.join(".git");
    println!("cargo:rerun-if-changed={}", git.display());
    let head = if git.is_dir() {
        git.join("HEAD")
    } else {
        let Ok(contents) = std::fs::read_to_string(&git) else {
            return;
        };
        let Some(dir) = contents.strip_prefix("gitdir:") else {
            return;
        };
        PathBuf::from(dir.trim()).join("HEAD")
    };
    println!("cargo:rerun-if-changed={}", head.display());
    if let Ok(contents) = std::fs::read_to_string(&head)
        && let Some(r) = contents.strip_prefix("ref: ")
    {
        println!(
            "cargo:rerun-if-changed={}",
            head.parent()
                .unwrap_or(Path::new("."))
                .join(r.trim())
                .display()
        );
    }
}

fn git_stamp(repo: &Path) -> Option<String> {
    let repo = repo.to_str()?;
    let sha = git_stdout(["-C", repo, "rev-parse", "--short=12", "HEAD"])?;
    if sha.is_empty() {
        return None;
    }
    let dirty = git_stdout(["-C", repo, "status", "--porcelain"]).is_some_and(|s| !s.is_empty());
    if dirty {
        Some(format!("{sha}.dirty"))
    } else {
        Some(sha)
    }
}

fn git_stdout<const N: usize>(args: [&str; N]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout)
        .ok()
        .map(|s| s.trim().to_string())
}
