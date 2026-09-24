//! `@embed('path') const x String`: a file read into a constant at compile
//! time.
//!
//! The compiler reads the file once, in the check pass, and decodes it for the
//! const's annotation; elaboration pools what was read and never touches the
//! file. The file is keyed by its resolved path, never the path as written,
//! and that path is a dependency of the module that names it: a change to the
//! file recompiles the module (see [`ModuleOrigin`](super::ModuleOrigin)).

use std::fmt;
use std::path::{Path, PathBuf};

use scarlet_syntax::module_path::lexical_absolute;

use super::{FileStat, bytes_hash};

/// What an `@embed` const holds, chosen by its annotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EmbedAs {
    /// The file's text, which must be UTF-8.
    String,
    /// The file's bytes, as they are.
    Binary,
}

/// The contents of an embedded file, decoded for its const's annotation.
/// Holding a `String` is the proof the file was UTF-8 when it was read.
#[derive(Debug)]
pub(crate) enum Embedded {
    String(String),
    Binary(Vec<u8>),
}

/// A file a module embedded, recorded on its
/// [`ModuleOrigin::File`](super::ModuleOrigin) so a change to it recompiles
/// the module.
#[derive(Debug, Clone)]
pub struct EmbeddedFile {
    /// Resolved: absolute and lexically normalised. The file's identity.
    pub(crate) path: PathBuf,
    /// [`bytes_hash`] of the bytes read, or `None` when the read failed: a
    /// missing file that appears later is a change too.
    pub(crate) hash: Option<u64>,
    /// `(mtime, len, ino)` as of the last time the file hashed equal to
    /// `hash`; `None` until then. See
    /// [`ModuleTable::source_changed`](super::ModuleTable::source_changed).
    pub(crate) stat: Option<FileStat>,
}

impl EmbeddedFile {
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

/// Why a path an `@embed` names cannot be resolved to a file.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum EmbedPathError {
    /// The module has no directory: an embedded stdlib module, or a buffer
    /// that was never saved.
    NoDirectory,
    Empty,
    Absolute,
    /// The directory joined with the path has no absolute form.
    Unresolvable,
}

/// Resolve `written` against `dir`, the directory of the module's own file.
///
/// Normalised as text, as a module path is ([`lexical_absolute`]): `..` is
/// allowed, symlinks are not followed, and an absolute path is refused, so the
/// program means the same file wherever it is built from.
pub(crate) fn resolve_path(dir: Option<&Path>, written: &str) -> Result<PathBuf, EmbedPathError> {
    if written.is_empty() {
        return Err(EmbedPathError::Empty);
    }
    let rel = Path::new(written);
    // `has_root` as well as `is_absolute`: on Windows `\x` is not absolute but
    // still ignores the directory it is joined to.
    if rel.is_absolute() || rel.has_root() {
        return Err(EmbedPathError::Absolute);
    }
    let dir = dir.ok_or(EmbedPathError::NoDirectory)?;
    lexical_absolute(&dir.join(rel)).map_err(|_| EmbedPathError::Unresolvable)
}

/// Why an embedded file could not become its const's value.
#[derive(Debug)]
pub(crate) enum ReadError {
    Io(std::io::Error),
    /// More bytes than one `String` or `Binary` holds.
    TooLarge {
        len: u64,
        max: usize,
    },
    /// Embedded as a `String`, but not UTF-8 from byte `offset` on.
    NotUtf8 {
        offset: usize,
    },
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadError::Io(e) => write!(f, "{e}"),
            ReadError::TooLarge { len, max } => write!(
                f,
                "it is {len} bytes, and one `String` or `Binary` holds at most {max}"
            ),
            ReadError::NotUtf8 { offset } => write!(
                f,
                "it is not UTF-8 (the first invalid byte is at offset {offset}); \
                 embed it as a `Binary` instead"
            ),
        }
    }
}

/// What one read of an embedded file found: the dependency to record, whatever
/// happened, and the value or why there is none.
pub(crate) struct Read {
    pub(crate) file: EmbeddedFile,
    pub(crate) value: Result<Embedded, ReadError>,
}

/// Read the file at `path` (already resolved) as `kind`, refusing one longer
/// than `max` bytes. Nothing is changed: no line-ending conversion, and a
/// byte-order mark stays.
pub(crate) fn read(path: PathBuf, kind: EmbedAs, max: usize) -> Read {
    let bytes = read_capped(&path, max);
    let hash = bytes.as_ref().ok().map(|b| bytes_hash(b));
    let value = bytes.and_then(|bytes| match kind {
        EmbedAs::Binary => Ok(Embedded::Binary(bytes)),
        EmbedAs::String => {
            String::from_utf8(bytes)
                .map(Embedded::String)
                .map_err(|e| ReadError::NotUtf8 {
                    offset: e.utf8_error().valid_up_to(),
                })
        }
    });
    Read {
        file: EmbeddedFile {
            path,
            hash,
            stat: None,
        },
        value,
    }
}

/// The file's bytes, checking the length before reading so a file too large
/// to embed is never held in memory whole, and again after, since the file
/// can grow in between.
fn read_capped(path: &Path, max: usize) -> Result<Vec<u8>, ReadError> {
    let too_large = |len: u64| ReadError::TooLarge { len, max };
    let len = std::fs::metadata(path).map_err(ReadError::Io)?.len();
    if len > max as u64 {
        return Err(too_large(len));
    }
    let bytes = std::fs::read(path).map_err(ReadError::Io)?;
    if bytes.len() > max {
        return Err(too_large(bytes.len() as u64));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let d = std::env::temp_dir().join(format!("al_embed_{tag}_{}_{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_path_resolves_against_the_directory_as_text() {
        let base = Path::new("/proj/src");
        assert_eq!(
            resolve_path(Some(base), "shaders/./world.metal"),
            Ok(PathBuf::from("/proj/src/shaders/world.metal"))
        );
        assert_eq!(
            resolve_path(Some(base), "../assets/x.bin"),
            Ok(PathBuf::from("/proj/assets/x.bin"))
        );
        // Two spellings of one file are one path.
        assert_eq!(
            resolve_path(Some(base), "a/../b.txt"),
            resolve_path(Some(base), "./b.txt")
        );
        // `..` past the root stays at the root.
        assert_eq!(
            resolve_path(Some(Path::new("/")), "../../x"),
            Ok(PathBuf::from("/x"))
        );
    }

    #[test]
    fn a_path_that_names_no_file_relative_to_the_module_is_refused() {
        assert_eq!(
            resolve_path(Some(Path::new("/p")), "/etc/passwd"),
            Err(EmbedPathError::Absolute)
        );
        assert_eq!(
            resolve_path(Some(Path::new("/p")), ""),
            Err(EmbedPathError::Empty)
        );
        assert_eq!(
            resolve_path(None, "x.txt"),
            Err(EmbedPathError::NoDirectory)
        );
    }

    #[test]
    fn bytes_are_kept_as_they_are() {
        let d = dir("verbatim");
        let p = d.join("crlf.txt");
        let text = "\u{feff}line one\r\nline two\r\n";
        std::fs::write(&p, text).unwrap();
        match read(p.clone(), EmbedAs::String, 1024).value {
            Ok(Embedded::String(s)) => assert_eq!(s, text),
            other => panic!("{other:?}"),
        }
        match read(p, EmbedAs::Binary, 1024).value {
            Ok(Embedded::Binary(b)) => assert_eq!(b, text.as_bytes()),
            other => panic!("{other:?}"),
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_string_must_be_utf8_and_says_where_it_is_not() {
        let d = dir("utf8");
        let p = d.join("x.bin");
        std::fs::write(&p, b"abc\xffdef").unwrap();
        let r = read(p.clone(), EmbedAs::String, 1024);
        let err = r.value.expect_err("not UTF-8");
        assert_eq!(
            err.to_string(),
            "it is not UTF-8 (the first invalid byte is at offset 3); embed it as a `Binary` instead"
        );
        // The dependency is recorded all the same: the file was read.
        assert_eq!(r.file.hash, Some(bytes_hash(b"abc\xffdef")));
        assert!(matches!(
            read(p, EmbedAs::Binary, 1024).value,
            Ok(Embedded::Binary(_))
        ));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The real limit is `MAX_CONST_BYTES`, about 2 GB; the check is the same
    /// at any size.
    #[test]
    fn a_file_past_the_limit_is_refused() {
        let d = dir("size");
        let p = d.join("big.bin");
        std::fs::write(&p, [0u8; 9]).unwrap();
        let r = read(p.clone(), EmbedAs::Binary, 8);
        assert_eq!(
            r.value.expect_err("too large").to_string(),
            "it is 9 bytes, and one `String` or `Binary` holds at most 8"
        );
        assert!(r.file.hash.is_none());
        assert!(read(p, EmbedAs::Binary, 9).value.is_ok());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_missing_file_is_a_recorded_dependency_with_no_hash() {
        let d = dir("missing");
        let r = read(d.join("nope.txt"), EmbedAs::String, 1024);
        assert!(matches!(r.value, Err(ReadError::Io(_))));
        assert_eq!(r.file.path, d.join("nope.txt"));
        assert!(r.file.hash.is_none());
        let _ = std::fs::remove_dir_all(&d);
    }
}
