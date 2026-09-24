//! Binaries: strings of bits, as the old VM had them (`vm/binary.rs` on
//! master before the rewrite).
//!
//! Bits are counted from the most significant bit of the first byte, and the
//! bits past a binary's end in its last byte are zero.
//!
//! A binary is one of two cells:
//!
//! - an **owner** holds the bits: a word with its length in bits, then its
//!   bytes, eight to a word, the first byte in the lowest bits of a word;
//! - a **slice** holds a reference to an owner, a bit offset and a length in
//!   bits. `binary.slice_bits` and a pattern's `rest:binary` make one, so they
//!   copy nothing, as their docs promise. A slice of a slice points at the
//!   owner, so there is never a chain of them.
//!
//! [`Bits`] is either, as "these bits of this owner". Everything that reads a
//! binary reads it through [`Bits`], so it does not matter which one it is.

use std::borrow::Cow;

use num_bigint::{BigInt, Sign};

use crate::heap::{Cell, Full, Heap, Kind};
use crate::value::Value;

/// The most bits one binary can hold: a cell is at most 2^28 words, and a
/// word holds 64 bits. A longer one is a full heap, not a binary.
pub(crate) const MAX_BITS: u64 = (1 << 28) * 64;

/// `len` bits of `owner`'s bytes, from bit `at`.
#[derive(Clone, Copy)]
pub(crate) struct Bits {
    owner: Cell,
    at: u64,
    pub(crate) len: u64,
}

/// The most bytes a binary can hold: a cell's worth of words, less its header
/// and the word holding its length.
pub(crate) const MAX_BYTES: usize = (crate::heap::MAX_CELL_WORDS - 2) * 8;

/// The bits `cell` holds, or `None` when it is not a binary.
pub(crate) fn bits(heap: &Heap, cell: Cell) -> Option<Bits> {
    let d = heap.data(cell);
    match heap.kind(cell)? {
        Kind::Binary => Some(Bits {
            owner: cell,
            at: 0,
            len: d.first().copied().unwrap_or(0),
        }),
        Kind::BinarySlice => Some(Bits {
            owner: Value::from_bits(d.first().copied()?).as_cell()?,
            at: d.get(1).copied()?,
            len: d.get(2).copied()?,
        }),
        Kind::String
        | Kind::BigInt
        | Kind::Ctor
        | Kind::Closure
        | Kind::Tuple
        | Kind::ArrayRoot
        | Kind::ArrayLeaf
        | Kind::ArrayBranch
        | Kind::Range
        | Kind::Map
        | Kind::MapNode
        | Kind::MapCollision => None,
    }
}

/// A new owner holding the first `len` bits of `bytes`, read as zeros past
/// its end. The bits past `len` in the last byte are cleared.
pub(crate) fn make(heap: &mut Heap, bytes: &[u8], len: u64) -> Result<Cell, Full> {
    fill(heap, len, |out| {
        let n = out.len().min(bytes.len());
        let (copied, rest) = out.split_at_mut(n);
        copied.copy_from_slice(bytes.get(..n).unwrap_or(&[]));
        rest.fill(0);
        Ok::<(), Full>(())
    })?
}

/// Whether an owner's bytes lie in its words in order, so they can be read
/// and filled as a slice of its own memory. Its words hold its bytes first
/// byte lowest, which is their order in memory on a little-endian host. On a
/// big-endian one, [`byte_view`] and [`fill`] go through a copy.
const BYTES_IN_PLACE: bool = cfg!(target_endian = "little");

/// A new owner of `len` bits, whose bytes `write` puts straight into the
/// cell ([`BYTES_IN_PLACE`]): it is handed exactly `len.div_ceil(8)` bytes
/// and must fill them all. The bits past `len` in the last byte are cleared
/// after. When `write` fails the cell is freed, and its error is the answer.
pub(crate) fn fill<E>(
    heap: &mut Heap,
    len: u64,
    write: impl FnOnce(&mut [u8]) -> Result<(), E>,
) -> Result<Result<Cell, E>, Full> {
    let n = usize::try_from(len.div_ceil(8)).map_err(|_| Full)?;
    let cell = heap.make_uninit(Kind::Binary, 1 + n.div_ceil(8))?;
    let data = heap.data_mut(cell);
    let [head, words @ ..] = data else {
        heap.release(cell);
        return Err(Full);
    };
    *head = len;
    // A reused cell holds what it last held: the bytes past `n` in the last
    // word must read as zeros, like any other bits past the end.
    if let Some(last) = words.last_mut() {
        *last = 0;
    }
    let wrote = if BYTES_IN_PLACE {
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(words);
        write(bytes.get_mut(..n).unwrap_or(&mut []))
    } else {
        let mut bytes = vec![0u8; n];
        let wrote = write(&mut bytes);
        for (w, chunk) in words.iter_mut().zip(bytes.chunks(8)) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            *w = u64::from_le_bytes(word);
        }
        wrote
    };
    if let Err(e) = wrote {
        heap.release(cell);
        return Ok(Err(e));
    }
    let pad = n as u64 * 8 - len;
    if pad > 0
        && let Some(w) = words.get_mut((n - 1) / 8)
    {
        let shift = (n - 1) % 8 * 8;
        let last = (*w >> shift) as u8 & (0xFFu8 << pad);
        *w = *w & !(0xFF << shift) | u64::from(last) << shift;
    }
    Ok(Ok(cell))
}

/// Bits `from .. from + len` of `b`, sharing its owner. The caller has
/// checked that they are inside it.
pub(crate) fn slice(heap: &mut Heap, b: Bits, from: u64, len: u64) -> Result<Cell, Full> {
    heap.share(b.owner);
    heap.make(
        Kind::BinarySlice,
        &[Value::cell(b.owner).bits(), b.at + from, len],
    )
}

/// Byte `i` of `b`'s owner, or 0 past its end.
fn owner_byte(heap: &Heap, owner: Cell, i: u64) -> u8 {
    let d = heap.data(owner);
    match d.get(1 + (i / 8) as usize) {
        Some(w) => (w >> ((i % 8) * 8)) as u8,
        None => 0,
    }
}

/// The 8 bits of `b` from bit `i`, the ones past its end read as 0.
fn byte(heap: &Heap, b: Bits, i: u64) -> u8 {
    if i >= b.len {
        return 0;
    }
    let p = b.at + i;
    let s = (p % 8) as u32;
    let hi = owner_byte(heap, b.owner, p / 8);
    let raw = if s == 0 {
        hi
    } else {
        let lo = owner_byte(heap, b.owner, p / 8 + 1);
        (hi << s) | (lo >> (8 - s))
    };
    let left = b.len - i;
    if left >= 8 {
        raw
    } else {
        raw & (0xFFu8 << (8 - left))
    }
}

/// Word `k` of `b`'s bytes, little-endian: bytes `8k` to `8k + 7`, the first
/// in the lowest bits. `None` when it runs past `b`'s end. When `b` starts on
/// a word of its owner, as a binary made whole does, this is that word as it
/// is, so a JSON tape (`json.rs`) reads a node without copying the tape.
pub(crate) fn word(heap: &Heap, b: Bits, k: u64) -> Option<u64> {
    let end = k.checked_add(1)?.checked_mul(64)?;
    if end > b.len {
        return None;
    }
    if b.at.is_multiple_of(64) {
        let at = usize::try_from(b.at / 64 + k).ok()?;
        return heap.data(b.owner).get(1 + at).copied();
    }
    let mut bytes = [0u8; 8];
    for (j, out) in bytes.iter_mut().enumerate() {
        *out = byte(heap, b, k * 64 + j as u64 * 8);
    }
    Some(u64::from_le_bytes(bytes))
}

/// Bytes `from` to `from + n` of `b`, or `None` when they run past its end.
pub(crate) fn byte_range(heap: &Heap, b: Bits, from: u64, n: u64) -> Option<Vec<u8>> {
    if from.checked_add(n)?.checked_mul(8)? > b.len {
        return None;
    }
    Some((from..from + n).map(|k| byte(heap, b, k * 8)).collect())
}

/// Up to `n` whole bytes of `b` from byte `from`, fewer where it ends first.
pub(crate) fn bytes_from(heap: &Heap, b: Bits, from: u64, n: u64) -> Vec<u8> {
    let whole = b.len / 8;
    let from = from.min(whole);
    let end = from.saturating_add(n).min(whole);
    if !b.at.is_multiple_of(8) {
        return (from..end).map(|k| byte(heap, b, k * 8)).collect();
    }
    // Starting on a byte of the owner, each byte is one of its bytes as it is.
    let d = heap.data(b.owner);
    let base = b.at / 8;
    (from..end)
        .map(|k| {
            let i = base + k;
            d.get(1 + (i / 8) as usize)
                .map_or(0, |w| (w >> (i % 8 * 8)) as u8)
        })
        .collect()
}

/// Whether bytes `from` to `from + text.len()` of `b` are `text`, read in
/// place.
pub(crate) fn bytes_are(heap: &Heap, b: Bits, from: u64, text: &[u8]) -> bool {
    let fits = (text.len() as u64)
        .checked_add(from)
        .and_then(|end| end.checked_mul(8))
        .is_some_and(|end| end <= b.len);
    fits && text
        .iter()
        .zip(from..)
        .all(|(&t, k)| byte(heap, b, k * 8) == t)
}

/// `b`'s bytes, read in place when they can be and copied when not. `b` must
/// be whole bytes; a last byte that is not whole is padded with zero bits, as
/// [`bytes`] does.
///
/// In place is a whole-byte binary starting on a byte of its owner, on a
/// little-endian host, where the owner's words hold its bytes in order in
/// memory. That is every binary a program builds whole, and every slice of
/// one at a byte, so handing one to the OS or a GPU copies it once, there.
pub(crate) fn byte_view(heap: &Heap, b: Bits) -> Cow<'_, [u8]> {
    match in_place(heap, b) {
        Some(bytes) => Cow::Borrowed(bytes),
        None => Cow::Owned(bytes(heap, b)),
    }
}

fn in_place(heap: &Heap, b: Bits) -> Option<&[u8]> {
    if !BYTES_IN_PLACE || !b.len.is_multiple_of(8) || !b.at.is_multiple_of(8) {
        return None;
    }
    let words = heap.data(b.owner).get(1..)?;
    let all: &[u8] = bytemuck::cast_slice(words);
    let from = usize::try_from(b.at / 8).ok()?;
    let n = usize::try_from(b.len / 8).ok()?;
    all.get(from..from.checked_add(n)?)
}

/// `b` as whole bytes, the last one padded with zero bits.
pub(crate) fn bytes(heap: &Heap, b: Bits) -> Vec<u8> {
    (0..b.len.div_ceil(8))
        .map(|k| byte(heap, b, k * 8))
        .collect()
}

/// `b`'s whole bytes: a last byte that is not whole is left out, as the
/// ASCII built-ins read it.
pub(crate) fn whole_bytes(heap: &Heap, b: Bits) -> Vec<u8> {
    (0..b.len / 8).map(|k| byte(heap, b, k * 8)).collect()
}

/// Whole byte `i` of `b`, or `None` past the last whole one.
pub(crate) fn whole_byte(heap: &Heap, b: Bits, i: u64) -> Option<u8> {
    (i < b.len / 8).then(|| byte(heap, b, i * 8))
}

/// Whether `a` and `b` hold the same bits.
pub(crate) fn equal(heap: &Heap, a: Bits, b: Bits) -> bool {
    a.len == b.len && (0..a.len.div_ceil(8)).all(|k| byte(heap, a, k * 8) == byte(heap, b, k * 8))
}

/// Whether `b` holds `prefix` from bit `at`.
pub(crate) fn has_at(heap: &Heap, b: Bits, at: u64, prefix: Bits) -> bool {
    let Some(end) = at.checked_add(prefix.len) else {
        return false;
    };
    if end > b.len {
        return false;
    }
    let window = Bits {
        owner: b.owner,
        at: b.at + at,
        len: prefix.len,
    };
    equal(heap, window, prefix)
}

/// All of `parts`, one after another, as a new owner: one cell, each of its
/// bytes written once. A part that lands on a byte of the result and starts
/// on a byte of its own owner is copied a run of whole bytes at a time,
/// straight from its owner's cell into the new one ([`Heap::copy_bytes`]).
/// Any other goes 8 bits at a time, shifted into place.
pub(crate) fn join(heap: &mut Heap, parts: &[Bits]) -> Result<Cell, Full> {
    let len = parts
        .iter()
        .try_fold(0u64, |total, p| total.checked_add(p.len))
        .filter(|&len| len <= MAX_BITS)
        .ok_or(Full)?;
    join_as(heap, len, parts.iter().copied())
}

/// [`join`] of `parts`, which are `len` bits between them.
fn join_as(heap: &mut Heap, len: u64, parts: impl Iterator<Item = Bits>) -> Result<Cell, Full> {
    let n = usize::try_from(len.div_ceil(8)).map_err(|_| Full)?;
    let cell = heap.make_uninit(Kind::Binary, 1 + n.div_ceil(8))?;
    if let [head, words @ ..] = heap.data_mut(cell) {
        *head = len;
        // A reused cell holds what it last held: the bytes past `n` in the
        // last word must read as zeros, like any other bits past the end.
        if let Some(last) = words.last_mut() {
            *last = 0;
        }
    }
    let mut out = Joined {
        cell,
        at: 0,
        carry: 0,
        run: Vec::new(),
    };
    for p in parts {
        let mut from = 0;
        if BYTES_IN_PLACE && out.at.is_multiple_of(8) && p.at.is_multiple_of(8) {
            out.flush(heap);
            let whole = p.len / 8;
            if copy_whole(heap, p, cell, out.at / 8, whole) {
                out.at += whole * 8;
                from = whole;
            }
        }
        for k in from..p.len.div_ceil(8) {
            let chunk = byte(heap, p, k * 8);
            out.push(heap, chunk, (p.len - k * 8).min(8));
        }
    }
    out.flush(heap);
    if !out.at.is_multiple_of(8) {
        let last = [out.carry];
        write(heap, cell, out.at / 8, &last);
    }
    Ok(cell)
}

/// Copy the first `n` bytes of `p`, which starts on a byte of its owner,
/// into the owner `cell` from its byte `at`. `false`, copying nothing, when
/// they do not fit.
fn copy_whole(heap: &mut Heap, p: Bits, cell: Cell, at: u64, n: u64) -> bool {
    // An owner's bytes follow the word holding its length.
    let (Ok(from), Ok(to), Ok(n)) = (
        usize::try_from(8 + p.at / 8),
        usize::try_from(8 + at),
        usize::try_from(n),
    ) else {
        return false;
    };
    heap.copy_bytes(p.owner, from, cell, to, n)
}

/// A binary [`join`] is writing, `at` bits in. `carry` holds the top `at % 8`
/// bits of the byte not yet whole, and `run` the whole bytes before it that
/// are not in the cell yet.
struct Joined {
    cell: Cell,
    at: u64,
    carry: u8,
    run: Vec<u8>,
}

impl Joined {
    /// The most bytes `run` holds before they go to the cell.
    const RUN: usize = 4096;

    /// Add the top `n` bits of `chunk`, the rest of which are zero.
    fn push(&mut self, heap: &mut Heap, chunk: u8, n: u64) {
        let s = (self.at % 8) as u32;
        let v = self.carry | (chunk >> s);
        if u64::from(s) + n >= 8 {
            self.run.push(v);
            // What did not fit, moved to the top: `chunk`'s low `s` bits.
            self.carry = chunk.checked_shl(8 - s).unwrap_or(0);
        } else {
            self.carry = v;
        }
        self.at += n;
        if self.run.len() >= Self::RUN {
            self.flush(heap);
        }
    }

    /// Put `run` in the cell, where it ends at the byte `carry` is part of.
    fn flush(&mut self, heap: &mut Heap) {
        if self.run.is_empty() {
            return;
        }
        let end = self.at / 8;
        write(heap, self.cell, end - self.run.len() as u64, &self.run);
        self.run.clear();
    }
}

/// Write `bytes` into the owner `cell` from its byte `at`. The caller has
/// made the cell long enough.
fn write(heap: &mut Heap, cell: Cell, at: u64, bytes: &[u8]) {
    let Some(words) = heap.data_mut(cell).get_mut(1..) else {
        return;
    };
    if BYTES_IN_PLACE {
        let all: &mut [u8] = bytemuck::cast_slice_mut(words);
        let into = usize::try_from(at)
            .ok()
            .and_then(|at| all.get_mut(at..at.checked_add(bytes.len())?));
        if let Some(into) = into {
            into.copy_from_slice(bytes);
        }
        return;
    }
    for (i, &b) in (at..).zip(bytes) {
        if let Some(w) = words.get_mut((i / 8) as usize) {
            let shift = i % 8 * 8;
            *w = *w & !(0xFF << shift) | u64::from(b) << shift;
        }
    }
}

/// `b`, `n` times over, as a new owner, or `Err(Full)` when that is longer
/// than a binary can be. A whole-byte `b` is copied into the cell `n` times;
/// any other is [`join`]ed, since each copy then starts at another bit.
pub(crate) fn repeat(heap: &mut Heap, b: Bits, n: u64) -> Result<Cell, Full> {
    let len = b
        .len
        .checked_mul(n)
        .filter(|&len| len <= MAX_BITS)
        .ok_or(Full)?;
    if len == 0 {
        return make(heap, &[], 0);
    }
    if !b.len.is_multiple_of(8) {
        let n = usize::try_from(n).map_err(|_| Full)?;
        return join_as(heap, len, std::iter::repeat_n(b, n));
    }
    let one = byte_view(heap, b).into_owned();
    fill(heap, len, |out| {
        for copy in out.chunks_exact_mut(one.len()) {
            copy.copy_from_slice(&one);
        }
        Ok::<(), Full>(())
    })?
}

/// A byte order, as `scarlet/binary.Endian` names one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Endian {
    Big,
    Little,
}

/// A Float as an f32. One past the largest f32 stops there, keeping its
/// sign, since `as f32` would make it an infinity; anything else rounds to
/// the nearest f32, subnormals and -0.0 included. A Float is never a NaN
/// (`docs/semantics.md`, "Floats"), so no f32 made here is one either.
pub(crate) fn narrow(x: f64) -> f32 {
    x.clamp(f64::from(f32::MIN), f64::from(f32::MAX)) as f32
}

/// `x`'s 4 bytes, in `endian`'s order.
pub(crate) fn f32_bytes(x: f32, endian: Endian) -> [u8; 4] {
    match endian {
        Endian::Big => x.to_be_bytes(),
        Endian::Little => x.to_le_bytes(),
    }
}

/// The f32 `bytes` spell in `endian`'s order, or `None` for an infinity or a
/// NaN, which no Float can hold.
pub(crate) fn read_f32(bytes: [u8; 4], endian: Endian) -> Option<f32> {
    let x = match endian {
        Endian::Big => f32::from_be_bytes(bytes),
        Endian::Little => f32::from_le_bytes(bytes),
    };
    x.is_finite().then_some(x)
}

/// The low `width` bits of `n`, most significant first. A negative `n` is
/// its two's complement, extended as far as `width` asks, so `<<-1:size(12)>>`
/// is twelve ones.
pub(crate) fn from_int(n: &BigInt, width: u64) -> Vec<u8> {
    let bytes = width.div_ceil(8) as usize;
    if bytes == 0 {
        return Vec::new();
    }
    let modulus = BigInt::from(1) << width;
    // `%` keeps the dividend's sign, so add the modulus back to land in
    // `0 .. 2^width`.
    let low = ((n % &modulus) + &modulus) % &modulus;
    let pad = bytes as u64 * 8 - width;
    let (_, be) = (low << pad).to_bytes_be();
    let mut out = vec![0u8; bytes];
    let start = bytes.saturating_sub(be.len());
    for (o, b) in out.iter_mut().skip(start).zip(be) {
        *o = b;
    }
    out
}

/// `width` bits of `b` from bit `at`, as an unsigned Int, most significant
/// bit first. Bits past `b`'s end read as 0.
pub(crate) fn read_uint(heap: &Heap, b: Bits, at: u64, width: u64) -> BigInt {
    let window = Bits {
        owner: b.owner,
        at: b.at + at.min(b.len),
        len: b.len.saturating_sub(at).min(width),
    };
    let n = BigInt::from_bytes_be(Sign::Plus, &bytes(heap, window));
    // `bytes` pads the window to whole bytes; drop the padding, then add the
    // zeros past `b`'s end that `width` still asks for.
    let n = n >> (window.len.div_ceil(8) * 8 - window.len);
    n << (width - window.len)
}

/// The UTF-8 code point at bit `at` of `b`, and how many bits it took, or
/// `None` when no valid one starts there.
pub(crate) fn read_utf8(heap: &Heap, b: Bits, at: u64) -> Option<(u32, u64)> {
    if at.checked_add(8)? > b.len {
        return None;
    }
    let b0 = byte(heap, b, at);
    let n: u64 = if b0 < 0x80 {
        1
    } else if b0 >> 5 == 0b110 {
        2
    } else if b0 >> 4 == 0b1110 {
        3
    } else if b0 >> 3 == 0b11110 {
        4
    } else {
        return None;
    };
    if at + n * 8 > b.len {
        return None;
    }
    let buf: Vec<u8> = (0..n).map(|k| byte(heap, b, at + k * 8)).collect();
    let c = std::str::from_utf8(&buf).ok()?.chars().next()?;
    Some((c as u32, n * 8))
}

/// How a binary shows: `<<1, 2, 3>>`, with a last byte that is not whole as
/// `5:size(3)`.
pub(crate) fn text(heap: &Heap, b: Bits) -> String {
    let mut parts: Vec<String> = (0..b.len / 8)
        .map(|k| byte(heap, b, k * 8).to_string())
        .collect();
    let rem = b.len % 8;
    if rem != 0 {
        let last = byte(heap, b, b.len - rem) >> (8 - rem);
        parts.push(format!("{last}:size({rem})"));
    }
    format!("<<{}>>", parts.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::View;

    fn owner(heap: &mut Heap, bytes: &[u8], len: u64) -> Bits {
        let cell = make(heap, bytes, len).expect("room");
        bits(heap, cell).expect("a binary")
    }

    #[test]
    fn a_binary_shows_its_bytes_and_a_partial_last_one() {
        let mut heap = Heap::default();
        let b = owner(&mut heap, &[1, 2, 3], 24);
        assert_eq!(text(&heap, b), "<<1, 2, 3>>");
        let b = owner(&mut heap, &[0xFF, 0b1011_1111], 11);
        assert_eq!(text(&heap, b), "<<255, 5:size(3)>>");
        let b = owner(&mut heap, &[], 0);
        assert_eq!(text(&heap, b), "<<>>");
    }

    #[test]
    fn from_int_writes_the_low_bits_most_significant_first() {
        assert_eq!(from_int(&1.into(), 4), vec![0b0001_0000]);
        assert_eq!(from_int(&65.into(), 8), vec![65]);
        assert_eq!(from_int(&0x1234.into(), 16), vec![0x12, 0x34]);
        assert_eq!(from_int(&0x1FF.into(), 8), vec![0xFF]);
        assert_eq!(from_int(&(-1).into(), 12), vec![0xFF, 0xF0]);
        assert_eq!(from_int(&7.into(), 0), Vec::<u8>::new());
        let big: BigInt = BigInt::from(1) << 100;
        let wide = from_int(&big, 104);
        assert_eq!(wide.len(), 13);
        assert_eq!(wide[0], 0x10);
    }

    /// A slice of a slice reads the right bits of the owner, at any offset.
    #[test]
    fn slices_read_through_to_the_owner() {
        let mut heap = Heap::default();
        let b = owner(&mut heap, &[0b1010_1100, 0b0011_0101, 0xFF], 24);
        let s1 = slice(&mut heap, b, 3, 17).expect("room");
        let s1 = bits(&heap, s1).expect("a slice");
        let s2 = slice(&mut heap, s1, 2, 9).expect("room");
        let s2 = bits(&heap, s2).expect("a slice");
        // Bits 5..14 of the owner: 1 0 0 0 0 1 1 0 1.
        assert_eq!(bytes(&heap, s2), vec![0b1000_0110, 0b1000_0000]);
        assert_eq!(read_uint(&heap, s2, 0, 9), BigInt::from(0b1_0000_1101));
        assert_eq!(read_uint(&heap, s2, 5, 8), BigInt::from(0b1101_0000));
        let want = owner(&mut heap, &[0b1000_0110, 0b1000_0000], 9);
        assert!(equal(&heap, s2, want));
    }

    #[test]
    fn join_packs_at_any_bit() {
        let mut heap = Heap::default();
        let a = owner(&mut heap, &[0b1010_0000], 3);
        let b = owner(&mut heap, &[0xFF, 0b1000_0000], 9);
        let j = join(&mut heap, &[a, b, a]).expect("room");
        let j = bits(&heap, j).expect("a binary");
        assert_eq!(j.len, 15);
        // 101, then nine 1s, then 101.
        assert_eq!(bytes(&heap, j), vec![0b1011_1111, 0b1111_1010]);
    }

    /// A small, seeded generator, so a failure names the case that made it.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            // xorshift64*
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// `join` as it was before its fast path: every part 8 bits at a time,
    /// each bit set on its own.
    fn join_bit_by_bit(heap: &Heap, parts: &[Bits]) -> (Vec<u8>, u64) {
        let len: u64 = parts.iter().map(|p| p.len).sum();
        let mut out = vec![0u8; len.div_ceil(8) as usize];
        let mut at = 0u64;
        for p in parts {
            for k in 0..p.len.div_ceil(8) {
                let chunk = byte(heap, *p, k * 8);
                for j in 0..(p.len - k * 8).min(8) {
                    if chunk & (0x80 >> j) != 0 {
                        let q = at + k * 8 + j;
                        out[(q / 8) as usize] |= 0x80 >> (q % 8);
                    }
                }
            }
            at += p.len;
        }
        (out, len)
    }

    /// Random parts of every shape a join meets: owners whole or ending
    /// mid-byte, empty ones, and slices starting on a byte and off one. The
    /// result is the same bits the bit-by-bit join gives, in a cell that may
    /// have been another binary's.
    #[test]
    fn join_gives_the_bits_the_bit_by_bit_join_did() {
        let mut rng = Rng(0x5CA7_1E7B_1A71_0001);
        let mut heap = Heap::default();
        for round in 0..2_000 {
            let mut parts = Vec::new();
            let mut held = Vec::new();
            for _ in 0..rng.below(12) {
                let len = match rng.below(4) {
                    0 => rng.below(8),
                    1 => rng.below(40) * 8,
                    _ => rng.below(300),
                };
                let bytes: Vec<u8> = (0..len.div_ceil(8)).map(|_| rng.next() as u8).collect();
                let cell = make(&mut heap, &bytes, len).expect("room");
                held.push(cell);
                let b = bits(&heap, cell).expect("a binary");
                let p = if len > 0 && rng.below(2) == 0 {
                    let from = rng.below(len);
                    let take = rng.below(len - from + 1);
                    let cut = slice(&mut heap, b, from, take).expect("room");
                    held.push(cut);
                    bits(&heap, cut).expect("a slice")
                } else {
                    b
                };
                parts.push(p);
            }
            let (want, len) = join_bit_by_bit(&heap, &parts);
            let joined = join(&mut heap, &parts).expect("room");
            let got = bits(&heap, joined).expect("a binary");
            assert_eq!(got.len, len, "round {round}");
            assert_eq!(bytes(&heap, got), want, "round {round}");
            // Every bit past the end reads as zero, in every word the cell has.
            let words = &heap.data(joined)[1..];
            let all: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            assert!(all[want.len()..].iter().all(|&b| b == 0), "round {round}");
            // Dirty the cells a later round's join will reuse.
            held.push(joined);
            for cell in held {
                heap.release(cell);
            }
        }
        assert_eq!(heap.live(), 0);
    }

    /// Parts that each land on a byte and start on one are copied cell to
    /// cell, every byte once; a part off a byte is not, and the parts after
    /// it no longer land on a byte.
    #[test]
    fn a_join_on_whole_bytes_copies_each_byte_once() {
        let mut heap = Heap::default();
        let data: Vec<u8> = (0..=255).collect();
        let part = owner(&mut heap, &data, 256 * 8);
        let parts = vec![part; 1_000];
        let before = heap.copied();
        let joined = join(&mut heap, &parts).expect("room");
        let joined = bits(&heap, joined).expect("a binary");
        let expected = if BYTES_IN_PLACE { 256_000 } else { 0 };
        assert_eq!(heap.copied() - before, expected);
        assert_eq!(bytes(&heap, joined), data.repeat(1_000));

        let odd = owner(&mut heap, &[0b1010_0000], 3);
        let before = heap.copied();
        let joined = join(&mut heap, &[part, odd, part]).expect("room");
        let joined = bits(&heap, joined).expect("a binary");
        let expected = if BYTES_IN_PLACE { 256 } else { 0 };
        assert_eq!(heap.copied() - before, expected);
        assert_eq!(joined.len, 256 * 8 * 2 + 3);
    }

    /// A part off a byte longer than the bytes [`Joined`] holds before
    /// writing them goes to the cell in several runs, each where it belongs.
    #[test]
    fn a_long_join_off_a_byte_writes_every_run_in_place() {
        let mut heap = Heap::default();
        let data: Vec<u8> = (0..10_000u32).map(|i| (i * 7 + i / 256) as u8).collect();
        let long = owner(&mut heap, &data, 10_000 * 8);
        let odd = owner(&mut heap, &[0b1010_0000], 3);
        let off = slice(&mut heap, long, 5, 9_000 * 8).expect("room");
        let off = bits(&heap, off).expect("a slice");
        let parts = [odd, long, odd, off, long];
        let (want, len) = join_bit_by_bit(&heap, &parts);
        let joined = join(&mut heap, &parts).expect("room");
        let got = bits(&heap, joined).expect("a binary");
        assert_eq!(got.len, len);
        assert_eq!(bytes(&heap, got), want);
    }

    #[test]
    fn repeat_gives_n_copies_of_bytes_or_bits() {
        let mut heap = Heap::default();
        let b = owner(&mut heap, &[1, 2], 16);
        for (n, want) in [(0, "<<>>"), (1, "<<1, 2>>"), (3, "<<1, 2, 1, 2, 1, 2>>")] {
            let r = repeat(&mut heap, b, n).expect("room");
            assert_eq!(text(&heap, bits(&heap, r).expect("a binary")), want);
        }
        // 101 three times: 1011 0110 1.
        let b = owner(&mut heap, &[0b1010_0000], 3);
        let r = repeat(&mut heap, b, 3).expect("room");
        assert_eq!(
            text(&heap, bits(&heap, r).expect("a binary")),
            "<<182, 1:size(1)>>"
        );
        let empty = owner(&mut heap, &[], 0);
        let r = repeat(&mut heap, empty, u64::MAX).expect("room");
        assert_eq!(bits(&heap, r).expect("a binary").len, 0);
        assert_eq!(repeat(&mut heap, b, u64::MAX), Err(Full));
        assert_eq!(repeat(&mut heap, b, MAX_BITS / 3 + 1), Err(Full));
    }

    /// Whatever 64 bits a Float is made from, its f32 is finite. A Float is
    /// made only through `Value::float`, which never holds a NaN; the bits
    /// that are not a NaN, infinities among them, are checked as they are.
    #[test]
    fn no_packed_f32_is_infinite_or_nan() {
        let mut rng = Rng(0xF10A_7320_0000_0002);
        let edges = [
            f64::MAX,
            f64::MIN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from(f32::MAX),
            f64::from(f32::MIN),
            // Half an ulp past f32::MAX: `as f32` rounds this to infinity.
            f64::from(f32::MAX) + 2f64.powi(103),
            -(f64::from(f32::MAX) + 2f64.powi(103)),
            f64::MIN_POSITIVE,
            -0.0,
        ];
        let random = (0..1_000_000).map(|_| f64::from_bits(rng.next()));
        for x in edges.into_iter().chain(random) {
            let View::Float(float) = Value::float(x).view() else {
                panic!("{x} is not a Float");
            };
            assert!(narrow(float).is_finite(), "{x:e} as a Float");
            if !x.is_nan() {
                assert!(narrow(x).is_finite(), "{x:e}");
            }
        }
        assert_eq!(narrow(f64::MAX), f32::MAX);
        assert_eq!(narrow(f64::MIN), f32::MIN);
        assert_eq!(narrow(f64::from(f32::MAX) + 2f64.powi(103)), f32::MAX);
    }

    /// Every class of finite f32 — zeros of both signs, subnormals, normals
    /// up to the largest — is the same f32 after going to a Float and back,
    /// and after being written and read in either byte order.
    #[test]
    fn every_finite_f32_packs_and_unpacks_as_itself() {
        let mut rng = Rng(0xF10A_7320_0000_0003);
        let classes = [
            0.0,
            f32::from_bits(1),
            f32::from_bits(0x007F_FFFF),
            f32::MIN_POSITIVE,
            1.0,
            f32::MAX,
        ];
        let signed = classes.into_iter().flat_map(|x| [x, -x]);
        let random = (0..1_000_000)
            .map(|_| f32::from_bits(rng.next() as u32))
            .filter(|x| x.is_finite());
        for x in signed.chain(random) {
            assert_eq!(narrow(f64::from(x)).to_bits(), x.to_bits(), "{x:e}");
            for endian in [Endian::Big, Endian::Little] {
                let back = read_f32(f32_bytes(x, endian), endian).expect("finite");
                assert_eq!(back.to_bits(), x.to_bits(), "{x:e} {endian:?}");
            }
        }
        assert_eq!(f32_bytes(1.0, Endian::Little), [0, 0, 128, 63]);
        assert_eq!(f32_bytes(1.0, Endian::Big), [63, 128, 0, 0]);
    }

    /// Four bytes that spell an infinity or a NaN, of any sign or payload,
    /// are no f32 a Float can hold.
    #[test]
    fn an_infinity_or_a_nan_reads_as_none() {
        let not_finite = [
            0x7F80_0000u32,
            0xFF80_0000,
            0x7FC0_0000,
            0xFFC0_0000,
            0x7F80_0001,
            0x7FBF_FFFF,
            0x7FFF_FFFF,
            0xFFFF_FFFF,
        ];
        for bits in not_finite {
            for endian in [Endian::Big, Endian::Little] {
                let bytes = match endian {
                    Endian::Big => bits.to_be_bytes(),
                    Endian::Little => bits.to_le_bytes(),
                };
                assert_eq!(read_f32(bytes, endian), None, "{bits:#x} {endian:?}");
            }
        }
    }

    #[test]
    fn utf8_reads_one_code_point_or_none() {
        let mut heap = Heap::default();
        let b = owner(&mut heap, "é!".as_bytes(), 24);
        assert_eq!(read_utf8(&heap, b, 0), Some(('é' as u32, 16)));
        assert_eq!(read_utf8(&heap, b, 16), Some(('!' as u32, 8)));
        assert_eq!(read_utf8(&heap, b, 8), None);
        assert_eq!(read_utf8(&heap, b, 24), None);
    }

    /// A binary made whole, and a slice of one at a byte, are read in place
    /// on a little-endian host: what goes to a file or a GPU is not copied on
    /// the way. Anything else is a copy of the same bytes.
    #[test]
    fn whole_bytes_on_a_byte_are_read_in_place() {
        let mut heap = Heap::default();
        let data: Vec<u8> = (0..20).collect();
        let b = owner(&mut heap, &data, 160);
        let at_a_byte = slice(&mut heap, b, 24, 80).expect("room");
        let at_a_byte = bits(&heap, at_a_byte).expect("a slice");
        let at_a_bit = slice(&mut heap, b, 3, 80).expect("room");
        let at_a_bit = bits(&heap, at_a_bit).expect("a slice");
        let part = owner(&mut heap, &data, 157);
        let little = cfg!(target_endian = "little");
        for (b, in_place) in [
            (b, little),
            (at_a_byte, little),
            (at_a_bit, false),
            (part, false),
        ] {
            let view = byte_view(&heap, b);
            assert_eq!(
                matches!(view, Cow::Borrowed(_)),
                in_place,
                "{}",
                text(&heap, b)
            );
            assert_eq!(*view, bytes(&heap, b)[..], "{}", text(&heap, b));
        }
        assert_eq!(*byte_view(&heap, at_a_byte), data[3..13]);
    }

    /// A cell a binary reuses may hold another's bytes. A new one reads as
    /// its own bytes and zeros after, however long, in every word.
    #[test]
    fn a_binary_in_a_reused_cell_holds_only_its_own_bytes() {
        let mut heap = Heap::default();
        for n in 1..=24 {
            let dirty = make(&mut heap, &[0xFF; 24], 24 * 8).expect("room");
            let words = heap.data(dirty).len();
            heap.release(dirty);
            let data: Vec<u8> = (1..=n).collect();
            let cell = make(&mut heap, &data, u64::from(n) * 8).expect("room");
            let b = bits(&heap, cell).expect("a binary");
            assert_eq!(bytes(&heap, b), data);
            let d = heap.data(cell);
            let want: Vec<u8> = data
                .iter()
                .copied()
                .chain(std::iter::repeat(0))
                .take((d.len() - 1) * 8)
                .collect();
            let got: Vec<u8> = d[1..].iter().flat_map(|w| w.to_le_bytes()).collect();
            assert_eq!(
                got,
                want,
                "{n} bytes, in a cell of {} words once {words}",
                d.len()
            );
            heap.release(cell);
        }
        assert_eq!(heap.live(), 0);
    }

    /// A fill that fails frees the cell it was writing, and gives the error.
    #[test]
    fn a_failed_fill_frees_its_cell() {
        let mut heap = Heap::default();
        let made = fill(&mut heap, 64, |out| {
            out.fill(7);
            Err::<(), &str>("refused")
        });
        assert_eq!(made.expect("room").err(), Some("refused"));
        assert_eq!(heap.live(), 0);
        let made = fill(&mut heap, 12, |out| {
            assert_eq!(out.len(), 2);
            out.copy_from_slice(&[0xAB, 0xFF]);
            Ok::<(), &str>(())
        });
        let cell = made.expect("room").expect("filled");
        let b = bits(&heap, cell).expect("a binary");
        assert_eq!(text(&heap, b), "<<171, 15:size(4)>>");
    }
}
