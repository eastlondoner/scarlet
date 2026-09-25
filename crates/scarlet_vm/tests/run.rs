//! Programs run end to end: compiled from Scarlet source, then run by the VM,
//! checking exactly what they print.

use scarlet_vm::{Host, Stop};

fn compile(src: &str) -> scarlet_core::core_ir::Program {
    let mut scanner = scarlet_core::scanner::new_scanner(src.to_string());
    let parsed = scarlet_core::parser::new_parser(&mut scanner).parse_program();
    let expr = scarlet_core::ast::Expression::BlockExpression(parsed.ast);
    let result = scarlet_core::bytecode::compile(&expr, None);
    assert!(result.success(), "{:?}", result.diagnostics);
    result.into_runnable().expect("a clean compile is runnable")
}

fn run(src: &str) -> Result<String, Stop> {
    let mut out = Vec::new();
    scarlet_vm::run(&compile(src), &Host::new(Vec::new(), Vec::new()), &mut out)?;
    Ok(String::from_utf8(out).expect("println writes UTF-8"))
}

fn prints(src: &str, want: &str) {
    assert_eq!(run(src).as_deref(), Ok(want), "{src}");
}

/// Run `src` in `host`'s world.
fn prints_in(host: &Host, src: &str, want: &str) {
    let mut out = Vec::new();
    let ran = scarlet_vm::run(&compile(src), host, &mut out);
    assert_eq!(ran, Ok(()), "{src}");
    assert_eq!(String::from_utf8(out).expect("UTF-8"), want, "{src}");
}

/// A fresh directory for one test's files.
fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("scarlet-vm-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

#[test]
fn one_plus_two() {
    prints("pub fn main() {\n\tprintln(1 + 2)\n}\n", "3\n");
}

#[test]
fn a_call_passes_its_arguments_and_returns_its_result() {
    prints(
        "fn add(a Int, b Int) Int { a + b }\n\
         fn twice(x Int) Int { add(x, x) }\n\
         pub fn main() {\n\
         \tprintln(twice(21))\n\
         \tprintln(add(1, add(2, 3)))\n\
         }\n",
        "42\n6\n",
    );
}

/// `docs/semantics.md`'s rules for `/` and `%`.
#[test]
fn division_follows_the_rules() {
    prints(
        "pub fn main() {\n\
         \tprintln(-7 / 2)\n\
         \tprintln(7 / 0)\n\
         \tprintln(-7 % 2)\n\
         \tprintln(7 % 0)\n\
         }\n",
        "-3\n0\n-1\n7\n",
    );
}

#[test]
fn comparisons_give_bools() {
    prints(
        "pub fn main() {\n\
         \tprintln(3 < 4)\n\
         \tprintln(3 == 4)\n\
         \tprintln(4 >= 4)\n\
         }\n",
        "True\nFalse\nTrue\n",
    );
}

#[test]
fn nil_prints_as_nil() {
    prints("pub fn main() {\n\tprintln(println(1))\n}\n", "1\nNil\n");
}

#[test]
fn if_picks_a_branch_in_both_positions() {
    prints(
        "fn max(a Int, b Int) Int {\n\
         \tif a > b { a } else { b }\n\
         }\n\
         pub fn main() {\n\
         \tx = if 1 < 2 { 10 } else { 20 }\n\
         \tprintln(x + max(3, 4))\n\
         \tprintln(if max(1, 2) == 2 { 7 } else { 8 })\n\
         }\n",
        "14\n7\n",
    );
}

/// A loop written as tail recursion: each step's call is a tail call, which
/// reuses the frame, so a million steps finish in the room of one.
#[test]
fn a_tail_recursive_loop_of_a_million_steps_finishes() {
    prints(
        "fn count(n Int, acc Int) Int {\n\
         \tif n == 0 { acc } else { count(n - 1, acc + 1) }\n\
         }\n\
         pub fn main() {\n\
         \tprintln(count(1000000, 0))\n\
         }\n",
        "1000000\n",
    );
}

/// Perceus reuse: a constructor whose cell nothing else holds any more is
/// overwritten in place rather than allocated. Here the loop gives its pair up
/// at the `match` and builds another on the way round, so one cell serves
/// every turn. `scarlet/internal` counts both, so the test pins the saving
/// rather than just the answer.
#[test]
fn a_loop_overwrites_the_cell_it_just_gave_up() {
    prints(
        "import scarlet/internal\n\
         type Pair {\n\
         \tPair(a Int, b Int)\n\
         }\n\
         fn spin(n Int, p Pair) Int {\n\
         \tmatch p {\n\
         \t\tPair(a, b) -> if n <= 0 then a + b else spin(n - 1, Pair(b, a + 1))\n\
         \t}\n\
         }\n\
         pub fn main() {\n\
         \tmade = internal.cells_made()\n\
         \treused = internal.cells_reused()\n\
         \tprintln(spin(1000, Pair(0, 0)))\n\
         \tprintln(internal.cells_made() - made)\n\
         \tprintln(internal.cells_reused() - reused)\n\
         }\n",
        "1000\n1\n1000\n",
    );
}

/// Reuse across a call, which needs the call to be a move: the caller hands
/// over its own reference, so the callee is the only holder and may overwrite
/// the cell. Every `Cons` of the new list is the matching one of the old,
/// written over, so the walk allocates nothing at all.
#[test]
fn a_walk_over_a_list_overwrites_the_list_it_reads() {
    prints(
        "import scarlet/internal\n\
         type List {\n\
         \tCons(head Int, tail List)\n\
         \tEmpty\n\
         }\n\
         fn build(n Int, acc List) List {\n\
         \tif n <= 0 then acc else build(n - 1, Cons(n, acc))\n\
         }\n\
         fn add_one(xs List) List {\n\
         \tmatch xs {\n\
         \t\tCons(h, t) -> Cons(h + 1, add_one(t))\n\
         \t\tEmpty -> Empty\n\
         \t}\n\
         }\n\
         fn sum(xs List, acc Int) Int {\n\
         \tmatch xs {\n\
         \t\tCons(h, t) -> sum(t, acc + h)\n\
         \t\tEmpty -> acc\n\
         \t}\n\
         }\n\
         pub fn main() {\n\
         \txs = build(1000, Empty)\n\
         \tmade = internal.cells_made()\n\
         \treused = internal.cells_reused()\n\
         \tprintln(sum(add_one(xs), 0))\n\
         \tprintln(internal.cells_made() - made)\n\
         \tprintln(internal.cells_reused() - reused)\n\
         }\n",
        "501500\n0\n1000\n",
    );
}

/// The cell is given up *after* the constructor here, so the only turn that
/// can take it is the next one. That needs the cell to survive the call to
/// self in tail position, which reuses the frame.
#[test]
fn a_loop_carries_its_cell_across_the_call_that_ends_it() {
    prints(
        "import scarlet/internal\n\
         type Pair {\n\
         \tPair(a Int, b Int)\n\
         }\n\
         fn go(n Int, acc Int) Int {\n\
         \tp = Pair(n, n + 1)\n\
         \ts = match p {\n\
         \t\tPair(a, b) -> a + b\n\
         \t}\n\
         \tif n <= 0 then acc + s else go(n - 1, acc + s)\n\
         }\n\
         pub fn main() {\n\
         \tmade = internal.cells_made()\n\
         \treused = internal.cells_reused()\n\
         \tprintln(go(1000, 0))\n\
         \tprintln(internal.cells_made() - made)\n\
         \tprintln(internal.cells_reused() - reused)\n\
         }\n",
        "1002001\n1\n1000\n",
    );
}

/// Reuse waits on the last reference. The same loop with the pair kept in a
/// list allocates every turn, and every pair still reads back as it was
/// written: overwriting a cell something else holds would show up here.
#[test]
fn a_cell_something_else_holds_is_not_overwritten() {
    prints(
        "import scarlet/array\n\
         import scarlet/internal\n\
         type Pair {\n\
         \tPair(a Int, b Int)\n\
         }\n\
         fn keep(n Int, ps Array(Pair)) Array(Pair) {\n\
         \tif n <= 0 then ps else keep(n - 1, array.concat(ps, [Pair(n, n)]))\n\
         }\n\
         pub fn main() {\n\
         \tmade = internal.cells_made()\n\
         \tps = keep(100, [])\n\
         \tprintln(array.fold(ps, 0, fn(t, p) {\n\
         \t\tmatch p {\n\
         \t\t\tPair(a, b) -> t + a + b\n\
         \t\t}\n\
         \t}))\n\
         \tprintln(internal.cells_made() - made > 100)\n\
         }\n",
        "10100\nTrue\n",
    );
}

/// Frames live on the VM's own stack, not Rust's, so a deep recursion that is
/// not a tail call just uses memory: nothing overflows.
#[test]
fn a_deep_recursion_uses_memory_not_the_rust_stack() {
    prints(
        "fn depth(n Int) Int {\n\
         \tif n == 0 { 0 } else { 1 + depth(n - 1) }\n\
         }\n\
         pub fn main() {\n\
         \tprintln(depth(200000))\n\
         }\n",
        "200000\n",
    );
}

/// A function that uses something not built yet still loads; only calling
/// it stops the run, and the stop says what it needs.
#[test]
fn only_calling_an_unbuilt_function_stops_the_run() {
    let src = "import scarlet/process\n\
               fn me() process.Pid { process.self() }\n\
               pub fn main() {\n\
               \tprintln(1)\n\
               }\n";
    prints(src, "1\n");
    let calls = "import scarlet/process\n\
                 fn me() process.Pid { process.self() }\n\
                 pub fn main() {\n\
                 \t_ = me()\n\
                 }\n";
    assert_eq!(
        run(calls),
        Err(Stop::NotBuiltYet("the built-in ProcessSelf".into()))
    );
}

#[test]
fn strings_print_join_and_interpolate() {
    prints(
        "fn greet(name String) String { 'hello, ' + name }\n\
         pub fn main() {\n\
         \tprintln('hello, world')\n\
         \tprintln(greet('Scarlet'))\n\
         \tprintln('2 + 2 = ${2 + 2}, and 3 < 4 is ${3 < 4}')\n\
         }\n",
        "hello, world\nhello, Scarlet\n2 + 2 = 4, and 3 < 4 is True\n",
    );
}

/// An Int is exact however big it gets: past 48 bits it moves to the heap,
/// and past 64 it keeps going.
#[test]
fn ints_are_exact_past_48_and_64_bits() {
    prints(
        "fn fact(n Int) Int { if n == 0 { 1 } else { n * fact(n - 1) } }\n\
         pub fn main() {\n\
         \tprintln(140737488355327 + 1)\n\
         \tprintln(9223372036854775807 + 1)\n\
         \tprintln(fact(25))\n\
         \tprintln(0 - 9223372036854775807 - 1)\n\
         }\n",
        "140737488355328\n9223372036854775808\n15511210043330985984000000\n-9223372036854775808\n",
    );
}

/// A result that fits comes back small, so a number has one form and `==`
/// cannot see two.
#[test]
fn a_big_result_that_fits_is_small_again() {
    prints(
        "pub fn main() {\n\
         \tbig = 9223372036854775807 * 4\n\
         \tprintln(big / 4 == 9223372036854775807)\n\
         \tprintln(big - big == 0)\n\
         \tprintln({big - big} + 1)\n\
         \tprintln(big > 1)\n\
         \tprintln(0 - big < 0)\n\
         }\n",
        "True\nTrue\n1\nTrue\nTrue\n",
    );
}

/// `docs/semantics.md`'s rules hold for big ints too, including the zero
/// cases the library would otherwise panic on. The number is built by
/// multiplying: a literal past 64 bits is still a compile error, because the
/// compiler keeps Int constants as `i64`.
#[test]
fn big_division_follows_the_rules() {
    prints(
        "pub fn main() {\n\
         \tbig = 1000000000000000 * 1000000000000000\n\
         \tprintln({0 - big} / 7)\n\
         \tprintln({0 - big} % 7)\n\
         \tprintln(big / 0)\n\
         \tprintln(big % 0 == big)\n\
         }\n",
        "-142857142857142857142857142857\n-1\n0\nTrue\n",
    );
}

/// Output that goes nowhere, like a pipe into `head` that has closed.
struct Closed;

impl std::io::Write for Closed {
    fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
        Err(std::io::ErrorKind::BrokenPipe.into())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn a_closed_output_stops_the_run_quietly() {
    let program = compile("pub fn main() {\n\tprintln(1)\n}\n");
    assert_eq!(
        scarlet_vm::run(&program, &Host::new(Vec::new(), Vec::new()), &mut Closed),
        Err(Stop::OutputClosed)
    );
}

#[test]
fn a_constructor_prints_as_it_is_written() {
    prints(
        "type Shape {\n\
         \tCircle(radius Int)\n\
         \tRect(w Int, h Int)\n\
         \tDot\n\
         }\n\
         pub fn main() {\n\
         \tprintln(Circle(radius: 2))\n\
         \tprintln(Rect(w: 3, h: 4))\n\
         \tprintln(Dot)\n\
         \tprintln(Some(1))\n\
         \tprintln(Some(None))\n\
         \tprintln(Err(Nil))\n\
         \tprintln(Ok('fine'))\n\
         \tprintln('${Some(True)} and ${None}')\n\
         }\n",
        "Circle(2)\nRect(3, 4)\nDot\nSome(1)\nSome(None)\nErr(Nil)\nOk(fine)\nSome(True) and None\n",
    );
}

/// A constructor named after its own type is a record, and shows its labels.
#[test]
fn a_record_shows_its_labels() {
    prints(
        "type Point {\n\
         \tPoint(x Int, y Int)\n\
         }\n\
         pub fn main() {\n\
         \tprintln(Point(x: 1, y: -2))\n\
         }\n",
        "Point{ x: 1, y: -2 }\n",
    );
}

/// A constructor holding anything but small values takes a line per field,
/// and what it holds is laid out the same way, one level in.
#[test]
fn a_nested_constructor_takes_a_line_per_field() {
    prints(
        "type Point {\n\
         \tPoint(x Int, y Int)\n\
         }\n\
         type Seg {\n\
         \tSeg(a Point, b Point)\n\
         }\n\
         pub fn main() {\n\
         \tprintln(Seg(a: Point(x: 1, y: 2), b: Point(x: 3, y: 4)))\n\
         \tprintln(Some(Some(1)))\n\
         \tprintln(Ok('a string of twenty or more'))\n\
         }\n",
        "Seg {\n  a: Point{ x: 1, y: 2 },\n  b: Point{ x: 3, y: 4 }\n}\n\
         Some(\n  Some(1)\n)\n\
         Ok(\n  a string of twenty or more\n)\n",
    );
}

/// A list 100,000 long is built by a loop and freed when `main` ends, and
/// neither takes a Rust call per link.
#[test]
fn a_long_list_is_built_and_freed() {
    prints(
        "type L {\n\
         \tCons(h Int, t L)\n\
         \tEnd\n\
         }\n\
         fn build(n Int, acc L) L {\n\
         \tif n == 0 { acc } else { build(n - 1, Cons(n, acc)) }\n\
         }\n\
         pub fn main() {\n\
         \t_l = build(100000, End)\n\
         \tprintln(Cons(0, End))\n\
         }\n",
        "Cons(0, End)\n",
    );
}

#[test]
fn a_match_picks_the_arm_that_fits() {
    prints(
        "type Shape {\n\
         \tCircle(radius Int)\n\
         \tRect(w Int, h Int)\n\
         \tDot\n\
         }\n\
         fn area(s Shape) Int {\n\
         \tmatch s {\n\
         \t\tCircle(r) -> 3 * r * r\n\
         \t\tRect(w, h) -> w * h\n\
         \t\tDot -> 0\n\
         \t}\n\
         }\n\
         pub fn main() {\n\
         \tprintln(area(Circle(radius: 2)))\n\
         \tprintln(area(Rect(w: 3, h: 4)))\n\
         \tprintln(area(Dot))\n\
         }\n",
        "12\n12\n0\n",
    );
}

/// A nested pattern becomes a match inside a match, and a case that fails
/// part-way jumps on to the next one.
#[test]
fn a_nested_pattern_falls_through_to_the_next_case() {
    prints(
        "fn describe(o Option(Option(Int))) String {\n\
         \tmatch o {\n\
         \t\tSome(Some(0)) -> 'zero'\n\
         \t\tSome(Some(n)) -> 'some ${n}'\n\
         \t\tSome(None) -> 'some none'\n\
         \t\tNone -> 'none'\n\
         \t}\n\
         }\n\
         pub fn main() {\n\
         \tprintln(describe(Some(Some(0))))\n\
         \tprintln(describe(Some(Some(5))))\n\
         \tprintln(describe(Some(None)))\n\
         \tprintln(describe(None))\n\
         }\n",
        "zero\nsome 5\nsome none\nnone\n",
    );
}

#[test]
fn a_match_on_ints_and_strings() {
    prints(
        "fn count(n Int) String {\n\
         \tmatch n {\n\
         \t\t0 -> 'none'\n\
         \t\t1 | 2 -> 'a few'\n\
         \t\t_ -> 'many'\n\
         \t}\n\
         }\n\
         fn greet(s String) String {\n\
         \tmatch s {\n\
         \t\t'hi' -> 'hello'\n\
         \t\t'' -> 'nothing'\n\
         \t\tother -> 'what is ${other}?'\n\
         \t}\n\
         }\n\
         pub fn main() {\n\
         \tprintln(count(0))\n\
         \tprintln(count(2))\n\
         \tprintln(count(9))\n\
         \tprintln(count(9223372036854775807))\n\
         \tprintln(greet('hi'))\n\
         \tprintln(greet(''))\n\
         \tprintln(greet('hiya'))\n\
         }\n",
        "none\na few\nmany\nmany\nhello\nnothing\nwhat is hiya?\n",
    );
}

#[test]
fn a_record_field_reads_by_name() {
    prints(
        "type Point {\n\
         \tPoint(x Int, y Int)\n\
         }\n\
         pub fn main() {\n\
         \tp = Point(x: 3, y: 4)\n\
         \tprintln(p.x * p.x + p.y * p.y)\n\
         }\n",
        "25\n",
    );
}

/// A list walked by recursion, and one walked by a loop, 100,000 long.
#[test]
fn a_list_is_walked() {
    prints(
        "type L {\n\
         \tCons(h Int, t L)\n\
         \tEnd\n\
         }\n\
         fn build(n Int, acc L) L {\n\
         \tif n == 0 { acc } else { build(n - 1, Cons(n, acc)) }\n\
         }\n\
         fn sum(l L) Int {\n\
         \tmatch l {\n\
         \t\tCons(h, t) -> h + sum(t)\n\
         \t\tEnd -> 0\n\
         \t}\n\
         }\n\
         fn length(l L, n Int) Int {\n\
         \tmatch l {\n\
         \t\tCons(_, t) -> length(t, n + 1)\n\
         \t\tEnd -> n\n\
         \t}\n\
         }\n\
         pub fn main() {\n\
         \tprintln(sum(build(10, End)))\n\
         \tprintln(length(build(100000, End), 0))\n\
         }\n",
        "55\n100000\n",
    );
}

#[test]
fn a_closure_sees_what_it_captured() {
    prints(
        "fn apply(f fn(Int) String, x Int) String { f(x) }\n\
         pub fn main() {\n\
         \tn = 5\n\
         \ts = 'x'\n\
         \tprintln(apply(fn(a) { '${s}${a + n}' }, 1))\n\
         }\n",
        "x6\n",
    );
}

/// A closure made in one call and called after that call has returned.
#[test]
fn a_returned_closure_keeps_its_captures() {
    prints(
        "fn adder(n Int) fn(Int) Int { fn(x) { x + n } }\n\
         fn twice(f fn(Int) Int, x Int) Int { f(f(x)) }\n\
         pub fn main() {\n\
         \tadd3 = adder(3)\n\
         \tprintln(add3(1))\n\
         \tprintln(twice(add3, 1))\n\
         \tprintln(adder(2)(4))\n\
         \tprintln(twice(adder(10), 0))\n\
         }\n",
        "4\n7\n6\n20\n",
    );
}

/// A lambda that calls itself keeps its captures on every call, and one
/// that loops by calling itself in tail position runs in constant space.
#[test]
fn a_lambda_calling_itself_keeps_its_captures() {
    prints(
        "pub fn main() {\n\
         \tk = 7\n\
         \tgo = fn(n) { if n == 0 { k } else { go(n - 1) } }\n\
         \tprintln(go(3))\n\
         \tprintln(go(100000))\n\
         \tdepth = fn(n) { if n == 0 { k } else { 1 + depth(n - 1) } }\n\
         \tprintln(depth(10))\n\
         \th = fn(n) { if n == 0 { go } else { h(n - 1) } }\n\
         \tprintln(h(2)(0))\n\
         }\n",
        "7\n7\n17\n7\n",
    );
}

/// A named function works as a value as well as a lambda does.
#[test]
fn a_named_function_is_a_value() {
    prints(
        "fn double(x Int) Int { x * 2 }\n\
         fn apply(f fn(Int) Int, x Int) Int { f(x) }\n\
         pub fn main() {\n\
         \tprintln(apply(double, 21))\n\
         \tf = double\n\
         \tprintln(f(4))\n\
         \tprintln(double)\n\
         \tprintln(Some(double))\n\
         }\n",
        "42\n8\n<fn#double>\nSome(<fn#double>)\n",
    );
}

#[test]
fn a_tuple_is_built_read_and_shown() {
    prints(
        "fn swap(t (Int, String)) (String, Int) { (t.1, t.0) }\n\
         fn sum(t (Int, (Int, Int))) Int {\n\
         \tmatch t {\n\
         \t\t(a, (b, c)) -> a + b + c\n\
         \t}\n\
         }\n\
         pub fn main() {\n\
         \tprintln(swap((1, 'one')))\n\
         \tprintln(sum((1, (2, 3))))\n\
         \tprintln((1, (2, 3)))\n\
         \tprintln(Some((True, Nil)))\n\
         \tprintln('${(1, 2)}')\n\
         }\n",
        "(one, 1)\n6\n(\n  1,\n  (2, 3)\n)\nSome(\n  (True, Nil)\n)\n(1, 2)\n",
    );
}

/// A tuple of small values stays on one line up to 80 columns, and takes a
/// line per element past that.
#[test]
fn a_wide_tuple_takes_a_line_per_element() {
    let exactly_80 =
        "('aaaaaaaaaaaaaaaaaa', 'bbbbbbbbbbbbbbbbbb', 'cccccccccccccccccc', 'dddddddddddddddddd')";
    prints(
        &format!("pub fn main() {{\n\tprintln({exactly_80})\n}}\n"),
        "(aaaaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbbbbb, cccccccccccccccccc, dddddddddddddddddd)\n",
    );
    let past_80 =
        "('aaaaaaaaaaaaaaaaaa', 'bbbbbbbbbbbbbbbbbb', 'cccccccccccccccccc', 'ddddddddddddddddddd')";
    prints(
        &format!("pub fn main() {{\n\tprintln({past_80})\n}}\n"),
        "(\n  aaaaaaaaaaaaaaaaaa,\n  bbbbbbbbbbbbbbbbbb,\n  cccccccccccccccccc,\n  ddddddddddddddddddd\n)\n",
    );
}

#[test]
fn an_array_is_built_and_shown() {
    prints(
        "pub fn main() {\n\
         \tprintln([1, 2, 3])\n\
         \tprintln([])\n\
         \tprintln(['a', 'b'])\n\
         \tprintln([[1, 2], [3, 4]])\n\
         \tprintln(Some([1]))\n\
         \tprintln('${[True, False]}')\n\
         }\n",
        "[1, 2, 3]\n[]\n[a, b]\n[\n  [1, 2],\n  [3, 4]\n]\nSome(\n  [1]\n)\n[True, False]\n",
    );
}

/// A long array of small values goes six to a line.
#[test]
fn a_long_array_goes_six_to_a_line() {
    prints(
        "fn upto(n Int, acc Array(Int)) Array(Int) {\n\
         \tif n == 0 { acc } else { upto(n - 1, [n, ..acc]) }\n\
         }\n\
         pub fn main() {\n\
         \tprintln(upto(40, []))\n\
         }\n",
        "[\n  1, 2, 3, 4, 5, 6, \n  7, 8, 9, 10, 11, 12, \n  13, 14, 15, 16, 17, 18, \n  \
         19, 20, 21, 22, 23, 24, \n  25, 26, 27, 28, 29, 30, \n  31, 32, 33, 34, 35, 36, \n  \
         37, 38, 39, 40\n]\n",
    );
}

/// Arrays walked the way Scarlet code walks them: head and rest, built by
/// appending in a loop and by prepending in one, joined and indexed.
#[test]
fn an_array_is_walked_built_and_joined() {
    prints(
        "fn sum(xs Array(Int)) Int {\n\
         \tmatch xs {\n\
         \t\t[] -> 0\n\
         \t\t[h, ..t] -> h + sum(t)\n\
         \t}\n\
         }\n\
         fn total(xs Array(Int), acc Int) Int {\n\
         \tmatch xs {\n\
         \t\t[] -> acc\n\
         \t\t[h, ..t] -> total(t, acc + h)\n\
         \t}\n\
         }\n\
         fn build(n Int, acc Array(Int)) Array(Int) {\n\
         \tif n == 0 { acc } else { build(n - 1, [..acc, n]) }\n\
         }\n\
         fn describe(xs Array(Int)) String {\n\
         \tmatch xs {\n\
         \t\t[] -> 'empty'\n\
         \t\t[a] -> 'one ${a}'\n\
         \t\t[a, b] -> 'two ${a} ${b}'\n\
         \t\t[a, _, ..rest] -> 'from ${a}, then ${rest}'\n\
         \t}\n\
         }\n\
         pub fn main() {\n\
         \tprintln(sum([1, 2, 3, 4]))\n\
         \tbig = build(10000, [])\n\
         \tprintln(total(big, 0))\n\
         \tprintln(describe([]))\n\
         \tprintln(describe([7]))\n\
         \tprintln(describe([7, 8]))\n\
         \tprintln(describe([7, 8, 9, 10]))\n\
         \tprintln([0, ..[1, 2], 3])\n\
         \tprintln([..[1, 2], ..[3, 4]])\n\
         \tprintln([10, 20, 30][1] or 99)\n\
         \tprintln([10, 20, 30][3] or 99)\n\
         \tprintln([10, 20, 30][-1] or 99)\n\
         \tprintln(big[9999] or 99)\n\
         \tprintln(big[9223372036854775807] or 99)\n\
         }\n",
        "10\n50005000\nempty\none 7\ntwo 7 8\nfrom 7, then [9, 10]\n[0, 1, 2, 3]\n[1, 2, 3, 4]\n\
         20\n99\n99\n1\n99\n",
    );
}

/// `xs[i]` is an `Option`: `Some` of the element, or `None` past either end.
#[test]
fn indexing_an_array_gives_an_option() {
    prints(
        "pub fn main() {\n\
         \txs = ['a', 'b', 'c']\n\
         \tprintln(xs[0])\n\
         \tprintln(xs[2])\n\
         \tprintln(xs[3])\n\
         \tprintln(xs[-1])\n\
         \tprintln(xs[9223372036854775807])\n\
         \tmatch xs[1] {\n\
         \t\tSome(s) -> println('found ${s}')\n\
         \t\tNone -> println('none')\n\
         \t}\n\
         \tprintln(xs[5] or 'default')\n\
         \tprintln(xs[1 - 2] or 'computed')\n\
         }\n",
        "Some(a)\nSome(c)\nNone\nNone\nNone\nfound b\ndefault\ncomputed\n",
    );
}

/// `xs[a..b]` is a `Result`: `Ok` of the elements when the range is inside
/// the array, and `Err(Nil)` when it is not, never a crash.
#[test]
fn a_slice_is_a_result() {
    prints(
        "fn upto(n Int, acc Array(Int)) Array(Int) {\n\
         \tif n == 0 { acc } else { upto(n - 1, [n, ..acc]) }\n\
         }\n\
         pub fn main() {\n\
         \txs = [1, 2, 3, 4, 5]\n\
         \tprintln(xs[1..3])\n\
         \tprintln(xs[0..5])\n\
         \tprintln(xs[2..2])\n\
         \tprintln(xs[5..5])\n\
         \tprintln(xs[3..9])\n\
         \tprintln(xs[3..1])\n\
         \tprintln(xs[-1..2])\n\
         \tprintln(xs[0..9223372036854775807])\n\
         \tbig = upto(5000, [])\n\
         \tmatch big[4990..4995] {\n\
         \t\tOk(part) -> println(part)\n\
         \t\tErr(Nil) -> println('missed')\n\
         \t}\n\
         \tprintln(xs[9..10] or [0])\n\
         }\n",
        "Ok(\n  [2, 3]\n)\nOk(\n  [1, 2, 3, 4, 5]\n)\nOk(\n  []\n)\nOk(\n  []\n)\n\
         Err(Nil)\nErr(Nil)\nErr(Nil)\nErr(Nil)\n[4991, 4992, 4993, 4994, 4995]\n[0]\n",
    );
}

/// A range stores only its two ends, and is an array of Ints in every way a
/// program can see: shown, indexed, walked, sliced, spread and joined.
#[test]
fn a_range_is_an_array_of_its_ints() {
    prints(
        "fn total(xs Array(Int), acc Int) Int {\n\
         \tmatch xs {\n\
         \t\t[] -> acc\n\
         \t\t[h, ..t] -> total(t, acc + h)\n\
         \t}\n\
         }\n\
         pub fn main() {\n\
         \tprintln(1..4)\n\
         \tprintln(3..3)\n\
         \tprintln(5..2)\n\
         \tprintln({ 10..20 }[2])\n\
         \tprintln({ 10..20 }[10])\n\
         \tprintln(total(0..100001, 0))\n\
         \tprintln({ 0..10 }[2..5])\n\
         \tprintln({ 0..10 }[8..11])\n\
         \tprintln([1, ..{ 5..8 }, 9])\n\
         \tprintln([..{ 0..2 }, ..{ 7..9 }])\n\
         \tprintln('${0..3}')\n\
         \tprintln(0..40)\n\
         \tprintln({ 9223372036854775800..9223372036854775807 }[6] or 0)\n\
         \tbig = 0..1000000000000\n\
         \tprintln(big[999999999999] or 0)\n\
         }\n",
        "[1, 2, 3]\n[]\n[]\nSome(12)\nNone\n5000050000\nOk(\n  [2, 3, 4]\n)\nErr(Nil)\n\
         [1, 5, 6, 7, 9]\n[0, 1, 7, 8]\n[0, 1, 2]\n\
         [\n  0, 1, 2, 3, 4, 5, \n  6, 7, 8, 9, 10, 11, \n  12, 13, 14, 15, 16, 17, \n  \
         18, 19, 20, 21, 22, 23, \n  24, 25, 26, 27, 28, 29, \n  30, 31, 32, 33, 34, 35, \n  \
         36, 37, 38, 39\n]\n9223372036854775806\n999999999999\n",
    );
}

/// `==` compares what values hold, of every kind the VM builds.
#[test]
fn equality_is_structural() {
    prints(
        "type Shape {\n\
         \tCircle(r Int)\n\
         \tDot\n\
         }\n\
         fn add(n Int) fn(Int) Int { fn(x) { x + n } }\n\
         pub fn main() {\n\
         \tprintln(1 == 1)\n\
         \tprintln(1 != 2)\n\
         \tprintln(9223372036854775807 + 1 == 9223372036854775807 + 1)\n\
         \tprintln('ab' == 'a${'b'}')\n\
         \tprintln('ab' == 'abc')\n\
         \tprintln(Circle(r: 2) == Circle(r: 2))\n\
         \tprintln(Circle(r: 2) == Circle(r: 3))\n\
         \tprintln(Circle(r: 2) == Dot)\n\
         \tprintln(Dot == Dot)\n\
         \tprintln(Some([1, 2]) == Some([1, 2]))\n\
         \tprintln((1, 'a') == (1, 'a'))\n\
         \tprintln((1, 'a') == (1, 'b'))\n\
         \tprintln([1, 2, 3] == [1, 2])\n\
         \tprintln(0..3 == [0, 1, 2])\n\
         \tprintln([0, 1, 2] == 0..3)\n\
         \tprintln(5..2 == 7..7)\n\
         \tprintln(0..3 == 1..4)\n\
         \tprintln(add(1) == add(1))\n\
         \tprintln(add(1) == add(2))\n\
         \tprintln(True == !False)\n\
         \tprintln(Nil == Nil)\n\
         }\n",
        "True\nTrue\nTrue\nTrue\nFalse\nTrue\nFalse\nFalse\nTrue\nTrue\nTrue\nFalse\nFalse\n\
         True\nTrue\nTrue\nFalse\nTrue\nFalse\nTrue\nTrue\n",
    );
}

/// Two lists 200,000 long compare without overflowing the stack, and
/// the first difference, however deep, decides.
#[test]
fn equality_walks_deep_values_in_constant_stack() {
    prints(
        "type L {\n\
         \tCons(h Int, t L)\n\
         \tEnd\n\
         }\n\
         fn build(n Int, last Int, acc L) L {\n\
         \tif n == 0 { Cons(last, acc) } else { build(n - 1, last, Cons(n, acc)) }\n\
         }\n\
         pub fn main() {\n\
         \tprintln(build(200000, 0, End) == build(200000, 0, End))\n\
         \tprintln(build(200000, 0, End) == build(200000, 1, End))\n\
         }\n",
        "True\nFalse\n",
    );
}

#[test]
fn the_first_built_ins_run() {
    prints(
        "import scarlet/array\n\
         import scarlet/int\n\
         import scarlet/string\n\
         pub fn main() {\n\
         \tprintln(string.inspect(Some([1, 2])))\n\
         \tprintln(string.inspect('as is'))\n\
         \tprintln(string.length('héllo'))\n\
         \tprintln(string.length(''))\n\
         \tprintln(array.length([1, 2, 3]))\n\
         \tprintln(array.length(0..1000000000000))\n\
         \tprintln(int.to_string(-42))\n\
         \tprintln(int.to_string(9223372036854775807 + 1))\n\
         \tprintln(array.map([1, 2, 3], fn(x) { x * 10 }))\n\
         \tprintln(array.reverse([1, 2, 3]))\n\
         }\n",
        "Some(\n  [1, 2]\n)\nas is\n5\n0\n3\n1000000000000\n-42\n9223372036854775808\n[10, 20, 30]\n[3, 2, 1]\n",
    );
}

/// Floats follow `docs/semantics.md`: no NaN and no infinity, ever.
#[test]
fn floats_are_never_nan_or_infinite() {
    prints(
        "import scarlet/float\n\
         fn double(x) { x + x }\n\
         fn big(x Float, n Int) Float { if n == 0 { x } else { big(x * 10.0, n - 1) } }\n\
         pub fn main() {\n\
         \tprintln(1.5 + 2.25)\n\
         \tprintln(1.0)\n\
         \tprintln(0.1 + 0.2)\n\
         \tprintln(-2.5 * 4.0)\n\
         \tprintln(1.0 / 0.0)\n\
         \tprintln(0.0 / 0.0)\n\
         \tprintln(7.5 % 0.0)\n\
         \tprintln(-7.5 % 2.0)\n\
         \tprintln(big(1.0, 400))\n\
         \tprintln(big(-1.0, 400))\n\
         \tprintln(big(1.0, 400) > 1.0)\n\
         \tprintln(0.0 == -0.0)\n\
         \tprintln(2.5 < 3.0)\n\
         \tprintln(double(1.25))\n\
         \tprintln(double(3))\n\
         \tn = 2.5\n\
         \tprintln(-n)\n\
         \tprintln('${Some(1.5)}')\n\
         \tprintln(float.floor(-2.5))\n\
         \tprintln(float.round(2.5))\n\
         \tprintln(float.truncate(-2.9))\n\
         \tprintln(float.floor(big(1.0, 20)))\n\
         \tprintln(float.from_int(3))\n\
         \tprintln(float.to_string(2.0))\n\
         }\n",
        &format!(
            "3.75\n1.0\n0.30000000000000004\n-10.0\n0.0\n0.0\n7.5\n-1.5\n\
             {max}.0\n-{max}.0\nTrue\nTrue\nTrue\n2.5\n6\n-2.5\nSome(1.5)\n-3\n3\n-2\n\
             100000000000000000000\n3.0\n2.0\n",
            max = f64::MAX
        ),
    );
}

/// Binaries: literals of every segment kind, shown as the old VM showed
/// them, taken apart by patterns, and the core of `scarlet/binary`.
#[test]
fn binaries_are_built_shown_and_matched() {
    prints(
        "import scarlet/binary\n\
         fn describe(b Binary) String {\n\
         \tmatch b {\n\
         \t\t<<1, rest:binary>> -> 'one then ${rest}'\n\
         \t\t<<n:size(16), _:binary>> -> 'a 16-bit ${n}'\n\
         \t\t_ -> 'something else'\n\
         \t}\n\
         }\n\
         fn first_char(b Binary) Int {\n\
         \tmatch b {\n\
         \t\t<<c:utf8, _:binary>> -> c\n\
         \t\t_ -> -1\n\
         \t}\n\
         }\n\
         pub fn main() {\n\
         \tprintln(<<1, 2, 3>>)\n\
         \tprintln(<<>>)\n\
         \tprintln(<<5:size(3)>>)\n\
         \tprintln(<<255, 5:size(3)>>)\n\
         \tprintln(<<-1:size(12)>>)\n\
         \tprintln(<<'hi':utf8>>)\n\
         \tprintln(<<1:size(4), 2:size(4)>>)\n\
         \tprintln(describe(<<1, 7, 9>>))\n\
         \tprintln(describe(<<2, 1>>))\n\
         \tprintln(describe(<<2>>))\n\
         \tprintln(first_char(<<'é':utf8>>))\n\
         \tprintln(first_char(<<255>>))\n\
         \tprintln(<<1, 2>> == <<1, 2>>)\n\
         \tprintln(<<1, 2>> == <<1, 2, 0>>)\n\
         \tprintln(binary.bit_size(<<1, 5:size(3)>>))\n\
         \tprintln(binary.byte_size(<<1, 5:size(3)>>))\n\
         \tprintln(binary.to_string(binary.from_string('héllo')))\n\
         \tprintln(binary.to_string(<<255>>))\n\
         \tprintln(binary.append(<<1>>, <<2, 3>>))\n\
         \tprintln(binary.slice_bits(<<1, 2, 3>>, 8, 16))\n\
         \tprintln(binary.slice_bits(<<1, 2, 3>>, 8, 17))\n\
         \tprintln(binary.slice_bits(<<1, 2, 3>>, -1, 8))\n\
         \tprintln(binary.slice_bytes(<<1, 2, 3>>, 1, 1) == Ok(<<2>>))\n\
         \tprintln(binary.drop_bytes(<<1, 2, 3>>, 2))\n\
         }\n",
        "<<1, 2, 3>>\n<<>>\n<<5:size(3)>>\n<<255, 5:size(3)>>\n<<255, 15:size(4)>>\n<<104, 105>>\n<<18>>\n\
         one then <<7, 9>>\na 16-bit 513\nsomething else\n233\n-1\nTrue\nFalse\n11\n2\n\
         Ok(héllo)\nErr(Nil)\n<<1, 2, 3>>\nOk(<<2, 3>>)\nErr(Nil)\nErr(Nil)\nTrue\n<<3>>\n",
    );
}

/// The rest of `scarlet/binary`, with its edge cases.
#[test]
fn the_rest_of_the_binary_module_runs() {
    prints(
        "import scarlet/binary\n\
         pub fn main() {\n\
         \tb = <<'hello':utf8>>\n\
         \tprintln(binary.byte_at(b, 0))\n\
         \tprintln(binary.byte_at(b, 4))\n\
         \tprintln(binary.byte_at(b, 5))\n\
         \tprintln(binary.byte_at(b, -1))\n\
         \tprintln(binary.byte_at(<<1, 5:size(3)>>, 1))\n\
         \tprintln(binary.index_of(b, <<'l':utf8>>, 0))\n\
         \tprintln(binary.index_of(b, <<'l':utf8>>, 3))\n\
         \tprintln(binary.index_of(b, <<'z':utf8>>, 0))\n\
         \tprintln(binary.index_of(b, <<>>, 2))\n\
         \tprintln(binary.index_of(b, <<'h':utf8>>, 99))\n\
         \tprintln(binary.parse_int(<<'1234':utf8>>, binary.Dec))\n\
         \tprintln(binary.parse_int(<<'ff':utf8>>, binary.Hex))\n\
         \tprintln(binary.parse_int(<<'FF':utf8>>, binary.Hex))\n\
         \tprintln(binary.parse_int(<<'12a':utf8>>, binary.Dec))\n\
         \tprintln(binary.parse_int(<<>>, binary.Dec))\n\
         \tprintln(binary.parse_int(<<'-1':utf8>>, binary.Dec))\n\
         \tprintln(binary.parse_int(<<'99999999999999999999':utf8>>, binary.Dec))\n\
         \tprintln(binary.eq_ignore_ascii_case(<<'Content-Length':utf8>>, <<'content-length':utf8>>))\n\
         \tprintln(binary.eq_ignore_ascii_case(<<'ab':utf8>>, <<'abc':utf8>>))\n\
         \tprintln(binary.to_ascii_lower(<<'HeLLo':utf8>>) == <<'hello':utf8>>)\n\
         \tprintln(binary.from_int_ascii(255, binary.Hex) == <<'ff':utf8>>)\n\
         \tprintln(binary.from_int_ascii(-42, binary.Dec) == <<'-42':utf8>>)\n\
         }\n",
        "104\n111\n-1\n-1\n-1\nSome(2)\nSome(3)\nNone\nSome(2)\nNone\n\
         Ok(1234)\nOk(255)\nOk(255)\nErr(Nil)\nErr(Nil)\nErr(Nil)\nOk(99999999999999999999)\n\
         True\nFalse\nTrue\nTrue\nTrue\n",
    );
}

#[test]
fn the_string_built_ins_run() {
    prints(
        "import scarlet/int\n\
         import scarlet/string\n\
         pub fn main() {\n\
         \tprintln(string.split('a,b,,c', ','))\n\
         \tprintln(string.split('abc', ''))\n\
         \tprintln(string.split('abc', 'x'))\n\
         \tprintln(string.split('', ','))\n\
         \tprintln(string.contains('hello', 'ell'))\n\
         \tprintln(string.contains('hello', ''))\n\
         \tprintln(string.contains('hello', 'z'))\n\
         \tprintln('[${string.trim('  hi \\t\\n')}]')\n\
         \tprintln(string.to_graphemes('e\\u{0301}a'))\n\
         \tprintln(int.from_string('-42'))\n\
         \tprintln(int.from_string('+007'))\n\
         \tprintln(int.from_string('20O'))\n\
         \tprintln(int.from_string(' 1'))\n\
         \tprintln(int.from_string('-'))\n\
         \tprintln(int.from_string('123456789012345678901234567890'))\n\
         }\n",
        "[a, b, , c]\n[a, b, c]\n[abc]\n[]\nTrue\nTrue\nFalse\n[hi]\n[e\u{0301}, a]\n\
         Ok(-42)\nOk(7)\nErr(Nil)\nErr(Nil)\nErr(Nil)\nOk(123456789012345678901234567890)\n",
    );
}

/// Bitwise operations treat an Int as an endless row of two's-complement
/// bits, as `int.scrl` says: no bit ever falls off an end.
#[test]
fn bitwise_operations_are_any_size() {
    prints(
        "import scarlet/binary\n\
         import scarlet/int\n\
         pub fn main() {\n\
         \tprintln(int.bitwise_and(12, 10))\n\
         \tprintln(int.bitwise_or(12, 10))\n\
         \tprintln(int.bitwise_xor(12, 10))\n\
         \tprintln(int.bitwise_not(0))\n\
         \tprintln(int.bitwise_not(5))\n\
         \tprintln(int.bitwise_and(-1, 255))\n\
         \tprintln(int.bitwise_shift_left(1, 64))\n\
         \tprintln(int.bitwise_shift_left(3, -1))\n\
         \tprintln(int.bitwise_shift_right(-8, 1))\n\
         \tprintln(int.bitwise_shift_right(-7, 1))\n\
         \tprintln(int.bitwise_shift_right(5, 100))\n\
         \tprintln(int.bitwise_shift_right(-5, 100))\n\
         \tprintln(int.bitwise_shift_right(1, -3))\n\
         \tbig = int.bitwise_shift_left(1, 100)\n\
         \tprintln(int.bitwise_and(big + 5, 7))\n\
         \tprintln(int.bitwise_shift_right(big, 99))\n\
         \tprintln(int.bitwise_and(int.bitwise_shift_right(-1, 4), int.bitwise_shift_left(1, 28) - 1))\n\
         \tprintln(binary.hex_byte(13) == <<'0D':utf8>>)\n\
         \tprintln(binary.hex_byte(255) == <<'FF':utf8>>)\n\
         }\n",
        "8\n14\n6\n-1\n-6\n255\n18446744073709551616\n1\n-4\n-4\n0\n-1\n8\n5\n2\n268435455\nTrue\nTrue\n",
    );
}

#[test]
fn the_map_built_ins_run() {
    prints(
        "import scarlet/map\n\
         pub fn main() {\n\
         \tm = map.set(map.set(map.new(), 'a', 1), 'b', 2)\n\
         \tprintln(map.get(m, 'a'))\n\
         \tprintln(map.get(m, 'z'))\n\
         \tprintln(map.has(m, 'b'))\n\
         \tprintln(map.size(m))\n\
         \tn = map.set(m, 'a', 10)\n\
         \tprintln('${map.get(n, 'a')} ${map.get(m, 'a')} ${map.size(n)}')\n\
         \tgone = map.delete(m, 'a')\n\
         \tprintln('${map.has(gone, 'a')} ${map.has(m, 'a')} ${map.size(gone)}')\n\
         \tprintln(map.size(map.delete(m, 'nope')))\n\
         \tprintln(gone)\n\
         \tprintln(map.keys(gone))\n\
         \tprintln(map.values(gone))\n\
         \tprintln(map.to_list(gone))\n\
         \tprintln(map.new())\n\
         \tprintln(map.delete(gone, 'b'))\n\
         }\n",
        "Some(1)\nNone\nTrue\n2\nSome(10) Some(1) 2\nFalse True 1\n2\n{b: 2}\n[b]\n[2]\n[\n  (b, 2)\n]\n{}\n{}\n",
    );
}

/// A map's order depends on its entries alone, so two maps that are `==`
/// list them the same, however they were built.
#[test]
fn equal_maps_are_in_the_same_order() {
    prints(
        "import scarlet/map\n\
         import scarlet/array\n\
         fn fill(m map.Map(Int, Int), from Int, to Int, step Int) map.Map(Int, Int) {\n\
         \tif from == to { m } else { fill(map.set(m, from, from * 2), from + step, to, step) }\n\
         }\n\
         pub fn main() {\n\
         \tup = fill(map.new(), 0, 500, 1)\n\
         \tdown = fill(map.new(), 499, -1, -1)\n\
         \tprintln(up == down)\n\
         \tprintln(map.keys(up) == map.keys(down))\n\
         \tprintln('${up}' == '${down}')\n\
         \tfewer = map.delete(map.set(down, 1000, 0), 1000)\n\
         \tprintln(fewer == up)\n\
         \tprintln(map.to_list(fewer) == map.to_list(up))\n\
         \tprintln(up == map.set(up, 7, 0))\n\
         \tprintln(up == map.delete(up, 7))\n\
         \tprintln(array.length(map.keys(up)))\n\
         }\n",
        "True\nTrue\nTrue\nTrue\nTrue\nFalse\nFalse\n500\n",
    );
}

/// A key is found by any value `==` to it, however that value was made.
#[test]
fn a_key_is_found_by_any_equal_value() {
    prints(
        "import scarlet/map\n\
         import scarlet/binary\n\
         import scarlet/int\n\
         type P { P(x Int, y String) }\n\
         fn one(key k, look k) Option(String) { map.get(map.set(map.new(), key, 'found'), look) }\n\
         pub fn main() {\n\
         \tprintln(one(0..3, [0, 1, 2]))\n\
         \tprintln(one(0.0, -0.0))\n\
         \tprintln(match binary.slice_bits(<<1, 2, 3>>, 8, 16) {\n\
         \t\tOk(b) -> one(<<2, 3>>, b)\n\
         \t\tErr(Nil) -> None\n\
         \t})\n\
         \tprintln(one(int.bitwise_shift_left(1, 100), int.bitwise_shift_left(2, 99)))\n\
         \tprintln(one((1, 'a'), (1, 'a')))\n\
         \tprintln(one(P(1, 'p'), P(1, 'p')))\n\
         \tprintln(one('abc', 'ab${'c'}'))\n\
         \tinner = map.set(map.set(map.new(), 1, 'one'), 2, 'two')\n\
         \tprintln(one(inner, map.set(map.set(map.new(), 2, 'two'), 1, 'one')))\n\
         \tprintln(one(0..3, [0, 1]))\n\
         \tprintln(one(P(1, 'p'), P(1, 'q')))\n\
         \tprintln(one(inner, map.delete(inner, 1)))\n\
         }\n",
        "Some(found)\nSome(found)\nSome(found)\nSome(found)\nSome(found)\nSome(found)\nSome(found)\nSome(found)\nNone\nNone\nNone\n",
    );
}

/// `os.argv` and `os.env` are what the host says. A name the OS lists twice
/// is the first one's, as `getenv` has it.
#[test]
fn argv_and_env_come_from_the_host() {
    let env = [("HOME", "/home/al"), ("LANG", "C"), ("HOME", "/elsewhere")];
    let host = Host::new(
        vec!["main.scrl".into(), "--fast".into(), "two words".into()],
        env.iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
    );
    prints_in(
        &host,
        "import scarlet/os\n\
         import scarlet/map\n\
         pub fn main() {\n\
         \tprintln(os.argv())\n\
         \tprintln(os.get_env('HOME'))\n\
         \tprintln(os.get_env('LANG'))\n\
         \tprintln(os.get_env('NOT_SET'))\n\
         \tprintln(map.size(os.env()))\n\
         \tprintln(os.env() == os.env())\n\
         }\n",
        "[main.scrl, --fast, two words]\nSome(/home/al)\nSome(C)\nNone\n2\nTrue\n",
    );
}

/// A file written is the file read back, and each way a path can fail is the
/// `IoError` that names it, holding the path.
#[test]
fn files_read_and_write_and_fail_with_their_error() {
    let dir = scratch("files");
    let file = dir.join("a.txt");
    std::fs::write(dir.join("plain"), "x").expect("a plain file");
    let src = "import scarlet/io\n\
         fn say(r Result(a, io.IoError)) String {\n\
         \tmatch r {\n\
         \t\tOk(v) -> 'ok ${v}'\n\
         \t\tErr(io.NotFound(p)) -> 'not found ${p}'\n\
         \t\tErr(io.IsADirectory(p)) -> 'a directory ${p}'\n\
         \t\tErr(io.NotADirectory(p)) -> 'not a directory ${p}'\n\
         \t\tErr(io.InvalidData(p)) -> 'not text ${p}'\n\
         \t\tErr(io.UnalignedBinary) -> 'not whole bytes'\n\
         \t\tErr(e) -> 'another error ${e}'\n\
         \t}\n\
         }\n\
         pub fn main() {\n\
         \tprintln(say(io.write_text('DIR/a.txt', 'hello')))\n\
         \tprintln(say(io.read_text('DIR/a.txt')))\n\
         \tprintln(say(io.write_file('DIR/a.txt', <<104, 105>>)))\n\
         \tprintln(say(io.read_file('DIR/a.txt')))\n\
         \tprintln(say(io.write_file('DIR/b.txt', <<1:4>>)))\n\
         \tprintln(say(io.read_file('DIR/missing')))\n\
         \tprintln(say(io.read_file('DIR')))\n\
         \tprintln(say(io.write_text('DIR/no/such/dir', 'x')))\n\
         \tprintln(say(io.read_file('DIR/plain/under')))\n\
         \tprintln(say(io.write_file('DIR/x.bin', <<255>>)))\n\
         \tprintln(say(io.read_text('DIR/x.bin')))\n\
         }\n"
    .replace("DIR", &dir.display().to_string());
    let d = dir.display();
    let want = format!(
        "ok Nil\nok hello\nok Nil\nok <<104, 105>>\nnot whole bytes\n\
         not found {d}/missing\na directory {d}\nnot found {d}/no/such/dir\n\
         not a directory {d}/plain/under\nok Nil\nnot text {d}/x.bin\n"
    );
    prints_in(&Host::new(Vec::new(), Vec::new()), &src, &want);
    assert_eq!(std::fs::read(&file).expect("written"), b"hi");
    assert!(
        !dir.join("b.txt").exists(),
        "an unaligned write made a file"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The monotonic clock never goes back; the wall clock says it is after
/// 2023. Random bytes come in the number asked for, and a negative number
/// is an error.
#[test]
fn the_clocks_and_random_bytes_run() {
    prints(
        "import scarlet/time\n\
         import scarlet/crypto\n\
         import scarlet/binary\n\
         pub fn main() {\n\
         \tstart = time.monotonic()\n\
         \tprintln(time.since_ms(time.monotonic(), start) >= 0)\n\
         \tprintln(time.epoch_ms() > 1700000000000)\n\
         \tprintln(match crypto.random_bytes(33) {\n\
         \t\tOk(b) -> binary.byte_size(b)\n\
         \t\tErr(Nil) -> -1\n\
         \t})\n\
         \tprintln(crypto.random_bytes(0))\n\
         \tprintln(crypto.random_bytes(-1))\n\
         \tprintln(crypto.random_bytes(16) == crypto.random_bytes(16))\n\
         }\n",
        "True\nTrue\n33\nOk(<<>>)\nErr(Nil)\nFalse\n",
    );
}

/// `<<1, 2, 3>>`, to put bytes into a program's source.
fn bytes_literal(bytes: &[u8]) -> String {
    let parts: Vec<String> = bytes.iter().map(u8::to_string).collect();
    format!("<<{}>>", parts.join(", "))
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
        .collect()
}

/// The digests and the HMAC tag give the published answers (FIPS 180-2's
/// `abc`, RFC 4231's second case), and a binary that is not whole bytes has
/// none. `const_eq` compares bits: the old VM dropped a last partial byte, so
/// `<<1, 2:4>>` and `<<1, 3:4>>` were equal. Signatures made here by aws-lc
/// check out, and anything of the wrong shape does not.
#[test]
fn the_crypto_built_ins_run() {
    use aws_lc_rs::rand::SystemRandom;
    use aws_lc_rs::signature::{
        ECDSA_P256_SHA256_ASN1_SIGNING, EcdsaKeyPair, Ed25519KeyPair, KeyPair,
    };
    let rng = SystemRandom::new();
    let p256 = EcdsaKeyPair::generate(&ECDSA_P256_SHA256_ASN1_SIGNING).expect("a P-256 key");
    let p256_sig = p256.sign(&rng, b"signed").expect("a P-256 signature");
    let ed_doc = Ed25519KeyPair::generate_pkcs8(&rng).expect("an Ed25519 key");
    let ed = Ed25519KeyPair::from_pkcs8(ed_doc.as_ref()).expect("an Ed25519 key");
    let ed_sig = ed.sign(b"signed");
    let src = "import scarlet/crypto\n\
         import scarlet/binary\n\
         pub fn main() {\n\
         \tprintln(crypto.sha256(<<'abc'>>) == Ok(SHA256_ABC))\n\
         \tprintln(crypto.sha512(<<'abc'>>) == Ok(SHA512_ABC))\n\
         \tprintln(crypto.sha1(<<'abc'>>) == Ok(SHA1_ABC))\n\
         \tprintln(crypto.hmac_sha256(<<'Jefe'>>, <<'what do ya want for nothing?'>>) == Ok(HMAC_JEFE))\n\
         \tprintln(crypto.sha256(<<1:4>>))\n\
         \tprintln(crypto.sha512(<<97, 1:4>>))\n\
         \tprintln(crypto.hmac_sha256(<<1:4>>, <<>>))\n\
         \tprintln(crypto.hmac_sha256(<<>>, <<1:4>>))\n\
         \tprintln(crypto.const_eq(<<1, 2, 3>>, <<1, 2, 3>>))\n\
         \tprintln(crypto.const_eq(<<1, 2, 3>>, <<1, 2, 4>>))\n\
         \tprintln(crypto.const_eq(<<1, 2>>, <<1, 2, 3>>))\n\
         \tprintln(crypto.const_eq(<<1, 2:4>>, <<1, 3:4>>))\n\
         \tprintln(crypto.const_eq(<<1, 2:4>>, <<1, 2:4>>))\n\
         \tprintln(crypto.const_eq(<<1:4>>, <<16>>))\n\
         \tprintln(crypto.p256_verify(P256_KEY, <<'signed'>>, P256_SIG))\n\
         \tprintln(crypto.p256_verify(P256_KEY, <<'signeD'>>, P256_SIG))\n\
         \tprintln(crypto.p256_verify(<<4, 1, 2>>, <<'signed'>>, P256_SIG))\n\
         \tprintln(crypto.p256_verify(P256_KEY, <<'signed', 1:4>>, P256_SIG))\n\
         \tprintln(crypto.ed25519_verify(ED_KEY, <<'signed'>>, ED_SIG))\n\
         \tprintln(crypto.ed25519_verify(ED_KEY, <<'signeD'>>, ED_SIG))\n\
         \tprintln(crypto.ed25519_verify(ED_KEY, <<'signed'>>, <<1, 2, 3>>))\n\
         \tprintln(crypto.ed25519_verify(ED_KEY, <<'signed'>>, P256_SIG))\n\
         }\n"
        .replace(
            "SHA256_ABC",
            &bytes_literal(&unhex(
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )),
        )
        .replace(
            "SHA512_ABC",
            &bytes_literal(&unhex(
                "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a\
                 2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
            )),
        )
        .replace(
            "SHA1_ABC",
            &bytes_literal(&unhex("a9993e364706816aba3e25717850c26c9cd0d89d")),
        )
        .replace(
            "HMAC_JEFE",
            &bytes_literal(&unhex(
                "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843",
            )),
        )
        .replace("P256_KEY", &bytes_literal(p256.public_key().as_ref()))
        .replace("P256_SIG", &bytes_literal(p256_sig.as_ref()))
        .replace("ED_KEY", &bytes_literal(ed.public_key().as_ref()))
        .replace("ED_SIG", &bytes_literal(ed_sig.as_ref()));
    prints(
        &src,
        "True\nTrue\nTrue\nTrue\nErr(Nil)\nErr(Nil)\nErr(Nil)\nErr(Nil)\n\
         True\nFalse\nFalse\nFalse\nTrue\nFalse\n\
         True\nFalse\nFalse\nFalse\n\
         True\nFalse\nFalse\nFalse\n",
    );
}

/// A parsed document is read in place: fields, elements, each kind of
/// scalar, and an integer past `i64::MAX`, which an `Int` holds exactly. A
/// `Doc` shows as what it points at, never the document. A `Json` tree
/// encodes, a `Number` that is not one becoming `null`.
#[test]
fn the_json_built_ins_run() {
    prints(
        "import scarlet/json\n\
         pub fn main() {\n\
         \tsrc = '{\"id\": 18446744073709551615, \"n\": -3, \"x\": 0.5, \"s\": \"a\\\\u00e9\", \"t\": [true, null], \"o\": {}}'\n\
         \tmatch json.parse(src) {\n\
         \t\tOk(d) -> {\n\
         \t\t\tprintln('${d} ${json.size(d)}')\n\
         \t\t\tprintln(json.int(json.field(d, 'id') or d))\n\
         \t\t\tprintln(json.int_text(json.field(d, 'n') or d))\n\
         \t\t\tprintln(json.float(json.field(d, 'x') or d))\n\
         \t\t\tprintln(json.string(json.field(d, 's') or d))\n\
         \t\t\tt = json.field(d, 't') or d\n\
         \t\t\tprintln(json.bool(json.index(t, 0) or d))\n\
         \t\t\tprintln(json.is_null(json.index(t, 1) or d))\n\
         \t\t\tprintln(json.index(t, 2))\n\
         \t\t\tprintln(json.field(d, 'missing'))\n\
         \t\t\tprintln(json.size(json.field(d, 'o') or d))\n\
         \t\t\tprintln(json.reencode(d))\n\
         \t\t}\n\
         \t\tErr(_) -> println('did not parse')\n\
         \t}\n\
         \tprintln(json.encode(json.List([json.Null, json.Boolean(True), json.Integer(-7), json.Real(1.0), json.Str('a\"b'), json.Object([('k', json.Number('1e9'))]), json.Number('x')])))\n\
         \tprintln(match json.parse('[1,]') {\n\
         \t\tOk(_) -> 'parsed'\n\
         \t\tErr(_) -> 'refused'\n\
         \t})\n\
         }\n",
        "<json object#0> 6\nSome(18446744073709551615)\nSome(-3)\nSome(0.5)\nSome(aé)\n\
         Some(True)\nTrue\nNone\nNone\n0\n\
         {\"id\":18446744073709551615,\"n\":-3,\"x\":0.5,\"s\":\"aé\",\"t\":[true,null],\"o\":{}}\n\
         [null,true,-7,1.0,\"a\\\"b\",{\"k\":1e9},null]\nrefused\n",
    );
}

/// A Float literal pattern matches as `==` does, so `-0.0` is `0.0`.
#[test]
fn a_float_pattern_matches_as_equality_does() {
    prints(
        "fn name(x Float) String {\n\
         \tmatch x {\n\
         \t\t0.0 -> 'zero'\n\
         \t\t1.5 -> 'one and a half'\n\
         \t\t_ -> 'other'\n\
         \t}\n\
         }\n\
         pub fn main() {\n\
         \tprintln(name(0.0))\n\
         \tprintln(name(-0.0))\n\
         \tprintln(name(1.5))\n\
         \tprintln(name(1.25 + 0.25))\n\
         \tprintln(name(2.0))\n\
         }\n",
        "zero\nzero\none and a half\none and a half\nother\n",
    );
}

/// What the tests of a `match` whose value is used, rather than returned,
/// share: a two-field constructor, which is always a cell, a function that
/// may give one, and one that reads one. `internal.cells_live()` counts the
/// cells held at a point, so a test pins that a value is freed by then,
/// rather than at the end of its frame.
const CELLS: &str = r"import scarlet/internal

type Big {
	Big(a Int, b Int)
}

fn make(n Int) Option(Big) {
	if n > 0 then Some(Big(n, n)) else None
}

fn use(b Big) Int {
	match b {
		Big(a, _) -> a
	}
}

";

/// `CELLS`, then `src`.
fn prints_cells(src: &str, want: &str) {
    prints(&format!("{CELLS}{src}"), want);
}

/// A `match` whose value is bound rather than returned lowers to a join
/// point, and each arm gives up the scrutinee and what it bound before the
/// code after the join runs, rather than at the end of the frame.
#[test]
fn a_used_match_frees_its_value_in_each_arm() {
    prints_cells(
        r"fn probe(n Int) Int {
	base = internal.cells_live()
	r = match make(n) {
		Some(b) -> use(b)
		None -> 0
	}
	println(internal.cells_live() - base)
	r
}

pub fn main() {
	println(probe(4))
	println(probe(0))
}
",
        "0\n4\n0\n0\n",
    );
}

/// A guard that fails falls through to the next arm, and every way out
/// frees what the arms bound.
#[test]
fn a_used_match_frees_its_value_behind_a_guard() {
    prints_cells(
        r"fn probe(n Int) Int {
	base = internal.cells_live()
	r = match make(n) {
		Some(b) if use(b) > 5 -> 1
		Some(b) -> use(b) + 2
		None -> 0
	}
	println(internal.cells_live() - base)
	r
}

pub fn main() {
	println(probe(1))
	println(probe(9))
	println(probe(0))
}
",
        "0\n3\n0\n1\n0\n0\n",
    );
}

/// A used match inside an arm of another is a join inside a join: the
/// inner one frees its own value, the outer arm's `Some` goes as the arm
/// starts, and its `b` once its last read is done.
#[test]
fn a_used_match_inside_another_frees_both() {
    prints_cells(
        r"fn probe(n Int) Int {
	base = internal.cells_live()
	r = match make(n) {
		Some(b) -> {
			inner = match make(use(b) - 1) {
				Some(c) -> use(c)
				None -> 10
			}
			println(internal.cells_live() - base)
			inner + use(b)
		}
		None -> 0
	}
	println(internal.cells_live() - base)
	r
}

pub fn main() {
	println(probe(3))
	println(probe(1))
}
",
        "1\n0\n5\n1\n0\n11\n",
    );
}

/// A loop that makes a value and matches on it every turn holds none of
/// them once the match is done, in any turn: the count is read after the
/// match in each, and every one is 0.
#[test]
fn a_used_match_in_a_loop_holds_nothing_after_it() {
    prints_cells(
        r"fn spin(n Int, acc Int, most Int, base Int) Int {
	x = match make(n) {
		Some(b) -> use(b)
		None -> 0
	}
	held = internal.cells_live() - base
	top = if held > most then held else most
	if n <= 0 {
		println(top)
		acc
	} else {
		spin(n - 1, acc + x, top, base)
	}
}

pub fn main() {
	base = internal.cells_live()
	println(spin(100, 0, 0, base))
}
",
        "0\n5050\n",
    );
}

/// A value an arm gives as the match's own survives into the code after
/// it, even though the arm bound it: here the `Big` in `r`, and the closure
/// in `f` with the `Big` it captured. Only the `Some` around each goes.
#[test]
fn a_value_a_used_match_gives_survives_it() {
    prints_cells(
        r"fn probe(n Int) Int {
	base = internal.cells_live()
	r = match make(n) {
		Some(b) -> b
		None -> Big(0, 0)
	}
	println(internal.cells_live() - base)
	f = match make(n) {
		Some(b) -> fn() { use(b) }
		None -> fn() { 0 }
	}
	println(internal.cells_live() - base)
	use(r) + f()
}

pub fn main() {
	println(probe(4))
}
",
        "1\n3\n8\n",
    );
}

/// A scrutinee the code after the match still reads is not given up by
/// the match: both of its cells are held until that last read.
#[test]
fn a_value_read_after_a_used_match_is_kept_until_then() {
    prints_cells(
        r"fn probe(n Int) Int {
	base = internal.cells_live()
	m = make(n)
	r = match m {
		Some(b) -> use(b)
		None -> 0
	}
	println(internal.cells_live() - base)
	s = match m {
		Some(b) -> use(b)
		None -> 0
	}
	println(internal.cells_live() - base)
	r + s
}

pub fn main() {
	println(probe(4))
}
",
        "2\n0\n8\n",
    );
}

/// Three used matches in one function: each frees its own value, so the
/// count stays at none rather than growing by two a match.
#[test]
fn used_matches_in_a_row_hold_nothing_between_them() {
    prints_cells(
        r"fn probe() Int {
	base = internal.cells_live()
	x = match make(1) {
		Some(b) -> use(b)
		None -> 0
	}
	println(internal.cells_live() - base)
	y = match make(2) {
		Some(b) -> use(b)
		None -> 0
	}
	println(internal.cells_live() - base)
	z = match make(3) {
		Some(b) -> use(b)
		None -> 0
	}
	println(internal.cells_live() - base)
	x + y + z
}

pub fn main() {
	println(probe())
}
",
        "0\n0\n0\n6\n",
    );
}

/// Every kind of cell an arm can bind is freed by a used match: an array, a
/// closure, a tuple, a binary and a string.
#[test]
fn a_used_match_frees_every_kind_of_cell() {
    prints(
        r"import scarlet/array
import scarlet/binary
import scarlet/internal
import scarlet/string

fn arr(n Int) Option(Array(Int)) {
	if n > 0 then Some([n, n + 1, n + 2]) else None
}

fn clo(n Int) Option(fn() Int) {
	if n > 0 then Some(fn() { n + 1 }) else None
}

fn tup(n Int) Option((Int, Array(Int))) {
	if n > 0 then Some((n, [n])) else None
}

fn bin(n Int) Option(Binary) {
	if n > 0 then Some(<<n:size(16), n:size(16)>>) else None
}

fn str(n Int) Option(String) {
	if n > 0 then Some('n is ${n}') else None
}

fn probe() Int {
	base = internal.cells_live()
	a = match arr(1) {
		Some(xs) -> array.length(xs)
		None -> 0
	}
	println(internal.cells_live() - base)
	c = match clo(1) {
		Some(f) -> f()
		None -> 0
	}
	println(internal.cells_live() - base)
	t = match tup(1) {
		Some((k, xs)) -> k + array.length(xs)
		None -> 0
	}
	println(internal.cells_live() - base)
	b = match bin(1) {
		Some(bits) -> binary.byte_size(bits)
		None -> 0
	}
	println(internal.cells_live() - base)
	s = match str(1) {
		Some(text) -> string.length(text)
		None -> 0
	}
	println(internal.cells_live() - base)
	a + c + t + b + s
}

pub fn main() {
	println(probe())
}
",
        "0\n0\n0\n0\n0\n17\n",
    );
}

/// A value whose type is a type variable is dropped at its last use like any
/// other: `x` is a `Big` here, and passing it to `f` for the last time hands
/// it over, so `f` frees it and nothing holds it when `twice` looks.
#[test]
fn a_generic_value_is_freed_at_its_last_use() {
    prints_cells(
        r"fn twice(x a, f fn(a) Int, base Int) Int {
	n = f(x)
	println(internal.cells_live() - base)
	n * 2
}

pub fn main() {
	base = internal.cells_live()
	println(twice(Big(3, 4), use, base))
}
",
        "0\n6\n",
    );
}

/// A string is a cell, and is freed at its last use rather than with its
/// frame.
#[test]
fn a_string_is_freed_at_its_last_use() {
    prints(
        r"import scarlet/internal
import scarlet/string

fn probe(n Int) Int {
	base = internal.cells_live()
	s = 'n is ${n}'
	k = string.length(s)
	println(internal.cells_live() - base)
	k
}

pub fn main() {
	println(probe(12))
}
",
        "0\n7\n",
    );
}

/// Seeded random programs of used matches, each checked against a liveness
/// oracle: after every statement, the cells held are exactly those reachable
/// from a variable a later statement still reads. The oracle also runs the
/// program, so a value freed too early shows up as a wrong answer or a
/// wrong count, and one freed too late as a count too high.
mod used_match_liveness {
    use std::collections::{BTreeSet, HashMap};

    /// Allocation happens only in the helpers, and none of them builds a
    /// constructor after dropping one of the same width, so no cell is ever
    /// hollowed for reuse and held: the count is exactly the cells reachable.
    const HEADER: &str = r"import scarlet/internal

type Big {
	Big(a Int, b Int, c Int)
}

type Wrap {
	One(x Big)
	Two(x Big, y Big)
	Zero
}

fn big(k Int) Big {
	Big(k, k, k)
}

fn one(b Big) Wrap {
	One(b)
}

fn two(a Big, b Big) Wrap {
	Two(a, b)
}

fn zero() Wrap {
	Zero
}

fn k_of(b Big) Int {
	match b {
		Big(k, _, _) -> k
	}
}

fn w_of(w Wrap) Int {
	match w {
		One(b) -> k_of(b)
		Two(a, b) -> k_of(a) + k_of(b)
		Zero -> 0
	}
}

fn join2(a Big, b Big) Big {
	big(k_of(a) + k_of(b))
}

";

    /// xorshift64: deterministic, seedable, no dependency.
    struct Rng(u64);

    impl Rng {
        fn below(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % n as u64) as usize
        }

        fn chance(&mut self, percent: usize) -> bool {
            self.below(100) < percent
        }
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Ty {
        Big,
        Wrap,
    }

    /// An expression of type `Big`.
    enum Big {
        New(i64),
        Var(String),
        Join(Box<Big>, Box<Big>),
        Match(Box<Match>),
        /// `{ temp = <inner>; join2(temp, with) }`: a used match inside an
        /// arm, so a join inside a join.
        Block {
            temp: String,
            inner: Box<Match>,
            with: Box<Big>,
        },
    }

    /// An expression of type `Wrap`.
    enum Wrap {
        Var(String),
        One(Big),
        Two(Big, Big),
        Zero,
    }

    struct Match {
        scrut: Wrap,
        arms: Vec<Arm>,
    }

    struct Arm {
        pat: Pat,
        /// `if k_of(<first binder>) > n`.
        guard: Option<i64>,
        body: Big,
    }

    enum Pat {
        One(Option<String>),
        Two(Option<String>, Option<String>),
        Zero,
    }

    enum Rhs {
        Big(Big),
        Wrap(Wrap),
    }

    struct Gen {
        rng: Rng,
        fresh: usize,
    }

    impl Gen {
        fn name(&mut self, prefix: &str) -> String {
            self.fresh += 1;
            format!("{prefix}{}", self.fresh)
        }

        fn pick(&mut self, scope: &[(String, Ty)], ty: Ty) -> Option<String> {
            let of: Vec<&String> = scope
                .iter()
                .filter(|(_, t)| *t == ty)
                .map(|(n, _)| n)
                .collect();
            (!of.is_empty()).then(|| of[self.rng.below(of.len())].clone())
        }

        /// A `Big` that allocates at most one cell and reads at most one local.
        fn big_atom(&mut self, scope: &[(String, Ty)]) -> Big {
            match self.pick(scope, Ty::Big) {
                Some(v) if self.rng.chance(60) => Big::Var(v),
                _ => Big::New(self.rng.below(9) as i64 + 1),
            }
        }

        fn wrap(&mut self, scope: &[(String, Ty)]) -> Wrap {
            match self.rng.below(5) {
                0 => match self.pick(scope, Ty::Wrap) {
                    Some(v) => Wrap::Var(v),
                    None => Wrap::Zero,
                },
                1 => Wrap::Zero,
                2 => Wrap::One(self.big_atom(scope)),
                _ => Wrap::Two(self.big_atom(scope), self.big_atom(scope)),
            }
        }

        fn big(&mut self, scope: &[(String, Ty)], depth: usize) -> Big {
            match self.rng.below(if depth == 0 { 3 } else { 5 }) {
                0 => self.big_atom(scope),
                1 => Big::Join(
                    Box::new(self.big_atom(scope)),
                    Box::new(self.big_atom(scope)),
                ),
                2 => match self.pick(scope, Ty::Big) {
                    Some(v) => Big::Var(v),
                    None => Big::New(1),
                },
                3 => Big::Match(Box::new(self.matches(scope, depth - 1))),
                _ => {
                    let inner = self.matches(scope, depth - 1);
                    let with = self.big_atom(scope);
                    Big::Block {
                        temp: self.name("t"),
                        inner: Box::new(inner),
                        with: Box::new(with),
                    }
                }
            }
        }

        /// Every constructor gets an arm, in a rotated order; a guarded one is
        /// followed by an unguarded one for the same constructor, so the
        /// match stays exhaustive.
        fn matches(&mut self, scope: &[(String, Ty)], depth: usize) -> Match {
            let scrut = self.wrap(scope);
            let mut widths = [1, 2, 0];
            widths.rotate_left(self.rng.below(3));
            let mut arms = Vec::new();
            for width in widths {
                if width > 0 && self.rng.chance(30) {
                    let n = self.rng.below(9) as i64;
                    arms.push(self.arm(scope, depth, width, Some(n)));
                }
                arms.push(self.arm(scope, depth, width, None));
            }
            Match { scrut, arms }
        }

        fn arm(
            &mut self,
            scope: &[(String, Ty)],
            depth: usize,
            width: usize,
            guard: Option<i64>,
        ) -> Arm {
            let names: Vec<String> = (0..width).map(|_| self.name("x")).collect();
            let mut inner = scope.to_vec();
            inner.extend(names.iter().map(|n| (n.clone(), Ty::Big)));
            let body = self.big(&inner, depth);
            let mut used = BTreeSet::new();
            big_reads(&body, &mut used);
            if guard.is_some() {
                used.extend(names.first().cloned());
            }
            let keep = |n: &String| used.contains(n).then(|| n.clone());
            let pat = match names.as_slice() {
                [] => Pat::Zero,
                [x] => Pat::One(keep(x)),
                [x, y] => Pat::Two(keep(x), keep(y)),
                _ => unreachable!(),
            };
            Arm { pat, guard, body }
        }
    }

    fn big_reads(b: &Big, out: &mut BTreeSet<String>) {
        match b {
            Big::New(_) => {}
            Big::Var(v) => {
                out.insert(v.clone());
            }
            Big::Join(a, b) => {
                big_reads(a, out);
                big_reads(b, out);
            }
            Big::Match(m) => match_reads(m, out),
            Big::Block { temp, inner, with } => {
                match_reads(inner, out);
                big_reads(with, out);
                out.remove(temp);
            }
        }
    }

    fn wrap_reads(w: &Wrap, out: &mut BTreeSet<String>) {
        match w {
            Wrap::Var(v) => {
                out.insert(v.clone());
            }
            Wrap::One(b) => big_reads(b, out),
            Wrap::Two(a, b) => {
                big_reads(a, out);
                big_reads(b, out);
            }
            Wrap::Zero => {}
        }
    }

    fn match_reads(m: &Match, out: &mut BTreeSet<String>) {
        wrap_reads(&m.scrut, out);
        for arm in &m.arms {
            big_reads(&arm.body, out);
        }
    }

    fn pad(depth: usize) -> String {
        "\t".repeat(depth)
    }

    fn big_src(b: &Big, depth: usize) -> String {
        match b {
            Big::New(k) => format!("big({k})"),
            Big::Var(v) => v.clone(),
            Big::Join(a, b) => format!("join2({}, {})", big_src(a, depth), big_src(b, depth)),
            Big::Match(m) => match_src(m, depth),
            Big::Block { temp, inner, with } => format!(
                "{{\n{p1}{temp} = {}\n{p1}join2({temp}, {})\n{p0}}}",
                match_src(inner, depth + 1),
                big_src(with, depth + 1),
                p0 = pad(depth),
                p1 = pad(depth + 1),
            ),
        }
    }

    fn wrap_src(w: &Wrap, depth: usize) -> String {
        match w {
            Wrap::Var(v) => v.clone(),
            Wrap::One(b) => format!("one({})", big_src(b, depth)),
            Wrap::Two(a, b) => format!("two({}, {})", big_src(a, depth), big_src(b, depth)),
            Wrap::Zero => "zero()".into(),
        }
    }

    fn match_src(m: &Match, depth: usize) -> String {
        let bind = |n: &Option<String>| n.clone().unwrap_or_else(|| "_".into());
        let mut s = format!("match {} {{\n", wrap_src(&m.scrut, depth));
        for arm in &m.arms {
            let (pat, first) = match &arm.pat {
                Pat::One(x) => (format!("One({})", bind(x)), x.clone()),
                Pat::Two(x, y) => (format!("Two({}, {})", bind(x), bind(y)), x.clone()),
                Pat::Zero => ("Zero".into(), None),
            };
            let guard = match (arm.guard, first) {
                (Some(n), Some(x)) => format!(" if k_of({x}) > {n}"),
                _ => String::new(),
            };
            s.push_str(&format!(
                "{}{pat}{guard} -> {}\n",
                pad(depth + 1),
                big_src(&arm.body, depth + 1)
            ));
        }
        s.push_str(&format!("{}}}", pad(depth)));
        s
    }

    /// The oracle's heap: a cell per allocation, never freed, so identity is
    /// an index and reachability is what decides the count.
    enum Cell {
        Big(i64),
        One(usize),
        Two(usize, usize),
    }

    #[derive(Clone, Copy)]
    enum Val {
        Big(usize),
        /// `None` is `Zero`, a value word.
        Wrap(Option<usize>),
    }

    #[derive(Default)]
    struct Oracle {
        cells: Vec<Cell>,
    }

    impl Oracle {
        fn alloc(&mut self, c: Cell) -> usize {
            self.cells.push(c);
            self.cells.len() - 1
        }

        fn k(&self, big: usize) -> i64 {
            match self.cells[big] {
                Cell::Big(k) => k,
                _ => unreachable!("not a Big"),
            }
        }

        fn big(&mut self, b: &Big, env: &HashMap<String, Val>) -> usize {
            match b {
                Big::New(k) => self.alloc(Cell::Big(*k)),
                Big::Var(v) => match env[v] {
                    Val::Big(c) => c,
                    Val::Wrap(_) => unreachable!("{v} is a Wrap"),
                },
                Big::Join(a, b) => {
                    let (a, b) = (self.big(a, env), self.big(b, env));
                    let k = self.k(a) + self.k(b);
                    self.alloc(Cell::Big(k))
                }
                Big::Match(m) => self.matches(m, env),
                Big::Block { inner, with, .. } => {
                    let t = self.matches(inner, env);
                    let w = self.big(with, env);
                    let k = self.k(t) + self.k(w);
                    self.alloc(Cell::Big(k))
                }
            }
        }

        fn wrap(&mut self, w: &Wrap, env: &HashMap<String, Val>) -> Option<usize> {
            match w {
                Wrap::Var(v) => match env[v] {
                    Val::Wrap(c) => c,
                    Val::Big(_) => unreachable!("{v} is a Big"),
                },
                Wrap::One(b) => {
                    let b = self.big(b, env);
                    Some(self.alloc(Cell::One(b)))
                }
                Wrap::Two(a, b) => {
                    let (a, b) = (self.big(a, env), self.big(b, env));
                    Some(self.alloc(Cell::Two(a, b)))
                }
                Wrap::Zero => None,
            }
        }

        fn matches(&mut self, m: &Match, env: &HashMap<String, Val>) -> usize {
            let scrut = self.wrap(&m.scrut, env);
            let fields: Vec<usize> = match scrut.map(|c| &self.cells[c]) {
                Some(Cell::One(x)) => vec![*x],
                Some(Cell::Two(x, y)) => vec![*x, *y],
                Some(Cell::Big(_)) => unreachable!("a Big scrutinee"),
                None => vec![],
            };
            for arm in &m.arms {
                let names = match &arm.pat {
                    Pat::One(x) if fields.len() == 1 => vec![x],
                    Pat::Two(x, y) if fields.len() == 2 => vec![x, y],
                    Pat::Zero if fields.is_empty() => vec![],
                    _ => continue,
                };
                if let Some(n) = arm.guard
                    && self.k(fields[0]) <= n
                {
                    continue;
                }
                let mut env = env.clone();
                for (name, &cell) in names.iter().zip(&fields) {
                    if let Some(name) = name {
                        env.insert(name.clone(), Val::Big(cell));
                    }
                }
                return self.big(&arm.body, &env);
            }
            unreachable!("the arms are exhaustive")
        }

        /// Cells reachable from `roots`.
        fn reachable(&self, roots: impl Iterator<Item = Val>) -> usize {
            let mut seen = BTreeSet::new();
            let mut stack: Vec<usize> = roots
                .filter_map(|v| match v {
                    Val::Big(c) => Some(c),
                    Val::Wrap(c) => c,
                })
                .collect();
            while let Some(c) = stack.pop() {
                if seen.insert(c) {
                    match self.cells[c] {
                        Cell::Big(_) => {}
                        Cell::One(x) => stack.push(x),
                        Cell::Two(x, y) => stack.extend([x, y]),
                    }
                }
            }
            seen.len()
        }
    }

    /// One program and what it must print.
    fn program(seed: u64) -> (String, String) {
        let mut g = Gen {
            rng: Rng(seed | 1),
            fresh: 0,
        };
        let mut scope: Vec<(String, Ty)> = Vec::new();
        let mut stmts: Vec<(String, Rhs)> = Vec::new();
        for _ in 0..3 + g.rng.below(5) {
            let rhs = match g.rng.below(6) {
                0 => Rhs::Wrap(g.wrap(&scope)),
                1 => Rhs::Big(g.big(&scope, 0)),
                _ => Rhs::Big(Big::Match(Box::new(g.matches(&scope, 2)))),
            };
            let (v, ty) = match rhs {
                Rhs::Big(_) => (g.name("v"), Ty::Big),
                Rhs::Wrap(_) => (g.name("w"), Ty::Wrap),
            };
            scope.push((v.clone(), ty));
            stmts.push((v, rhs));
        }
        let reads: Vec<BTreeSet<String>> = stmts
            .iter()
            .map(|(_, rhs)| {
                let mut r = BTreeSet::new();
                match rhs {
                    Rhs::Big(b) => big_reads(b, &mut r),
                    Rhs::Wrap(w) => wrap_reads(w, &mut r),
                }
                r
            })
            .collect();
        // Every variable is read at least once, or the program does not
        // compile: what no statement reads, the last line does.
        let read_somewhere: BTreeSet<&String> = reads.iter().flatten().collect();
        let last: Vec<&(String, Ty)> = scope
            .iter()
            .filter(|(v, _)| !read_somewhere.contains(v))
            .collect();

        let mut src = format!("{HEADER}pub fn main() {{\n\tbase = internal.cells_live()\n");
        let mut want = String::new();
        let mut oracle = Oracle::default();
        let mut env: HashMap<String, Val> = HashMap::new();
        for (i, (v, rhs)) in stmts.iter().enumerate() {
            let (text, val) = match rhs {
                Rhs::Big(b) => (big_src(b, 1), Val::Big(oracle.big(b, &env))),
                Rhs::Wrap(w) => (wrap_src(w, 1), Val::Wrap(oracle.wrap(w, &env))),
            };
            src.push_str(&format!(
                "\t{v} = {text}\n\tprintln(internal.cells_live() - base)\n"
            ));
            env.insert(v.clone(), val);
            let live = env.iter().filter(|(name, _)| {
                reads[i + 1..].iter().any(|r| r.contains(*name))
                    || last.iter().any(|(l, _)| l == *name)
            });
            want.push_str(&format!("{}\n", oracle.reachable(live.map(|(_, v)| *v))));
        }
        let mut total = 0;
        let mut terms = Vec::new();
        for (v, ty) in &last {
            match (ty, env[v]) {
                (Ty::Big, Val::Big(c)) => {
                    terms.push(format!("k_of({v})"));
                    total += oracle.k(c);
                }
                (Ty::Wrap, Val::Wrap(c)) => {
                    terms.push(format!("w_of({v})"));
                    total += match c.map(|c| &oracle.cells[c]) {
                        Some(Cell::One(x)) => oracle.k(*x),
                        Some(Cell::Two(x, y)) => oracle.k(*x) + oracle.k(*y),
                        _ => 0,
                    };
                }
                _ => unreachable!("a variable of the wrong type"),
            }
        }
        terms.push("0".into());
        src.push_str(&format!(
            "\tprintln({})\n\tprintln(internal.cells_live() - base)\n}}\n",
            terms.join(" + ")
        ));
        want.push_str(&format!("{total}\n0\n"));
        (src, want)
    }

    #[test]
    fn the_cells_held_after_each_statement_are_the_ones_still_read() {
        const SEED: u64 = 0x5eed_f00d;
        for i in 0..300 {
            let (src, want) = program(SEED.wrapping_add(i).wrapping_mul(0x9e37_79b9_7f4a_7c15));
            assert_eq!(
                super::run(&src).as_deref(),
                Ok(want.as_str()),
                "program #{i}:\n{src}"
            );
        }
    }
}
