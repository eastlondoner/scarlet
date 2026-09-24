//! `@embed('path') const x String`: a file read into a constant at compile
//! time. Every refusal has its own message; the file is found relative to the
//! module that names it and known by its resolved path; and it is a
//! dependency of that module, so an edit to it recompiles the module and only
//! what depends on it.

use std::io::Write as _;
use std::process::{Command, Stdio};

use scarlet::bytecode::{self, IncrementalSession};
use scarlet::core_ir::Const;

mod common;
use common::{Project, parse, recheck, run_al};

/// `al check` the project's `main.scrl`, asserting it fails with exactly one
/// error whose message is `msg`.
#[track_caller]
fn refused(p: &Project, main: &str, msg: &str) {
    p.write("main.scrl", main);
    let out = run_al("check", &p.dir.join("main.scrl"));
    assert!(!out.success, "expected a refusal:\n{}", out.combined());
    assert!(!out.combined().contains("panicked"), "{}", out.combined());
    let errors: Vec<&str> = out
        .stderr
        .lines()
        .filter_map(|l| l.strip_prefix("error: "))
        .collect();
    assert_eq!(errors.len(), 1, "want one error, got:\n{}", out.stderr);
    assert!(
        errors[0].starts_with(msg),
        "want `{msg}`, got:\n{}",
        out.stderr
    );
}

/// A program printing each embedded const, around whatever `decls` declare.
fn program(decls: &str, body: &str) -> String {
    format!("{decls}\npub fn main() {{\n{body}}}\n")
}

#[test]
fn a_const_takes_a_text_file_as_a_string_and_any_file_as_a_binary() {
    let p = Project::new("embed_run");
    p.write("greeting.txt", "hello\r\n\u{feff}world");
    std::fs::write(p.dir.join("data.bin"), [0u8, 1, 254, 255]).unwrap();
    p.write(
        "main.scrl",
        &program(
            "import scarlet/binary\nimport scarlet/string\n\n\
             @embed('greeting.txt')\nconst greeting String\n\n\
             @embed('./data.bin')\npub const data Binary\n",
            "\tprintln(string.length(greeting))\n\tprintln(data)\n\tprintln(binary.byte_size(data))\n",
        ),
    );
    // Run from elsewhere: the path is the module's, not the process's.
    let out = run_al("run", &p.dir.join("main.scrl"));
    assert!(out.success, "{}", out.combined());
    // No line-ending conversion, and the byte-order mark stays.
    assert_eq!(out.stdout, "13\n<<0, 1, 254, 255>>\n4\n");
}

/// A module in another directory embeds relative to its own file, not the
/// entry's: `lib/text.txt`, not `./text.txt`.
#[test]
fn a_sub_module_embeds_relative_to_its_own_directory() {
    let p = Project::new("embed_sub");
    std::fs::create_dir_all(p.dir.join("lib")).unwrap();
    p.write("text.txt", "the entry's");
    p.write("lib/text.txt", "the module's");
    p.write("lib/m.scrl", "@embed('text.txt')\npub const text String\n");
    p.write(
        "main.scrl",
        &program(
            "import ./lib/m\n\n@embed('text.txt')\nconst mine String\n",
            "\tprintln(m.text)\n\tprintln(mine)\n",
        ),
    );
    let out = run_al("run", &p.dir.join("main.scrl"));
    assert!(out.success, "{}", out.combined());
    assert_eq!(out.stdout, "the module's\nthe entry's\n");
}

#[test]
fn every_refusal_says_what_is_wrong() {
    let p = Project::new("embed_errors");
    p.write("a.txt", "text");
    std::fs::write(p.dir.join("bad.txt"), b"ok\xc3(").unwrap();
    let main = |decls: &str| program(decls, "\tNil\n");
    let cases: &[(&str, &str)] = &[
        (
            "@embed('a.txt')\nconst a\n",
            "An @embed const must declare its type: `String` for text or `Binary` for bytes",
        ),
        (
            "@embed('a.txt')\nconst a Int\n",
            "An @embed const is a `String` or a `Binary`, not `Int`",
        ),
        (
            "@embed('a.txt')\nconst a String = 'x'\n",
            "An @embed const takes its value from the file, so it cannot also have `= value`",
        ),
        (
            "@embed(a)\nconst a String\n",
            "@embed takes exactly one argument: the file's path as a string, e.g. @embed('shaders/world.metal')",
        ),
        (
            "@embed('a.txt', 'b.txt')\nconst a String\n",
            "@embed takes exactly one argument: the file's path as a string, e.g. @embed('shaders/world.metal')",
        ),
        (
            "@embed('a.txt')\n@embed('a.txt')\nconst a String\n",
            "`@embed` may appear only once on a const",
        ),
        (
            "@embed('${1}.txt')\nconst a String\n",
            "Attribute arguments are plain strings: interpolation is not allowed here",
        ),
        (
            "@embed('a.txt')\nfn f() String {\n\t'x'\n}\n",
            "'@embed' may only be used on `const` declarations",
        ),
        (
            "@embed('a.txt')\ntype T {\n\tT\n}\n",
            "'@embed' may only be used on `const` declarations",
        ),
        (
            "@exhaustive\nconst a = 1\n",
            "'@exhaustive' may only be used on types",
        ),
        (
            "@embed('/etc/hosts')\nconst a String\n",
            "@embed takes a path relative to this module's directory, not an absolute one: '/etc/hosts'",
        ),
        (
            "@embed('')\nconst a String\n",
            "@embed needs a file's path, not an empty string",
        ),
        (
            "@embed('nope.txt')\nconst a String\n",
            "Cannot embed 'nope.txt': cannot read ",
        ),
        (
            "@embed('bad.txt')\nconst a String\n",
            "Cannot embed 'bad.txt' as a `String`: it is not UTF-8 (the first invalid byte is at offset 2); embed it as a `Binary` instead",
        ),
    ];
    for (decls, msg) in cases {
        refused(&p, &main(decls), msg);
    }
    // A const is top-level only, `@embed` or not.
    refused(
        &p,
        "pub fn main() {\n\t@embed('a.txt')\n\tconst a String\n\tNil\n}\n",
        "const declarations are only allowed at the top level",
    );
    // The same bytes are a fine `Binary`.
    p.write(
        "main.scrl",
        &program("@embed('bad.txt')\nconst a Binary\n", "\tprintln(a)\n"),
    );
    let out = run_al("run", &p.dir.join("main.scrl"));
    assert!(out.success, "{}", out.combined());
    assert_eq!(out.stdout, "<<111, 107, 195, 40>>\n");
}

/// The missing file's message names where it looked: the resolved path.
#[test]
fn a_missing_file_is_named_by_where_it_was_looked_for() {
    let p = Project::new("embed_missing");
    std::fs::create_dir_all(p.dir.join("sub")).unwrap();
    p.write(
        "main.scrl",
        &program("@embed('sub/../x.txt')\nconst a String\n", "\tNil\n"),
    );
    let out = run_al("check", &p.dir.join("main.scrl"));
    let want = format!(
        "Cannot embed 'sub/../x.txt': cannot read {}: ",
        p.dir.join("x.txt").display()
    );
    assert!(
        out.stderr.contains(&want),
        "want `{want}` in:\n{}",
        out.stderr
    );
}

/// A source with no directory (an unsaved buffer) has nothing to resolve a
/// path against, and the stdlib is compiled into Scarlet: neither may embed.
/// `@vm` on a const is checked here too, where `@vm` is otherwise allowed, so
/// its message is the only one.
#[test]
fn a_source_with_no_directory_cannot_embed_and_vm_is_for_functions() {
    let src = program("@embed('a.txt')\nconst a String\n", "\tNil\n");
    let r = bytecode::check(&parse(&src), None);
    let msgs: Vec<&str> = r.diagnostics.iter().map(|d| d.message.as_str()).collect();
    assert_eq!(
        msgs,
        ["@embed needs a file on disk: this source has no directory to find 'a.txt' in"]
    );

    let p = Project::new("embed_std");
    p.write("a.txt", "text");
    let r = bytecode::check_as_module(
        &parse("@embed('a.txt')\npub const a String\n"),
        Some(&p.dir),
        vec!["scarlet".to_string(), "embedded".to_string()],
    );
    let msgs: Vec<&str> = r.diagnostics.iter().map(|d| d.message.as_str()).collect();
    assert_eq!(
        msgs,
        [
            "'@embed' is not allowed in the standard library, which is compiled into Scarlet and has no directory to read a file from"
        ]
    );

    // `@vm` is the stdlib's own attribute, and still only for functions.
    let r = bytecode::check_as_module(
        &parse("@vm(add)\npub const a = 1\n"),
        Some(&p.dir),
        vec!["scarlet".to_string(), "embedded".to_string()],
    );
    let msgs: Vec<&str> = r.diagnostics.iter().map(|d| d.message.as_str()).collect();
    assert_eq!(msgs, ["'@vm' may only be used on functions"]);
}

/// How many `String` constants in the program hold exactly `text`.
fn string_consts(program: &scarlet::core_ir::Program, text: &str) -> usize {
    program
        .consts
        .iter()
        .filter(|c| matches!(c, Const::String(s) if s == text))
        .count()
}

/// Two modules naming one file by different relative paths read one file:
/// one dependency, one constant.
#[test]
fn two_spellings_of_one_path_are_one_file() {
    let p = Project::new("embed_one");
    std::fs::create_dir_all(p.dir.join("sub")).unwrap();
    std::fs::create_dir_all(p.dir.join("data")).unwrap();
    p.write("data/x.txt", "shared text");
    p.write(
        "sub/m.scrl",
        "@embed('../data/x.txt')\npub const x String\n",
    );
    let main = program(
        "import ./sub/m\n\n@embed('data/./x.txt')\nconst y String\n",
        "\tprintln(m.x)\n\tprintln(y)\n",
    );
    p.write("main.scrl", &main);

    let mut s = IncrementalSession::new();
    let r = s.check(&parse(&main), Some(&p.dir));
    assert!(r.success(), "{:?}", r.diagnostics);
    let files: Vec<_> = s.embedded_files().into_iter().collect();
    assert_eq!(files, [p.dir.join("data/x.txt")]);

    let r = bytecode::compile(&parse(&main), Some(&p.dir));
    assert!(r.success(), "{:?}", r.diagnostics);
    let program = r.into_runnable().expect("clean");
    assert_eq!(string_consts(&program, "shared text"), 1);
}

/// One spelling in two directories names two files.
#[test]
fn one_spelling_in_two_directories_is_two_files() {
    let p = Project::new("embed_two");
    for d in ["a", "b"] {
        std::fs::create_dir_all(p.dir.join(d)).unwrap();
        p.write(&format!("{d}/x.txt"), &format!("from {d}"));
        p.write(
            &format!("{d}/m.scrl"),
            "@embed('x.txt')\npub const x String\n",
        );
    }
    let main = program(
        "import ./a/m as a\nimport ./b/m as b\n",
        "\tprintln(a.x)\n\tprintln(b.x)\n",
    );
    p.write("main.scrl", &main);

    let mut s = IncrementalSession::new();
    let r = s.check(&parse(&main), Some(&p.dir));
    assert!(r.success(), "{:?}", r.diagnostics);
    let files: Vec<_> = s.embedded_files().into_iter().collect();
    assert_eq!(files, [p.dir.join("a/x.txt"), p.dir.join("b/x.txt")]);

    let out = run_al("run", &p.dir.join("main.scrl"));
    assert!(out.success, "{}", out.combined());
    assert_eq!(out.stdout, "from a\nfrom b\n");
}

const OTHER: &str = "pub fn other() Int {\n\t1\n}\n";
const MID: &str = "import ./sub/m\n\npub fn text() String {\n\tm.text\n}\n";
const EMBEDDER: &str = "@embed('s.txt')\npub const text String\n";
const ENTRY: &str = "import ./other\nimport ./mid\n\npub fn main() {\n\tprintln(other.other())\n\tprintln(mid.text())\n}\n";

/// `other` compiles first and depends on nothing embedded; `mid` imports
/// `sub/m`, which embeds `sub/s.txt`.
fn embedding_project(tag: &str) -> (Project, IncrementalSession) {
    let p = Project::new(tag);
    std::fs::create_dir_all(p.dir.join("sub")).unwrap();
    p.write("other.scrl", OTHER);
    p.write("mid.scrl", MID);
    p.write("sub/m.scrl", EMBEDDER);
    p.write("sub/s.txt", "one");
    p.write("main.scrl", ENTRY);
    let mut s = IncrementalSession::new();
    let r = s.check(&parse(ENTRY), Some(&p.dir));
    assert!(r.success(), "{:?}", r.diagnostics);
    assert_eq!(s.compile_count(), 3, "other, mid and m compile once");
    (p, s)
}

/// Editing an embedded file recompiles the module that embeds it and what
/// depends on it, and nothing else; the next check sees the new contents.
#[test]
fn editing_an_embedded_file_recompiles_its_module_and_no_other() {
    let (p, mut s) = embedding_project("embed_edit");

    let r = recheck(&mut s, &p, ENTRY);
    assert!(r.success(), "{:?}", r.diagnostics);
    assert_eq!(s.compile_count(), 3, "nothing changed, nothing recompiles");

    // A different length, so the stat gate cannot mistake it for the old one.
    std::fs::write(p.dir.join("sub/s.txt"), b"not \xff utf-8").unwrap();
    let r = recheck(&mut s, &p, ENTRY);
    assert!(!r.success(), "the edited file is re-read and refused");
    assert!(
        r.diagnostics.iter().any(|d| d.message.contains(
            "Cannot embed 's.txt' as a `String`: it is not UTF-8 (the first invalid byte is at offset 4)"
        )),
        "{:?}",
        r.diagnostics
    );
    assert_eq!(s.compile_count(), 5, "m and its dependent mid; not other");

    p.write("sub/s.txt", "two");
    let r = recheck(&mut s, &p, ENTRY);
    assert!(r.success(), "{:?}", r.diagnostics);
    assert_eq!(s.compile_count(), 7);
}

/// `invalidate_path` on an embedded file (the LSP's watched-file change)
/// evicts the embedding module and its dependents even when the stat gate
/// would have kept them, and leaves an unrelated module cached.
#[test]
fn invalidating_an_embedded_file_evicts_the_module_that_read_it() {
    let (p, mut s) = embedding_project("embed_invalidate");
    s.invalidate_path(&p.dir.join("sub/s.txt"));
    let r = recheck(&mut s, &p, ENTRY);
    assert!(r.success(), "{:?}", r.diagnostics);
    assert_eq!(s.compile_count(), 5, "m and mid again; other stays cached");

    // By another spelling of the same file.
    s.invalidate_path(&p.dir.join("sub/../sub/s.txt"));
    let r = recheck(&mut s, &p, ENTRY);
    assert!(r.success(), "{:?}", r.diagnostics);
    assert_eq!(s.compile_count(), 7);

    // A file nothing embedded evicts nothing.
    s.invalidate_path(&p.dir.join("unrelated.txt"));
    let r = recheck(&mut s, &p, ENTRY);
    assert!(r.success(), "{:?}", r.diagnostics);
    assert_eq!(s.compile_count(), 7);
}

/// A file that is missing when the module compiles is a dependency too: when
/// it appears, the module recompiles and the error goes.
#[test]
fn a_missing_file_that_appears_recompiles_its_module() {
    let (p, mut s) = embedding_project("embed_appear");
    std::fs::remove_file(p.dir.join("sub/s.txt")).unwrap();
    let r = recheck(&mut s, &p, ENTRY);
    assert!(!r.success(), "a deleted embedded file is an error");
    p.write("sub/s.txt", "back");
    let r = recheck(&mut s, &p, ENTRY);
    assert!(r.success(), "{:?}", r.diagnostics);
}

/// The REPL compiles each entry by replaying every earlier definition ahead of
/// it, so an `@embed` const is read again at every entry: an edited file shows
/// in the next one. Here the session edits the file itself, between entries.
#[test]
fn the_repl_reads_an_edited_file_at_the_next_entry() {
    let p = Project::new("embed_repl");
    p.write("note.txt", "first");
    let entries = "import scarlet/io\n\
                   @embed('note.txt') const note String\n\
                   println(note)\n\
                   io.write_text('note.txt', 'second')\n\
                   println(note)\n";
    let mut child = Command::new(env!("CARGO_BIN_EXE_scarlet"))
        .arg("repl")
        .current_dir(&p.dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn scarlet repl");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(entries.as_bytes())
        .expect("write stdin");
    let out = common::wait_or_kill(child, common::CHILD_TIMEOUT_SECS);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let first = stdout
        .find("first")
        .unwrap_or_else(|| panic!("{stdout}\n{stderr}"));
    let second = stdout
        .find("second")
        .unwrap_or_else(|| panic!("{stdout}\n{stderr}"));
    assert!(first < second, "{stdout}");
}

/// A const is made once, at the module's toplevel, and read as a global
/// after that: reading an embedded `pub const` from a function allocates
/// nothing, however often. That is what the declaration form is for. The same
/// text written as a literal in the function is made on every call, which is
/// what shows this test can fail.
#[test]
fn reading_an_embedded_const_from_a_function_allocates_nothing() {
    let p = Project::new("embed_cells");
    std::fs::create_dir_all(p.dir.join("lib")).unwrap();
    p.write("lib/shader.metal", "kernel void k() {}\n");
    p.write(
        "lib/shaders.scrl",
        "@embed('shader.metal')\npub const source String\n",
    );
    p.write(
        "main.scrl",
        "import scarlet/internal\n\
         import scarlet/string\n\
         import ./lib/shaders\n\n\
         fn embedded(n Int, acc Int) Int {\n\
         \tif n <= 0 then acc else embedded(n - 1, acc + string.length(shaders.source))\n\
         }\n\n\
         fn literal(n Int, acc Int) Int {\n\
         \tif n <= 0 then acc else literal(n - 1, acc + string.length('kernel void k() {}\\n'))\n\
         }\n\n\
         pub fn main() {\n\
         \tmade = internal.cells_made()\n\
         \tlength = embedded(100, 0)\n\
         \tafter_embedded = internal.cells_made()\n\
         \tsame = literal(100, 0)\n\
         \tafter_literal = internal.cells_made()\n\
         \tprintln(length == same)\n\
         \tprintln(after_embedded - made)\n\
         \tprintln(after_literal - after_embedded > 0)\n\
         }\n",
    );
    let out = run_al("run", &p.dir.join("main.scrl"));
    assert!(out.success, "{}", out.combined());
    assert_eq!(out.stdout, "True\n0\nTrue\n");
}

/// An embedded file is watched: the LSP workspace reports every file the
/// documents it analysed embedded, for the client to watch.
#[test]
fn the_editor_is_told_which_files_to_watch() {
    let p = Project::new("embed_lsp");
    std::fs::create_dir_all(p.dir.join("sub")).unwrap();
    p.write("sub/s.txt", "x");
    p.write("sub/m.scrl", EMBEDDER);
    p.write("own.txt", "y");
    let src = "import ./sub/m\n\n@embed('own.txt')\nconst own String\n\npub fn main() {\n\tprintln(m.text)\n\tprintln(own)\n}\n";
    p.write("a.scrl", src);
    let mut ws = scarlet::lsp::Workspace::new();
    ws.add_workspace_root(p.dir.clone());
    let uri = format!("file://{}", p.dir.join("a.scrl").display());
    let diags = ws.open_document(&uri, src);
    assert!(diags.is_empty(), "{diags:?}");
    let files: Vec<_> = ws.embedded_files().into_iter().collect();
    assert_eq!(files, [p.dir.join("own.txt"), p.dir.join("sub/s.txt")]);
}
