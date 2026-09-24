//! `scarlet dis` and the REPL's `:dis`: a compiled program's Core IR as text.
//!
//! The stdlib compiles into the same [`Program`] as the user's code, so a
//! listing always picks: the entry file's own functions, or every function
//! whose name matches, from any module.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use crate::core_ir::{Const, ConstId, CoreExpr, Program};
use crate::module::ModuleKey;
use crate::tivec::Idx as _;

/// Which functions a listing shows.
#[derive(Clone, Copy)]
pub enum Filter<'a> {
    /// The entry file's functions, then its toplevel.
    Entry,
    /// Every function, from any module, whose name contains this.
    Named(&'a str),
}

/// The listing, or `None` when `filter` matches no function.
///
/// Each function is preceded by its `fn#N`, the number a `call fn#N` or
/// `closure fn#N` elsewhere in the listing refers to. The listing ends with
/// the values of the constants it loads, so `c3` can be read, and the names
/// of the constructors it shows, so `ctor 512.0(%1)` can be.
pub fn listing(program: &Program, filter: Filter<'_>) -> Option<String> {
    let entry = ModuleKey::main();
    let mut out = summary(program, &entry);
    let mut shown: Vec<&CoreExpr> = Vec::new();
    for (i, f) in (&program.fns).into_iter().enumerate() {
        let keep = match filter {
            Filter::Entry => f.module == entry.as_str(),
            Filter::Named(needle) => f.name.contains(needle),
        };
        if keep {
            let _ = write!(out, "\n; fn#{i}\n{f}");
            shown.push(&f.core.body);
        }
    }
    if let Filter::Entry = filter {
        let _ = write!(out, "\n; toplevel\n{}", program.toplevel);
        shown.push(&program.toplevel.core.body);
    }
    if shown.is_empty() {
        return None;
    }
    consts(program, &shown, &mut out);
    types(program, &shown, &mut out);
    Some(out)
}

/// How much of a long `String` or `Binary` constant a listing shows. An
/// `@embed` const is a whole file, and a listing is not the place to read one.
const SHOWN_CHARS: usize = 48;
const SHOWN_BYTES: usize = 16;

/// One comment line per constant `bodies` load: `; c3 'hello'`. A long
/// string or binary is cut short, with its whole length.
fn consts(program: &Program, bodies: &[&CoreExpr], out: &mut String) {
    let mut ids = BTreeSet::new();
    for body in bodies {
        body.for_each_const(|c: ConstId| {
            ids.insert(c);
        });
    }
    if ids.is_empty() {
        return;
    }
    out.push_str("\n; consts\n");
    for id in ids {
        let _ = write!(out, "; {id} ");
        match program.consts.get(id.index()) {
            Some(Const::Int(n)) => {
                let _ = writeln!(out, "{n}");
            }
            Some(Const::Float(x)) => {
                let _ = writeln!(out, "{x:?}");
            }
            Some(Const::String(text)) => {
                let shown: String = text.chars().take(SHOWN_CHARS).collect();
                let _ = write!(out, "'{}'", shown.escape_debug());
                if shown.len() < text.len() {
                    let _ = write!(out, "... ({} bytes)", text.len());
                }
                out.push('\n');
            }
            Some(Const::Binary { bytes, bit_len }) => {
                let shown: Vec<String> = bytes
                    .iter()
                    .take(SHOWN_BYTES)
                    .map(|b| b.to_string())
                    .collect();
                let more = if bytes.len() > SHOWN_BYTES {
                    ", ..."
                } else {
                    ""
                };
                let _ = write!(out, "<<{}{more}>>", shown.join(", "));
                if bytes.len() > SHOWN_BYTES || bit_len % 8 != 0 {
                    let _ = write!(out, " ({bit_len} bits)");
                }
                out.push('\n');
            }
            None => {
                let _ = writeln!(out, "is not in the program");
            }
        }
    }
}

/// One comment line per type a constructor in `bodies` belongs to:
/// `; 512 Option: .0 Some(value), .1 None`.
fn types(program: &Program, bodies: &[&CoreExpr], out: &mut String) {
    let mut ids = BTreeSet::new();
    for body in bodies {
        body.for_each_variant(|v| {
            ids.insert(v.type_id);
        });
    }
    if ids.is_empty() {
        return;
    }
    out.push_str("\n; types\n");
    for id in ids {
        let Some(t) = program.types.get(&id) else {
            let _ = writeln!(out, "; {} has no names", id.0);
            continue;
        };
        let _ = write!(out, "; {} {}:", id.0, t.name);
        for (i, v) in t.variants.iter().enumerate() {
            let sep = if i == 0 { " " } else { ", " };
            let _ = write!(out, "{sep}.{i} {}", v.name);
            if !v.fields.is_empty() {
                let _ = write!(out, "({})", v.fields.join(", "));
            }
        }
        out.push('\n');
    }
}

/// One comment line on the program's shape, so a filtered listing still says
/// what it was taken from.
fn summary(program: &Program, entry: &ModuleKey) -> String {
    let total = (&program.fns).into_iter().count();
    let own = (&program.fns)
        .into_iter()
        .filter(|f| f.module == entry.as_str())
        .count();
    let start = match program.main {
        Some(main) => format!("starts at {main} ({})", program.fns[main].name),
        None => "runs its toplevel".to_string(),
    };
    format!(
        "; {total} functions, {own} from this file; {} module inits; {} globals; {start}\n",
        program.inits.len(),
        program.globals,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn program(src: &str) -> Program {
        let mut scanner = crate::scanner::new_scanner(src.to_string());
        let parsed = crate::parser::new_parser(&mut scanner).parse_program();
        let result =
            crate::bytecode::compile(&crate::ast::Expression::BlockExpression(parsed.ast), None);
        assert!(result.success(), "{:?}", result.diagnostics);
        result.into_runnable().expect("a clean compile is runnable")
    }

    const SQUARE: &str = "fn square(x Int) Int { x * x }\n\
                          pub fn main() {\n\
                          \tprintln(square(3))\n\
                          }\n";

    #[test]
    fn the_entry_listing_is_this_files_functions_and_its_toplevel() {
        let text = listing(&program(SQUARE), Filter::Entry).expect("main has functions");
        assert!(text.contains("fn main.square(%0:"), "{text}");
        assert!(text.contains("IntMul(%0, %0)"), "{text}");
        assert!(text.contains("\n; toplevel\nfn main.__main__("), "{text}");
        assert!(
            !text.contains("fn scarlet."),
            "listed a stdlib function:\n{text}"
        );
    }

    /// A listing names each constant it loads, and cuts a long one short: an
    /// `@embed` const is a whole file.
    #[test]
    fn a_listing_shows_its_constants_and_cuts_long_ones_short() {
        let long = "x".repeat(200);
        let src = format!(
            "pub fn main() {{\n\tprintln('hi')\n\tprintln('{long}')\n\tprintln(<<1, 2>>)\n}}\n"
        );
        let text = listing(&program(&src), Filter::Entry).expect("main exists");
        assert!(text.contains("\n; consts\n"), "{text}");
        assert!(text.contains(" 'hi'\n"), "{text}");
        let cut = format!(" '{}'... (200 bytes)\n", "x".repeat(48));
        assert!(text.contains(&cut), "{text}");
        assert!(!text.contains(&"x".repeat(49)), "{text}");
    }

    #[test]
    fn a_named_listing_reaches_into_the_stdlib() {
        let src = "import scarlet/string\n\
                   pub fn main() {\n\
                   \tprintln(string.replace('a', 'a', 'b'))\n\
                   }\n";
        let text = listing(&program(src), Filter::Named("replace")).expect("replace exists");
        assert!(text.contains("fn scarlet/string.replace("), "{text}");
        assert!(!text.contains("; toplevel"), "{text}");
    }

    /// A listing ends with the names of the constructors it shows, and only
    /// those.
    #[test]
    fn the_listing_names_its_constructors() {
        let src = "type Shape {\n\
                   \tCircle(radius Int)\n\
                   \tDot\n\
                   }\n\
                   fn area(s Shape) Int {\n\
                   \tmatch s {\n\
                   \t\tCircle(r) -> r * r\n\
                   \t\tDot -> 0\n\
                   \t}\n\
                   }\n\
                   pub fn main() {\n\
                   \tprintln(area(Circle(radius: 2)))\n\
                   }\n";
        let text = listing(&program(src), Filter::Entry).expect("main has functions");
        let (_, types) = text.split_once("\n; types\n").expect("a types section");
        let lines: Vec<&str> = types.lines().collect();
        assert_eq!(lines.len(), 1, "{types}");
        assert!(
            lines[0].ends_with(" Shape: .0 Circle(radius), .1 Dot"),
            "{types}"
        );
    }

    #[test]
    fn a_name_nothing_has_lists_nothing() {
        assert!(listing(&program(SQUARE), Filter::Named("nope")).is_none());
    }

    /// The header counts what the listing was taken from, so a filtered
    /// listing still says how big the program is and where it starts.
    #[test]
    fn the_summary_names_where_the_program_starts() {
        let text = listing(&program(SQUARE), Filter::Named("square")).expect("square exists");
        let first = text.lines().next().unwrap_or_default();
        assert!(first.starts_with("; "), "{first}");
        assert!(first.contains("2 from this file"), "{first}");
        assert!(first.ends_with("(main)"), "{first}");
    }
}
