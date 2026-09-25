//! Perceus reference-count insertion on Core IR: the frame-limited
//! drop-guided algorithm of Lorenzen & Leijen (ICFP'22, Fig. 5) as a Core→Core
//! pass. ANF makes last-use a linear backward scan.
//!
//!  1. Backward liveness inserts `Drop` after each heap bind's last read.
//!     `If`/`Match` joins equalise ownership so every path releases every owned
//!     slot exactly once (Fig. 5's Δ-rule). A `LetJoin` is walked through, not
//!     around: its `Tail`s continue into the code after it, so what that code
//!     reads is live at each of them, and each branch drops the rest itself.
//!  2. A forward walk pairs each `Drop` token with a later same-shape `Ctor`
//!     through a LIFO stack, forked at branches and met again after a join.
//!  3. The reuse walk runs a second time seeded with the intersection of the
//!     first run's back-edge token sets, so a tail-self-recursive loop can
//!     reuse the previous iteration's cells.

use std::collections::{BTreeMap, BTreeSet};

use super::{Atom, Callee, CoreBind, CoreExpr, CoreFn, CorePat, JoinId, LocalId, ReuseShape};
use crate::typed_ir::{RTy, ResolvedPool};

/// Run Perceus on a single lowered function. `pool` is consulted only to
/// classify a bind's `ty` as heap-shaped and to read a tuple's width. Its types
/// are resolved by construction, so no unsolved variable can reach here and
/// make `is_heap` answer `false` by accident.
pub(crate) fn perceus(pool: &ResolvedPool, f: CoreFn) -> CoreFn {
    let mut cx = Perceus::new(pool, f.locals_end());
    let CoreFn {
        params,
        body,
        ret_ty,
    } = f;
    for p in &params {
        cx.record_bind(p, None);
    }
    let (body, live) = cx.drop_pass(body, &Exit::Return);
    // Params are owned on entry; any not read by the body drop at its head so
    // the frame's release count still balances.
    let dead_params: Vec<LocalId> = params
        .iter()
        .map(|p| p.id)
        .filter(|id| !live.contains(id))
        .collect();
    let body = cx.wrap_drops(&dead_params, None, body);
    CoreFn {
        params,
        body: reuse_pass(body),
        ret_ty,
    }
}

/// Locals live at a program point. `BTreeSet` so drop insertion order, and
/// thus golden snapshots, is deterministic.
type Live = BTreeSet<LocalId>;

/// A hollowed cell available for a downstream same-shape `Ctor` to overwrite.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Token {
    slot: LocalId,
    shape: ReuseShape,
    /// Loop-carried from a previous iteration via [`ReuseWalk::back_edge_seed`].
    /// Its slot may be bound inside an arbitrary branch, so the only site sure
    /// to have it in scope is the `Let` binding that same id. Carried tokens
    /// therefore pair by self-slot only, never the LIFO shape fallback.
    carried: bool,
}

/// Where a `Tail` in the expression being transformed sends its value.
enum Exit {
    /// Out of the function: a return, or a tail call. The frame's references
    /// go with it, so a `Tail` needs no drop after it.
    Return,
    /// Into a `LetJoin`'s bind, after which the code after the join runs.
    Join {
        /// The bind's type, for a local minted to hold the value.
        ty: RTy,
        /// Locals the code after the join reads. Every path through the join
        /// must still own them when it reaches a `Tail`.
        after: Live,
    },
}

struct Perceus<'p> {
    pool: &'p ResolvedPool,
    /// The next local id free to mint, above every one the function uses.
    next_local: u32,
    /// Every bind's resolved type, for the heap-shape gate on `Drop`.
    ty: BTreeMap<LocalId, RTy>,
    /// Allocation shape of a bind when its rhs proves one. Locals with no entry
    /// drop with `shape: None` and are never offered for reuse.
    shape: BTreeMap<LocalId, ReuseShape>,
    /// Live-in set of each `LetCont`'s continuation, recorded when the
    /// `LetCont` is peeled — before its body, which holds every `Goto` to it —
    /// so a `Goto` can always look its target up here.
    cont_live: BTreeMap<JoinId, Live>,
}

/// One node peeled off the spine by [`Perceus::drop_pass`], awaiting its
/// body's live set.
enum SpineFrame {
    /// A `Let`: the live set decides which drops go between rhs and body.
    Let {
        bind: CoreBind,
        rhs_live: Live,
        rhs: Atom,
    },
    /// A `LetJoin`, whose join is transformed only once the body's live set is
    /// known, since that is what each of its `Tail`s must leave owned.
    Join { bind: CoreBind, join: Box<CoreExpr> },
    /// A `LetCont` whose continuation was drop-processed at peel time. It is
    /// rebuilt around the body with no drops of its own: the continuation's
    /// live-ins reach the enclosing live set through the `Goto`s to it.
    Cont { id: JoinId, cont: CoreExpr },
}

impl<'p> Perceus<'p> {
    fn new(pool: &'p ResolvedPool, next_local: u32) -> Self {
        Perceus {
            pool,
            next_local,
            ty: BTreeMap::new(),
            shape: BTreeMap::new(),
            cont_live: BTreeMap::new(),
        }
    }

    fn record_bind(&mut self, b: &CoreBind, rhs_shape: Option<ReuseShape>) {
        self.ty.insert(b.id, b.ty);
        if let Some(s) = rhs_shape {
            self.shape.insert(b.id, s);
        }
    }

    /// Whether the local's type allocates a Perceus-managed heap cell. A local
    /// with no recorded type was never bound by this pass, so it owns nothing.
    fn is_heap_local(&self, id: LocalId) -> bool {
        self.ty.get(&id).is_some_and(|&t| self.pool.is_heap(t))
    }

    /// A fresh local of type `ty`, numbered above every one the function had.
    fn mint(&mut self, ty: RTy, shape: Option<ReuseShape>) -> CoreBind {
        let b = CoreBind::new(LocalId(self.next_local), ty);
        self.next_local += 1;
        self.record_bind(&b, shape);
        b
    }

    /// Transform `e`, returning `(e', live)` where `live` is the outer-scope
    /// locals `e'` reads and so the caller's responsibility. Drops are inserted
    /// for locals that go dead inside `e`. The spine is walked iteratively so a
    /// long straight-line body does not recurse on the Rust stack.
    fn drop_pass(&mut self, mut e: CoreExpr, exit: &Exit) -> (CoreExpr, Live) {
        let mut spine: Vec<SpineFrame> = Vec::new();
        let (mut body, mut live) = loop {
            match e {
                CoreExpr::Let { bind, rhs, body } => {
                    let rhs_live = atom_live(&rhs);
                    self.record_bind(&bind, ctor_shape(&rhs));
                    spine.push(SpineFrame::Let {
                        bind,
                        rhs_live,
                        rhs,
                    });
                    e = *body;
                }
                CoreExpr::LetJoin { bind, join, body } => {
                    self.record_bind(&bind, None);
                    spine.push(SpineFrame::Join { bind, join });
                    e = *body;
                }
                CoreExpr::LetCont { id, cont, body } => {
                    // Transformed before the body so its live-in set is known at
                    // every `Goto(id)` the body contains. A cont inside a join
                    // ends in that join's `Tail`s, so it shares the exit.
                    let (cont, cont_live) = self.drop_pass(*cont, exit);
                    self.cont_live.insert(id, cont_live);
                    spine.push(SpineFrame::Cont { id, cont });
                    e = *body;
                }
                CoreExpr::Drop { body, .. } => {
                    // Strip and re-derive, so the pass is safe to rerun.
                    e = *body;
                }
                CoreExpr::Tail(a) => break self.drop_tail(a, exit),
                CoreExpr::Goto(id) => {
                    // The edge hands the cont ownership of exactly its live-in
                    // set. Treating the `Goto` as a terminal reading that set
                    // makes the backward pass drop everything else before the
                    // jump, so no local is released both before an edge and
                    // inside the cont, or on neither.
                    let live = self
                        .cont_live
                        .get(&id)
                        .cloned()
                        .unwrap_or_else(|| unscoped_goto(id));
                    break (CoreExpr::Goto(id), live);
                }
                CoreExpr::If {
                    cond,
                    then,
                    els,
                    ty,
                } => {
                    let (then, live_t) = self.drop_pass(*then, exit);
                    let (els, live_e) = self.drop_pass(*els, exit);
                    // Both branches must release the same set, so the branch
                    // that does not need a local drops it at entry.
                    let then = self.wrap_drops(&set_diff(&live_e, &live_t), None, then);
                    let els = self.wrap_drops(&set_diff(&live_t, &live_e), None, els);
                    // `cond` is read here and never dropped: it is a Bool,
                    // which the immediates pass has made a value word, so it
                    // owns no cell on any path.
                    let mut live: Live = &live_t | &live_e;
                    live.insert(cond);
                    break (
                        CoreExpr::If {
                            cond,
                            then: Box::new(then),
                            els: Box::new(els),
                            ty,
                        },
                        live,
                    );
                }
                CoreExpr::Match { scrut, arms, ty } => {
                    break self.drop_match(scrut, arms, ty, exit);
                }
            }
        };
        while let Some(frame) = spine.pop() {
            match frame {
                SpineFrame::Let {
                    bind,
                    rhs_live,
                    rhs,
                } => {
                    // Newly dead here: rhs operands whose last read is this
                    // one, plus the bind itself if the body never reads it.
                    // They drop between rhs and body, as early as Perceus
                    // permits, so reuse tokens are hot.
                    let mut dead: Vec<LocalId> = rhs_live
                        .iter()
                        .copied()
                        .filter(|x| !live.contains(x))
                        .collect();
                    if !live.contains(&bind.id) {
                        dead.push(bind.id);
                    }
                    body = self.wrap_drops(&dead, None, body);
                    live.remove(&bind.id);
                    live.extend(rhs_live);
                    body = CoreExpr::Let {
                        bind,
                        rhs,
                        body: Box::new(body),
                    };
                }
                SpineFrame::Join { bind, join } => {
                    // Each `Tail` in the join is followed by `body`, so what
                    // `body` reads is live at every one of them. The join
                    // drops everything else on its own paths, which leaves
                    // only the bind's drop for here, when `body` never reads
                    // it. What `body` reads passes through the join owned, so
                    // it is in `join_live` too.
                    let read_by_body = live.remove(&bind.id);
                    let exit = Exit::Join {
                        ty: bind.ty,
                        after: live,
                    };
                    let (join, join_live) = self.drop_pass(*join, &exit);
                    if !read_by_body {
                        body = self.wrap_drops(&[bind.id], None, body);
                    }
                    live = join_live;
                    body = CoreExpr::LetJoin {
                        bind,
                        join: Box::new(join),
                        body: Box::new(body),
                    };
                }
                SpineFrame::Cont { id, cont } => {
                    body = CoreExpr::LetCont {
                        id,
                        cont: Box::new(cont),
                        body: Box::new(body),
                    };
                }
            }
        }
        (body, live)
    }

    fn drop_match(
        &mut self,
        scrut: LocalId,
        arms: Vec<(CorePat, CoreExpr)>,
        ty: RTy,
        exit: &Exit,
    ) -> (CoreExpr, Live) {
        struct Arm {
            pat: CorePat,
            body: CoreExpr,
            live: Live,
            dead_binders: Vec<LocalId>,
            scrut_shape: Option<ReuseShape>,
        }
        // The match reads `scrut`; each arm then owns it. `scrut` is live-in to
        // the Match but is not propagated into `live_union` from the arms: it
        // drops at each arm head unless that arm re-reads it.
        let mut live_union = Live::new();
        let mut lowered: Vec<Arm> = Vec::with_capacity(arms.len());
        for (pat, body) in arms {
            let (bound, scrut_shape) = self.record_pat(&pat, scrut);
            let (body, mut live) = self.drop_pass(body, exit);
            let dead_binders: Vec<LocalId> = bound
                .iter()
                .copied()
                .filter(|b| !live.contains(b))
                .collect();
            for b in &bound {
                live.remove(b);
            }
            live_union.extend(live.iter().copied());
            lowered.push(Arm {
                pat,
                body,
                live,
                dead_binders,
                scrut_shape,
            });
        }
        let scrut_live_after = live_union.contains(&scrut);
        let arms = lowered
            .into_iter()
            .map(|a| {
                // When the pattern proved the variant, the scrutinee's drop
                // carries that arity, so the reuse walk sees a sized token even
                // for a multi-variant type.
                let mut drops = set_diff(&live_union, &a.live);
                if !scrut_live_after && !a.live.contains(&scrut) {
                    drops.push(scrut);
                }
                drops.extend(a.dead_binders);
                let body = self.wrap_drops(&drops, a.scrut_shape.map(|s| (scrut, s)), a.body);
                (a.pat, body)
            })
            .collect();
        let mut live = live_union;
        live.insert(scrut);
        (CoreExpr::Match { scrut, arms, ty }, live)
    }

    /// A `Tail`, with what dies at it. Returned from the function, it takes
    /// the frame's references with it: a tail call reads its arguments before
    /// they go, so no `Drop` of one may come before it.
    ///
    /// Into a join, nothing runs between the `Tail` and the code after the
    /// join, so there is nowhere to drop a local the `Tail` reads for the
    /// last time. Its reference is handed to the join's bind instead, with
    /// [`Atom::Move`]: `Tail(x)` becomes `Tail(move x)`, and any other value
    /// reading such a local is bound first,
    /// `let t = a; drop <dying>; Tail(move t)`, so the drops follow the read
    /// as they would after a `Let`. This is Lean's and Koka's rule that a jump
    /// to a join point consumes its argument.
    fn drop_tail(&mut self, a: Atom, exit: &Exit) -> (CoreExpr, Live) {
        let reads = atom_live(&a);
        let Exit::Join { ty, after } = exit else {
            return (CoreExpr::Tail(a), reads);
        };
        let live: Live = &reads | after;
        let dying: Vec<LocalId> = reads
            .iter()
            .copied()
            .filter(|x| !after.contains(x) && self.is_heap_local(*x))
            .collect();
        let tail = match a {
            _ if dying.is_empty() => CoreExpr::Tail(a),
            Atom::Local(x) | Atom::Move(x) => CoreExpr::Tail(Atom::Move(x)),
            a => {
                let t = self.mint(*ty, ctor_shape(&a));
                let value = if self.is_heap_local(t.id) {
                    Atom::Move(t.id)
                } else {
                    Atom::Local(t.id)
                };
                CoreExpr::Let {
                    bind: t,
                    rhs: a,
                    body: Box::new(self.wrap_drops(&dying, None, CoreExpr::Tail(value))),
                }
            }
        };
        (tail, live)
    }

    /// Register the locals a pattern binds. Returns the scrutinee's shape: a
    /// `Ctor` arm proves an arity, other patterns pass through what it had.
    fn record_pat(&mut self, pat: &CorePat, scrut: LocalId) -> (Vec<LocalId>, Option<ReuseShape>) {
        match pat {
            CorePat::Wild | CorePat::Lit(_) => (Vec::new(), self.shape.get(&scrut).copied()),
            CorePat::Bind(b) => {
                let s = self.shape.get(&scrut).copied();
                self.record_bind(b, s);
                (vec![b.id], s)
            }
            CorePat::Ctor { fields, .. } => {
                for fb in fields {
                    self.record_bind(fb, None);
                }
                (
                    fields.iter().map(|b| b.id).collect(),
                    Some(ReuseShape::ctor(fields.len())),
                )
            }
        }
    }

    /// Prefix `body` with `Drop{x}` for each heap-shaped `x` in `dead`, in
    /// reverse so the textual order matches `dead`. Non-heap locals are skipped:
    /// `Op::Drop` on an unboxed prim is a no-op that still costs dispatch.
    /// `arm_scrut` overrides the recorded shape for one id.
    fn wrap_drops(
        &mut self,
        dead: &[LocalId],
        arm_scrut: Option<(LocalId, ReuseShape)>,
        mut body: CoreExpr,
    ) -> CoreExpr {
        for &x in dead.iter().rev() {
            if !self.is_heap_local(x) {
                continue;
            }
            let shape = match arm_scrut {
                Some((s, sh)) if s == x => Some(sh),
                _ => self.shape.get(&x).copied(),
            };
            body = CoreExpr::Drop {
                local: x,
                shape,
                body: Box::new(body),
            };
        }
        body
    }
}

/// State of one function's reuse walk. Built fresh per [`reuse_pass`], so a
/// walk never observes another function's back-edge token sets.
struct ReuseWalk {
    /// Token sets reaching each `Tail(Call{Self_})` in the current walk.
    self_tails: Vec<Vec<Token>>,
}

/// Frame-limited reuse over the drop-annotated body, with a second seeded walk
/// for loop-carried tokens on tail-self-recursive bodies.
fn reuse_pass(mut body: CoreExpr) -> CoreExpr {
    let mut walk = ReuseWalk {
        self_tails: Vec::new(),
    };
    walk.pair(&mut body, &mut Vec::new(), None);
    // Only tokens present at *every* self-tail dominate the loop head. The
    // runtime `into_reuse_addr` debug-assert on rc==1 relies on that.
    if let Some(mut seed) = walk.back_edge_seed() {
        walk.pair(&mut body, &mut seed, None);
    }
    body
}

impl ReuseWalk {
    /// Consumes `self_tails`, so a later walk cannot mix in stale snapshots.
    fn back_edge_seed(&mut self) -> Option<Vec<Token>> {
        let tails = std::mem::take(&mut self.self_tails);
        let first = tails.first()?;
        let seed: Vec<Token> = first
            .iter()
            .filter(|t| tails.iter().all(|ts| ts.contains(t)))
            .map(|t| Token {
                carried: true,
                ..*t
            })
            .collect();
        (!seed.is_empty()).then_some(seed)
    }

    /// LIFO reuse pairing over `e`. `avail` is the stack of hollowed cells
    /// dominating the current point; forked (cloned) at branches, snapshotted
    /// at self-tails. `join` is `Some` inside a `LetJoin`'s join: it collects
    /// the stack reaching each `Tail` there, the join's exits.
    fn pair(
        &mut self,
        mut e: &mut CoreExpr,
        avail: &mut Vec<Token>,
        mut join: Option<&mut Vec<Vec<Token>>>,
    ) {
        loop {
            match e {
                CoreExpr::Drop { local, shape, body } => {
                    // A 0-payload cell has nothing to reuse: the header is the
                    // whole allocation.
                    if let Some(s) = *shape
                        && s.fields > 0
                    {
                        avail.push(Token {
                            slot: *local,
                            shape: s,
                            carried: false,
                        });
                    }
                    e = body;
                }
                CoreExpr::Let { bind, rhs, body } => {
                    // A non-tail `Call` in `rhs` does NOT invalidate tokens: the
                    // parked cell lives in this frame's slot, untouched by the
                    // callee. Frame-limited (ICFP'22 §4) constrains reuse to
                    // this frame; it does not fence intra-frame call sites.
                    if let Atom::Ctor { fields, reuse, .. } = rhs {
                        let want = ReuseShape::ctor(fields.len());
                        // Prefer this bind's own slot: `StoreLocal` is about to
                        // overwrite it, so a same-slot token must be consumed
                        // here or discarded by the retain below.
                        let i = avail
                            .iter()
                            .rposition(|t| t.slot == bind.id && t.shape == want)
                            .or_else(|| avail.iter().rposition(|t| !t.carried && t.shape == want));
                        if let Some(i) = i {
                            *reuse = Some(avail.remove(i).slot);
                        }
                    }
                    // The slot now holds a live value; a token for it would read
                    // that, not a hollowed cell.
                    avail.retain(|t| t.slot != bind.id);
                    e = body;
                }
                CoreExpr::LetJoin {
                    bind,
                    join: j,
                    body,
                } => {
                    // The code after the join is reached from each of its
                    // exits, so a cell is still parked there only if it is on
                    // every exit: one taken on some path, or parked on only
                    // some, is not. A join no `Tail` leaves gives nothing.
                    let mut exits: Vec<Vec<Token>> = Vec::new();
                    self.pair(j, &mut avail.clone(), Some(&mut exits));
                    *avail = match exits.split_first() {
                        Some((first, rest)) => first
                            .iter()
                            .filter(|t| rest.iter().all(|x| x.contains(t)))
                            .copied()
                            .collect(),
                        None => Vec::new(),
                    };
                    avail.retain(|t| t.slot != bind.id);
                    e = body;
                }
                CoreExpr::LetCont { cont, body, .. } => {
                    // A shared cont is entered from many `Goto` edges, and a
                    // token parked on one edge may be a live slot on another, so
                    // it starts with an empty stack. Declaring a cont executes
                    // nothing, so the body's `avail` is untouched. Its `Tail`s
                    // are exits of the same join as the body's.
                    self.pair(cont, &mut Vec::new(), join.as_deref_mut());
                    e = body;
                }
                CoreExpr::If { then, els, .. } => {
                    let mut a2 = avail.clone();
                    self.pair(then, avail, join.as_deref_mut());
                    self.pair(els, &mut a2, join);
                    return;
                }
                CoreExpr::Match { arms, .. } => {
                    for (_, body) in arms {
                        self.pair(body, &mut avail.clone(), join.as_deref_mut());
                    }
                    return;
                }
                CoreExpr::Tail(a) => {
                    match a {
                        Atom::Ctor { fields, reuse, .. } => {
                            let want = ReuseShape::ctor(fields.len());
                            if let Some(i) =
                                avail.iter().rposition(|t| !t.carried && t.shape == want)
                            {
                                *reuse = Some(avail.remove(i).slot);
                            }
                        }
                        // In a join this is an ordinary call, not a back edge.
                        Atom::Call {
                            callee: Callee::Self_,
                            ..
                        } if join.is_none() => {
                            self.self_tails.push(avail.clone());
                        }
                        // A self-tail filter: every other atom, current or
                        // future, is by definition not a self tail call.
                        #[allow(unknown_lints, wildcard_over_own_enum)]
                        _ => {}
                    }
                    if let Some(exits) = join {
                        exits.push(avail.clone());
                    }
                    return;
                }
                // Tokens die at the edge: the target cont starts empty.
                CoreExpr::Goto(_) => return,
            }
        }
    }
}

/// Free locals of an atom. A duplicate operand (`add(x, x)`) collapses; the
/// VM's `PushLocal` dup keeps RC correct.
fn atom_live(a: &Atom) -> Live {
    let mut live = Live::new();
    a.for_each_operand(|id| {
        live.insert(id);
    });
    live
}

/// `a \ b` as a sorted vec.
fn set_diff(a: &Live, b: &Live) -> Vec<LocalId> {
    a.difference(b).copied().collect()
}

/// A `Goto` whose join was never peeled is a lowering bug. Proceeding with an
/// empty live-in set would drop locals the continuation still reads, so abort
/// in release too, like emit's `unbound_join`.
#[allow(clippy::panic)]
#[cold]
#[inline(never)]
fn unscoped_goto(id: JoinId) -> ! {
    panic!(
        "internal compiler error: perceus reached goto to join {id} with no \
         enclosing LetCont. Report this as a compiler bug."
    )
}

/// Take a module toplevel's pinned globals back out of Perceus's hands: no
/// `Drop` or `Atom::Move` releases one, and no `Ctor` reuses one's cell.
///
/// Perceus sees only the toplevel, where a global's last read is not its last
/// use: function bodies read it through `Load::Global` for as long as the
/// program runs. Releasing or overwriting it at the toplevel's last read would
/// free a value the program still holds.
pub(crate) fn keep_globals(mut f: CoreFn) -> CoreFn {
    let pinned: BTreeSet<LocalId> = f
        .body
        .toplevel_globals()
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    if !pinned.is_empty() {
        spare(&mut f.body, &pinned);
    }
    f
}

/// [`keep_globals`]'s walk. Iterative along each spine and recursive only into
/// branches, so a long toplevel costs no stack depth.
fn spare(mut e: &mut CoreExpr, pinned: &BTreeSet<LocalId>) {
    loop {
        if let CoreExpr::Drop { local, body, .. } = &mut *e
            && pinned.contains(local)
        {
            let rest = std::mem::replace(&mut **body, CoreExpr::Tail(Atom::Nil));
            *e = rest;
            continue;
        }
        match e {
            CoreExpr::Let { rhs, body, .. } => {
                spare_atom(rhs, pinned);
                e = body;
            }
            CoreExpr::LetJoin { join, body, .. } => {
                spare(join, pinned);
                e = body;
            }
            CoreExpr::LetCont { cont, body, .. } => {
                spare(cont, pinned);
                e = body;
            }
            CoreExpr::Drop { body, .. } => e = body,
            CoreExpr::If { then, els, .. } => {
                spare(then, pinned);
                e = els;
            }
            CoreExpr::Match { arms, .. } => {
                for (_, body) in arms {
                    spare(body, pinned);
                }
                return;
            }
            CoreExpr::Tail(a) => {
                spare_atom(a, pinned);
                return;
            }
            CoreExpr::Goto(_) => return,
        }
    }
}

/// A move of a pinned global gives its reference up as a `Drop` would, so it
/// becomes a read that shares.
fn spare_atom(a: &mut Atom, pinned: &BTreeSet<LocalId>) {
    if let Atom::Ctor { reuse, .. } = a
        && reuse.is_some_and(|r| pinned.contains(&r))
    {
        *reuse = None;
    }
    if let Atom::Move(x) = *a
        && pinned.contains(&x)
    {
        *a = Atom::Local(x);
    }
}

/// Known allocation shape of a `Let`'s rhs. Only a `Ctor` rhs proves a shape;
/// `Call`/`PrimOp` results have no statically-known arity.
fn ctor_shape(a: &Atom) -> Option<ReuseShape> {
    match a {
        Atom::Ctor { fields, .. } => Some(ReuseShape::ctor(fields.len())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_ir::PrimOp;
    use crate::core_ir::testkit::{bind, ctor, func, local, variant};
    use crate::core_ir::{ConstId, FuncIdx};
    use crate::type_def::TypeId;
    use crate::types::PrimIds;

    /// The same arena the elaborator hands `perceus` in the real pipeline.
    fn pool() -> ResolvedPool {
        ResolvedPool::new(PrimIds::default())
    }

    /// A nominal, non-primitive type: heap-shaped, unknown allocation width.
    fn con(p: &mut ResolvedPool, id: i32) -> RTy {
        p.mk_con(TypeId(id), &[])
    }

    /// The unboxed `Int` — heap-shaped `false`, so no `Drop`.
    fn int_ty(p: &mut ResolvedPool) -> RTy {
        let int = p.prims().int;
        p.mk_con(int, &[])
    }

    fn count_drops(e: &CoreExpr) -> usize {
        match e {
            CoreExpr::Drop { body, .. } => 1 + count_drops(body),
            CoreExpr::Let { body, .. } => count_drops(body),
            CoreExpr::LetJoin { join, body, .. } => count_drops(join) + count_drops(body),
            CoreExpr::LetCont { cont, body, .. } => count_drops(cont) + count_drops(body),
            CoreExpr::If { then, els, .. } => count_drops(then) + count_drops(els),
            CoreExpr::Match { arms, .. } => arms.iter().map(|(_, b)| count_drops(b)).sum(),
            CoreExpr::Tail(_) | CoreExpr::Goto(_) => 0,
        }
    }

    fn ctor_reuses(e: &CoreExpr) -> Vec<Option<LocalId>> {
        fn go(e: &CoreExpr, out: &mut Vec<Option<LocalId>>) {
            match e {
                CoreExpr::Let { rhs, body, .. } => {
                    if let Atom::Ctor { reuse, .. } = rhs {
                        out.push(*reuse);
                    }
                    go(body, out);
                }
                CoreExpr::LetJoin { join, body, .. } => {
                    go(join, out);
                    go(body, out);
                }
                CoreExpr::LetCont { cont, body, .. } => {
                    go(cont, out);
                    go(body, out);
                }
                CoreExpr::Drop { body, .. } => go(body, out),
                CoreExpr::If { then, els, .. } => {
                    go(then, out);
                    go(els, out);
                }
                CoreExpr::Match { arms, .. } => {
                    for (_, b) in arms {
                        go(b, out);
                    }
                }
                CoreExpr::Tail(Atom::Ctor { reuse, .. }) => out.push(*reuse),
                CoreExpr::Tail(_) | CoreExpr::Goto(_) => {}
            }
        }
        let mut out = Vec::new();
        go(e, &mut out);
        out
    }

    fn find_drop(mut e: &CoreExpr, id: LocalId) -> bool {
        loop {
            match e {
                CoreExpr::Drop { local, .. } if *local == id => return true,
                CoreExpr::Drop { body, .. }
                | CoreExpr::Let { body, .. }
                | CoreExpr::LetJoin { body, .. } => e = body,
                CoreExpr::LetCont { cont, body, .. } => {
                    return find_drop(cont, id) || find_drop(body, id);
                }
                CoreExpr::Tail(_) | CoreExpr::Goto(_) => return false,
                CoreExpr::If { then, els, .. } => {
                    return find_drop(then, id) || find_drop(els, id);
                }
                CoreExpr::Match { arms, .. } => {
                    return arms.iter().any(|(_, b)| find_drop(b, id));
                }
            }
        }
    }

    /// `map` shape: match xs { Cons(h,t) -> Cons(h+h, self t) | Nil -> Nil }.
    /// The Cons arm's tail Cons reuses `%0` across the recursive call; the Nil
    /// arm's arity-0 ctor does not pair.
    /// A toplevel keeps every value it pins to a global: Perceus's drop of
    /// one and a constructor's reuse of its cell both go, while a plain
    /// temporary's drop stays.
    #[test]
    fn keep_globals_spares_pinned_globals_only() {
        let mut pool = pool();
        let obj = con(&mut pool, 7);
        let mut global = bind(0, obj);
        global.global = Some(crate::typed_ir::GlobalSlot(0));
        let reuse_global = Atom::Ctor {
            variant: variant(),
            fields: Vec::new(),
            reuse: Some(local(0)),
        };
        let top = func(
            Vec::new(),
            CoreExpr::Let {
                bind: global,
                rhs: ctor(&[]),
                body: Box::new(CoreExpr::Let {
                    bind: bind(1, obj),
                    rhs: ctor(&[]),
                    body: Box::new(CoreExpr::Drop {
                        local: local(0),
                        shape: Some(ReuseShape::ctor(0)),
                        body: Box::new(CoreExpr::Drop {
                            local: local(1),
                            shape: Some(ReuseShape::ctor(0)),
                            body: Box::new(CoreExpr::Tail(reuse_global)),
                        }),
                    }),
                }),
            },
            obj,
        );
        let kept = keep_globals(top);
        assert_eq!(count_drops(&kept.body), 1, "only the temporary is dropped");
        assert!(
            matches!(
                kept.body,
                CoreExpr::Let { ref body, .. }
                    if matches!(**body, CoreExpr::Let { ref body, .. }
                        if matches!(**body, CoreExpr::Drop { local, .. } if local == LocalId(1)))
            ),
            "the global's drop is the one removed"
        );
        assert_eq!(
            ctor_reuses(&kept.body),
            vec![None, None, None],
            "no constructor reuses the global's cell"
        );
    }

    #[test]
    fn map_reuse_across_call_and_arm_shape() {
        let mut pool = pool();
        let list = con(&mut pool, 99);
        let int = int_ty(&mut pool);
        let cons_body = CoreExpr::Let {
            bind: bind(3, int),
            rhs: Atom::prim(PrimOp::IntAdd, vec![local(1), local(1)]),
            body: Box::new(CoreExpr::Let {
                bind: bind(4, list),
                rhs: Atom::Call {
                    callee: Callee::Self_,
                    args: vec![local(2)],
                },
                body: Box::new(CoreExpr::Tail(ctor(&[3, 4]))),
            }),
        };
        let f = func(
            vec![bind(0, list)],
            CoreExpr::Match {
                scrut: local(0),
                arms: vec![
                    (
                        CorePat::Ctor {
                            variant: variant(),
                            fields: vec![bind(1, int), bind(2, list)],
                        },
                        cons_body,
                    ),
                    (
                        CorePat::Ctor {
                            variant: variant(),
                            fields: vec![],
                        },
                        CoreExpr::Tail(ctor(&[])),
                    ),
                ],
                ty: list,
            },
            list,
        );
        let f = perceus(&pool, f);
        let reuses = ctor_reuses(&f.body);
        assert_eq!(reuses.len(), 2);
        assert_eq!(
            reuses[0],
            Some(local(0)),
            "Cons-arm tail ctor reuses %0 across the call (canonical map reuse):\n{}",
            f.body
        );
        assert_eq!(
            reuses[1], None,
            "Nil-arm arity-0 ctor does not pair (0-payload reuse is a no-op):\n{}",
            f.body
        );
        let CoreExpr::Match { arms, .. } = &f.body else {
            panic!()
        };
        assert!(find_drop(&arms[0].1, local(0)));
        assert!(find_drop(&arms[1].1, local(0)));
    }

    /// Straight-line `match xs { Cons(h,t) -> Cons(h,t) }`: the arity-2 tail
    /// Cons reuses the scrutinee dropped at the arm head.
    #[test]
    fn match_scrutinee_reused_in_arm() {
        let mut pool = pool();
        let list = con(&mut pool, 99);
        let int = int_ty(&mut pool);
        let f = func(
            vec![bind(0, list)],
            CoreExpr::Match {
                scrut: local(0),
                arms: vec![(
                    CorePat::Ctor {
                        variant: variant(),
                        fields: vec![bind(1, int), bind(2, list)],
                    },
                    CoreExpr::Tail(ctor(&[1, 2])),
                )],
                ty: list,
            },
            list,
        );
        let f = perceus(&pool, f);
        assert_eq!(ctor_reuses(&f.body), vec![Some(local(0))], "{}", f.body);
    }

    /// `dot_loop` shape: two 3-arity ctors, consumed by a Call, then a
    /// tail-self recursion. Loop-carried reuse pairs each ctor with the
    /// previous iteration's dropped cell.
    #[test]
    fn loop_carried_reuse_across_tail_self() {
        let mut pool = pool();
        let point = con(&mut pool, 99);
        let int = int_ty(&mut pool);
        let body = CoreExpr::Let {
            bind: bind(2, point),
            rhs: ctor(&[0, 0, 0]),
            body: Box::new(CoreExpr::Let {
                bind: bind(3, point),
                rhs: ctor(&[0, 0, 0]),
                body: Box::new(CoreExpr::Let {
                    bind: bind(4, int),
                    rhs: Atom::Call {
                        callee: Callee::Known(FuncIdx(7)),
                        args: vec![local(2), local(3)],
                    },
                    body: Box::new(CoreExpr::If {
                        cond: local(0),
                        then: Box::new(CoreExpr::Tail(Atom::Local(local(1)))),
                        els: Box::new(CoreExpr::Tail(Atom::Call {
                            callee: Callee::Self_,
                            args: vec![local(0), local(4)],
                        })),
                        ty: int,
                    }),
                }),
            }),
        };
        let f = perceus(&pool, func(vec![bind(0, int), bind(1, int)], body, int));
        let reuses = ctor_reuses(&f.body);
        assert_eq!(
            reuses,
            vec![Some(local(2)), Some(local(3))],
            "each Ctor reuses its OWN slot (self-pairing); cross-pairing would read a live slot:\n{}",
            f.body
        );
        assert!(find_drop(&f.body, local(2)));
        assert!(find_drop(&f.body, local(3)));
    }

    /// A tail call reads its arguments before its frame's references go, so
    /// no argument is dropped ahead of it: that drop would come before the
    /// read. Other locals the frame holds still drop as usual.
    #[test]
    fn a_tail_call_argument_is_not_dropped_before_the_call() {
        let mut pool = pool();
        let list = con(&mut pool, 99);
        for callee in [Callee::Self_, Callee::Known(FuncIdx(3))] {
            let body = CoreExpr::Let {
                bind: bind(1, list),
                rhs: ctor(&[0]),
                body: Box::new(CoreExpr::Tail(Atom::Call {
                    callee,
                    args: vec![local(1)],
                })),
            };
            let f = perceus(&pool, func(vec![bind(0, list)], body, list));
            assert!(!find_drop(&f.body, local(1)), "{}", f.body);
            assert!(find_drop(&f.body, local(0)), "{}", f.body);
        }
    }

    /// Without a self-tail edge nothing loop-carries.
    #[test]
    fn no_loop_carry_without_tail_self() {
        let mut pool = pool();
        let t = con(&mut pool, 99);
        let int = int_ty(&mut pool);
        let body = CoreExpr::Let {
            bind: bind(1, t),
            rhs: ctor(&[0, 0]),
            body: Box::new(CoreExpr::Let {
                bind: bind(2, int),
                rhs: Atom::prim(PrimOp::IntAdd, vec![local(1)]),
                body: Box::new(CoreExpr::Tail(Atom::Local(local(2)))),
            }),
        };
        let f = perceus(&pool, func(vec![bind(0, int)], body, int));
        assert_eq!(ctor_reuses(&f.body), vec![None], "{}", f.body);
        assert!(find_drop(&f.body, local(1)), "%1 dropped after last use");
    }

    #[test]
    fn no_drop_for_unboxed_prims() {
        let mut pool = pool();
        let int = int_ty(&mut pool);
        let f = perceus(
            &pool,
            func(
                vec![bind(0, int)],
                CoreExpr::Tail(Atom::Const(ConstId(0))),
                int,
            ),
        );
        assert_eq!(count_drops(&f.body), 0);
    }

    /// A heap local from a `Call` rhs still drops, but with `shape: None`
    /// (unknown arity), so it never pairs.
    #[test]
    fn call_result_drops_without_shape() {
        let mut pool = pool();
        let t = con(&mut pool, 99);
        let f = perceus(
            &pool,
            func(
                vec![],
                CoreExpr::Let {
                    bind: bind(0, t),
                    rhs: Atom::Call {
                        callee: Callee::Known(FuncIdx(0)),
                        args: vec![],
                    },
                    body: Box::new(CoreExpr::Tail(ctor(&[]))),
                },
                t,
            ),
        );
        let CoreExpr::Let { body, .. } = &f.body else {
            panic!()
        };
        let CoreExpr::Drop { local, shape, .. } = &**body else {
            panic!("expected Drop for dead %0:\n{}", f.body)
        };
        assert_eq!(*local, LocalId(0));
        assert_eq!(*shape, None, "Call result has no known arity");
        assert_eq!(
            ctor_reuses(&f.body),
            vec![None],
            "shape-less drop never pairs"
        );
    }

    /// A local live into one `If` branch only drops at the head of the other,
    /// so both paths release it exactly once.
    #[test]
    fn if_join_equalises_ownership() {
        let mut pool = pool();
        let t = con(&mut pool, 99);
        let int = int_ty(&mut pool);
        let f = perceus(
            &pool,
            func(
                vec![bind(0, t), bind(1, int)],
                CoreExpr::If {
                    cond: local(1),
                    then: Box::new(CoreExpr::Tail(Atom::Local(local(0)))),
                    els: Box::new(CoreExpr::Tail(Atom::Call {
                        callee: Callee::Known(FuncIdx(0)),
                        args: vec![],
                    })),
                    ty: t,
                },
                t,
            ),
        );
        let CoreExpr::If { then, els, .. } = &f.body else {
            panic!()
        };
        assert_eq!(count_drops(then), 0, "then reads %0 as its return value");
        assert_eq!(count_drops(els), 1, "els drops %0 at head");
    }

    /// A rigid quantified variable may be instantiated at a cell, so a dead
    /// one drops like a nominal: the `Drop` of a value word gives up nothing.
    #[test]
    fn a_type_variable_drops_like_a_nominal() {
        let mut pool = pool();
        let generic = pool.mk_bound(0);
        let int = int_ty(&mut pool);
        let t = con(&mut pool, 99);
        for ty in [generic, t] {
            let f = perceus(
                &pool,
                func(
                    vec![bind(0, ty), bind(1, int)],
                    CoreExpr::Tail(Atom::Local(local(1))),
                    int,
                ),
            );
            assert_eq!(count_drops(&f.body), 1, "dead param drops at head");
            assert!(find_drop(&f.body, local(0)), "{}", f.body);
        }
    }

    #[test]
    fn shape_mismatch_does_not_pair() {
        let mut pool = pool();
        let t = con(&mut pool, 99);
        let f = perceus(
            &pool,
            func(
                vec![bind(0, t)],
                CoreExpr::Let {
                    bind: bind(1, t),
                    rhs: ctor(&[0, 0]),
                    body: Box::new(CoreExpr::Let {
                        bind: bind(2, t),
                        rhs: Atom::prim(PrimOp::IntAdd, vec![local(1)]),
                        body: Box::new(CoreExpr::Tail(ctor(&[2, 2, 2]))),
                    }),
                },
                t,
            ),
        );
        assert_eq!(
            ctor_reuses(&f.body),
            vec![None, None],
            "arity-2 drop cannot feed arity-3 ctor:\n{}",
            f.body
        );
    }

    /// Ownership equalisation across `Goto` edges: `%0` is live into the shared
    /// cont, `%2` is not. Every owned local is released exactly once on every
    /// path, before the edge or inside the cont, never both.
    #[test]
    fn goto_edges_equalise_ownership() {
        let mut pool = pool();
        let t = con(&mut pool, 99);
        let int = int_ty(&mut pool);
        let f = perceus(
            &pool,
            func(
                vec![bind(0, t), bind(1, int), bind(2, t)],
                CoreExpr::LetCont {
                    id: JoinId(0),
                    cont: Box::new(CoreExpr::Tail(Atom::Local(local(0)))),
                    body: Box::new(CoreExpr::If {
                        cond: local(1),
                        then: Box::new(CoreExpr::Goto(JoinId(0))),
                        els: Box::new(CoreExpr::Tail(Atom::Local(local(2)))),
                        ty: t,
                    }),
                },
                t,
            ),
        );
        let CoreExpr::LetCont { cont, body, .. } = &f.body else {
            panic!("LetCont survives the pass:\n{}", f.body)
        };
        assert_eq!(count_drops(cont), 0, "cont returns %0:\n{}", f.body);
        let CoreExpr::If { then, els, .. } = &**body else {
            panic!("{}", f.body)
        };
        assert!(
            find_drop(then, local(2)),
            "Goto edge drops the local the cont does not consume:\n{}",
            f.body
        );
        assert!(
            !find_drop(then, local(0)),
            "Goto edge must not drop a cont live-in (the cont owns it):\n{}",
            f.body
        );
        assert!(
            find_drop(els, local(0)),
            "non-Goto path drops %0:\n{}",
            f.body
        );
        assert!(!find_drop(els, local(2)), "els returns %2:\n{}", f.body);
    }

    /// A local whose last use is inside the shared cont drops there, once, and
    /// no edge drops it before its `Goto`.
    #[test]
    fn drop_inside_shared_cont_not_on_edges() {
        let mut pool = pool();
        let t = con(&mut pool, 99);
        let int = int_ty(&mut pool);
        let cont = CoreExpr::Let {
            bind: bind(2, int),
            rhs: Atom::prim(PrimOp::IntAdd, vec![local(0)]),
            body: Box::new(CoreExpr::Tail(Atom::Local(local(2)))),
        };
        let f = perceus(
            &pool,
            func(
                vec![bind(0, t), bind(1, int)],
                CoreExpr::LetCont {
                    id: JoinId(0),
                    cont: Box::new(cont),
                    body: Box::new(CoreExpr::If {
                        cond: local(1),
                        then: Box::new(CoreExpr::Goto(JoinId(0))),
                        els: Box::new(CoreExpr::Goto(JoinId(0))),
                        ty: int,
                    }),
                },
                int,
            ),
        );
        assert_eq!(
            count_drops(&f.body),
            1,
            "exactly one Drop, inside the cont:\n{}",
            f.body
        );
        let CoreExpr::LetCont { cont, .. } = &f.body else {
            panic!("{}", f.body)
        };
        assert!(
            find_drop(cont, local(0)),
            "%0's last use is in the cont, so it drops there:\n{}",
            f.body
        );
    }

    /// `let %9 = (match %0 { V(%1) -> f(%1) }); %9`: the arm drops the
    /// scrutinee, and since nothing runs between its value and the code after
    /// the join, it binds the call to a fresh local, drops `%1` after it, and
    /// hands the fresh local over with `move`. Nothing drops after the join.
    #[test]
    fn a_join_arm_hands_over_the_value_that_reads_a_local_last() {
        let mut pool = pool();
        let t = con(&mut pool, 99);
        let join = CoreExpr::Match {
            scrut: local(0),
            arms: vec![(
                CorePat::Ctor {
                    variant: variant(),
                    fields: vec![bind(1, t)],
                },
                CoreExpr::Tail(Atom::Call {
                    callee: Callee::Known(FuncIdx(3)),
                    args: vec![local(1)],
                }),
            )],
            ty: t,
        };
        let f = perceus(
            &pool,
            func(
                vec![bind(0, t)],
                CoreExpr::LetJoin {
                    bind: bind(9, t),
                    join: Box::new(join),
                    body: Box::new(CoreExpr::Tail(Atom::Local(local(9)))),
                },
                t,
            ),
        );
        let CoreExpr::LetJoin { join, body, .. } = &f.body else {
            panic!("{}", f.body)
        };
        assert_eq!(count_drops(body), 0, "{}", f.body);
        let CoreExpr::Match { arms, .. } = &**join else {
            panic!("{}", f.body)
        };
        let want = "drop %0 [ctor:1]
let %10:";
        let arm = format!("{}", crate::core_ir::Indented(&arms[0].1, 0));
        assert!(arm.starts_with(want), "{arm}");
        assert!(
            arm.ends_with("= call fn#3(%1)\ndrop %1\nret move %10\n"),
            "{arm}"
        );
    }

    /// An arm's value that is a local the code after the join still reads is
    /// shared, not moved, and no arm drops it.
    #[test]
    fn a_join_arm_shares_a_local_the_code_after_reads() {
        let mut pool = pool();
        let t = con(&mut pool, 99);
        let int = int_ty(&mut pool);
        let f = perceus(
            &pool,
            func(
                vec![bind(0, t), bind(1, int)],
                CoreExpr::LetJoin {
                    bind: bind(2, t),
                    join: Box::new(CoreExpr::If {
                        cond: local(1),
                        then: Box::new(CoreExpr::Tail(Atom::Local(local(0)))),
                        els: Box::new(CoreExpr::Tail(ctor(&[]))),
                        ty: t,
                    }),
                    body: Box::new(CoreExpr::Tail(Atom::Call {
                        callee: Callee::Known(FuncIdx(3)),
                        args: vec![local(0), local(2)],
                    })),
                },
                t,
            ),
        );
        let CoreExpr::LetJoin { join, .. } = &f.body else {
            panic!("{}", f.body)
        };
        assert_eq!(count_drops(join), 0, "{}", f.body);
        let CoreExpr::If { then, .. } = &**join else {
            panic!("{}", f.body)
        };
        assert!(
            matches!(**then, CoreExpr::Tail(Atom::Local(l)) if l == local(0)),
            "{}",
            f.body
        );
    }

    /// A cell every arm of a join parks is still parked after it, so a
    /// constructor there takes it; one only some arms park is not.
    #[test]
    fn a_cell_parked_on_every_exit_of_a_join_is_reused_after_it() {
        let mut pool = pool();
        let t = con(&mut pool, 99);
        let int = int_ty(&mut pool);
        let arm = |then_parks: bool| {
            let first = CoreExpr::Match {
                scrut: local(0),
                arms: vec![(
                    CorePat::Ctor {
                        variant: variant(),
                        fields: vec![bind(3, int), bind(4, int)],
                    },
                    CoreExpr::Tail(Atom::Local(local(3))),
                )],
                ty: int,
            };
            let second = if then_parks {
                CoreExpr::Match {
                    scrut: local(0),
                    arms: vec![(
                        CorePat::Ctor {
                            variant: variant(),
                            fields: vec![bind(5, int), bind(6, int)],
                        },
                        CoreExpr::Tail(Atom::Local(local(6))),
                    )],
                    ty: int,
                }
            } else {
                CoreExpr::Tail(Atom::Local(local(1)))
            };
            CoreExpr::If {
                cond: local(1),
                then: Box::new(first),
                els: Box::new(second),
                ty: int,
            }
        };
        for (every, want) in [(true, Some(local(0))), (false, None)] {
            let f = perceus(
                &pool,
                func(
                    vec![bind(0, t), bind(1, int)],
                    CoreExpr::LetJoin {
                        bind: bind(2, int),
                        join: Box::new(arm(every)),
                        body: Box::new(CoreExpr::Tail(ctor(&[2, 2]))),
                    },
                    t,
                ),
            );
            assert_eq!(ctor_reuses(&f.body), vec![want], "{}", f.body);
        }
    }

    /// A token parked before the `LetCont` pairs on the fallthrough path but
    /// never inside the cont, whose token stack starts empty.
    #[test]
    fn cont_enters_with_empty_reuse_stack() {
        let mut pool = pool();
        let t = con(&mut pool, 99);
        let int = int_ty(&mut pool);
        let f = perceus(
            &pool,
            func(
                vec![bind(0, int)],
                CoreExpr::Let {
                    bind: bind(1, t),
                    rhs: ctor(&[0, 0]),
                    body: Box::new(CoreExpr::Let {
                        bind: bind(2, int),
                        rhs: Atom::prim(PrimOp::IntAdd, vec![local(1)]),
                        // Drop %1 [Enum:2] lands here, ahead of the LetCont.
                        body: Box::new(CoreExpr::LetCont {
                            id: JoinId(0),
                            cont: Box::new(CoreExpr::Tail(ctor(&[0, 0]))),
                            body: Box::new(CoreExpr::If {
                                cond: local(2),
                                then: Box::new(CoreExpr::Goto(JoinId(0))),
                                els: Box::new(CoreExpr::Tail(ctor(&[0, 0]))),
                                ty: t,
                            }),
                        }),
                    }),
                },
                t,
            ),
        );
        assert_eq!(
            ctor_reuses(&f.body),
            vec![None, None, Some(local(1))],
            "the cont's ctor must not claim %1's token; the body-path ctor may:\n{}",
            f.body
        );
    }
}
