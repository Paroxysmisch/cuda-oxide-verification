//! Phase 2: a real `dialect-mir` -> Viper textual translator, walking the
//! actual pliron op graph (not hand-written per example). Deliberately
//! covers a fixed, small, documented subset -- straight-line arithmetic,
//! one level of shared-memory indirection, structured if/else, and a
//! single-back-edge while loop -- not arbitrary Rust. See NOTES.md for
//! exactly what's covered and what isn't.
//!
//! Core design point: structural translation (control flow, arithmetic,
//! memory ops) is mechanical and derived from real SSA values -- there is
//! one fixed rule per `dialect-mir` op, auditable independently of any
//! specific kernel. Only `verify.barrier`'s permission clauses are
//! authored directly in Viper syntax (see ghost_ops.rs), because
//! `dialect-mir` has no SSA representation of permissions to derive them
//! from. Everything else here has a real SSA value behind it.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use dialect_mir::ops::{MirConstantOp, MirFuncOp, MirLoadOp, MirStoreOp};
use pliron::{
    basic_block::BasicBlock,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    value::Value,
};

use crate::ghost_ops::{VerifyAssertOp, VerifyBarrierOp, VerifyInvariantOp, VerifyPermInvariantOp};

pub struct Translator<'c> {
    ctx: &'c Context,
    /// Every SSA value (block argument or op result) gets a stable Viper
    /// identifier the first time it's seen.
    names: HashMap<Value, String>,
    counter: usize,
    /// Values that denote "the shared tile itself" (the result of
    /// `mir.shared_alloc`, or a function parameter standing in for one).
    tile_values: HashSet<Value>,
    /// For a pointer value computed by `mir.ptr_offset` from a tile value,
    /// the Viper index EXPRESSION text into `cells`.
    tile_index: HashMap<Value, String>,
    /// Pointers from `mir.alloca` -- a plain scalar stack slot (the
    /// pre-`mem2reg` representation of a mutable local, per `mir.alloca`'s
    /// own doc comment). Loads/stores through these are a direct
    /// read/assignment of one Viper local, not a `cells[...]` access.
    scalar_locals: HashSet<Value>,
    /// For a constant/arithmetic/comparison value, its fully-INLINED Viper
    /// expression (operands substituted recursively, not just named).
    /// Needed for `verify.invariant`: Silicon's loop treatment havocs
    /// every local the body writes to and only re-assumes what the
    /// invariant *textually* states, so an invariant clause that's just a
    /// bare variable name (itself havoc'd) proves nothing -- confirmed by
    /// running it and watching Silicon lose the bound. The clause needs
    /// the real arithmetic inlined, down to the one variable (`stride`,
    /// here) that's genuinely loop-carried and invariant-protected.
    expr_text: HashMap<Value, String>,
}

impl<'c> Translator<'c> {
    pub fn new(ctx: &'c Context) -> Self {
        Translator {
            ctx,
            names: HashMap::new(),
            counter: 0,
            tile_values: HashSet::new(),
            tile_index: HashMap::new(),
            scalar_locals: HashSet::new(),
            expr_text: HashMap::new(),
        }
    }

    /// The inlined expression for `v` if one is tracked (a constant or a
    /// pure arithmetic/comparison result), else just its variable name --
    /// e.g. for a loop-carried mutable local, which is exactly the base
    /// case inlining should bottom out at.
    fn atom(&mut self, v: Value) -> String {
        if let Some(e) = self.expr_text.get(&v) {
            e.clone()
        } else {
            self.name_of(v)
        }
    }

    /// Pre-bind a known value (typically a block/function argument) to a
    /// fixed Viper name, e.g. binding the entry block's shared-tile
    /// argument to the literal name `cells`.
    pub fn bind(&mut self, v: Value, name: &str) {
        self.names.insert(v, name.to_string());
    }

    /// Mark a pre-bound value as denoting the shared tile itself (so
    /// `mir.ptr_offset` from it is recognized as a tile index, not treated
    /// as an opaque pointer).
    pub fn mark_tile(&mut self, v: Value) {
        self.tile_values.insert(v);
    }

    fn name_of(&mut self, v: Value) -> String {
        if let Some(n) = self.names.get(&v) {
            return n.clone();
        }
        let n = format!("v{}", self.counter);
        self.counter += 1;
        self.names.insert(v, n.clone());
        n
    }

    fn result0(&mut self, op: Ptr<Operation>) -> Value {
        op.deref(self.ctx).get_result(0)
    }

    fn opid(&self, op: Ptr<Operation>) -> String {
        Operation::get_opid(op, self.ctx).to_string()
    }

    /// Translate every non-terminator op in `block` into `out`, in order.
    /// Returns the collected `verify.invariant` condition names, if any
    /// (used when `block` is being translated as a loop header).
    ///
    /// `redeclare` is true for the normal, first-time translation of a
    /// block (`var n: T := expr`). A loop header's straight-line ops --
    /// everything that computes its continuation condition and its
    /// invariants' truth values -- genuinely re-execute every time control
    /// reaches the header, including via the back edge; a real MIR
    /// interpreter would re-run them each iteration. So the loop-handling
    /// code below calls this a second time, with `redeclare = false`, to
    /// emit a duplicate copy at the tail of the loop body that REASSIGNS
    /// the same (already-minted) names rather than redeclaring them --
    /// otherwise the condition and invariants would silently be computed
    /// once from the initial state and never actually re-checked, which
    /// is wrong regardless of whether Silicon happens to still accept it.
    fn translate_ops(
        &mut self,
        out: &mut String,
        block: Ptr<BasicBlock>,
        indent: usize,
        redeclare: bool,
    ) -> Vec<String> {
        let pad = "  ".repeat(indent);
        let mut invariants = Vec::new();
        let terminator = self.terminator(block);
        let ops: Vec<Ptr<Operation>> = block.deref(self.ctx).iter(self.ctx).collect();
        for op in ops {
            if op == terminator {
                continue;
            }
            let id = self.opid(op);
            match id.as_str() {
                "mir.shared_alloc" => {
                    // The tile already exists as the bound `cells` param;
                    // just mark this op's result as denoting it too.
                    let r = self.result0(op);
                    let n = self.name_of(r);
                    self.tile_values.insert(r);
                    let _ = n; // no Viper statement: cells already exists
                }
                "mir.constant" => {
                    let const_op = MirConstantOp::new(op);
                    let attr = const_op.get_attr_value(self.ctx).expect("mir.constant value attr");
                    let lit = attr.value().to_u64();
                    let r = self.result0(op);
                    let n = self.name_of(r);
                    self.expr_text.insert(r, lit.to_string());
                    if redeclare {
                        writeln!(out, "{pad}var {n}: Int := {lit}").unwrap();
                    } else {
                        writeln!(out, "{pad}{n} := {lit}").unwrap();
                    }
                }
                "mir.alloca" => {
                    let r = self.result0(op);
                    let n = self.name_of(r);
                    self.scalar_locals.insert(r);
                    assert!(redeclare, "mir.alloca must not be re-executed (it's a one-time stack slot)");
                    writeln!(out, "{pad}var {n}: Int").unwrap();
                }
                "mir.add" | "mir.sub" | "mir.mul" | "mir.rem" | "mir.div" => {
                    let sym = match id.as_str() {
                        "mir.add" => "+",
                        "mir.sub" => "-",
                        "mir.mul" => "*",
                        "mir.rem" => "%",
                        "mir.div" => "\\",
                        _ => unreachable!(),
                    };
                    let o = op.deref(self.ctx);
                    let opd0 = o.get_operand(0);
                    let opd1 = o.get_operand(1);
                    drop(o);
                    let a = self.name_of(opd0);
                    let b = self.name_of(opd1);
                    let a_atom = self.atom(opd0);
                    let b_atom = self.atom(opd1);
                    let r = self.result0(op);
                    let n = self.name_of(r);
                    self.expr_text.insert(r, format!("{a_atom} {sym} {b_atom}"));
                    if redeclare {
                        writeln!(out, "{pad}var {n}: Int := {a} {sym} {b}").unwrap();
                    } else {
                        writeln!(out, "{pad}{n} := {a} {sym} {b}").unwrap();
                    }
                }
                "mir.lt" | "mir.le" | "mir.gt" | "mir.ge" | "mir.eq" | "mir.ne" => {
                    let sym = match id.as_str() {
                        "mir.lt" => "<",
                        "mir.le" => "<=",
                        "mir.gt" => ">",
                        "mir.ge" => ">=",
                        "mir.eq" => "==",
                        "mir.ne" => "!=",
                        _ => unreachable!(),
                    };
                    let o = op.deref(self.ctx);
                    let opd0 = o.get_operand(0);
                    let opd1 = o.get_operand(1);
                    drop(o);
                    let a = self.name_of(opd0);
                    let b = self.name_of(opd1);
                    let a_atom = self.atom(opd0);
                    let b_atom = self.atom(opd1);
                    let r = self.result0(op);
                    let n = self.name_of(r);
                    self.expr_text.insert(r, format!("{a_atom} {sym} {b_atom}"));
                    if redeclare {
                        writeln!(out, "{pad}var {n}: Bool := {a} {sym} {b}").unwrap();
                    } else {
                        writeln!(out, "{pad}{n} := {a} {sym} {b}").unwrap();
                    }
                }
                "mir.ptr_offset" => {
                    let o = op.deref(self.ctx);
                    let base = o.get_operand(0);
                    let offset = o.get_operand(1);
                    drop(o);
                    let offset_name = self.name_of(offset);
                    let idx_expr = if self.tile_values.contains(&base) {
                        offset_name
                    } else if let Some(base_idx) = self.tile_index.get(&base) {
                        format!("({base_idx} + {offset_name})")
                    } else {
                        panic!("mir.ptr_offset base is not a recognized tile pointer");
                    };
                    let r = self.result0(op);
                    self.tile_index.insert(r, idx_expr);
                    // record a name too, purely for readability in dumps
                    self.name_of(r);
                }
                "mir.load" => {
                    let load_op = MirLoadOp::new(op);
                    let ptr = load_op.address_opd(self.ctx);
                    let r = self.result0(op);
                    if self.scalar_locals.contains(&ptr) {
                        // A load of an unaliased scalar local is just its
                        // current value -- alias the result to the SAME
                        // Viper name rather than minting (and emitting a
                        // statement for) a fresh copy. Two concrete
                        // problems this avoids, found by actually running
                        // Silicon on the loop below: (1) Silicon havocs
                        // every local the loop body writes to at the top
                        // of each iteration, including any such copy, so a
                        // copy's relationship to the original is lost
                        // right when the invariant needs it; (2) it's what
                        // `mem2reg` is going to do to this exact load
                        // anyway, so it's the more faithful translation,
                        // not just the convenient one.
                        let local_name = self.name_of(ptr);
                        self.names.insert(r, local_name);
                    } else {
                        let n = self.name_of(r);
                        let rhs = if let Some(idx) = self.tile_index.get(&ptr).cloned() {
                            format!("cells[{idx}].val")
                        } else {
                            format!("{}.val", self.name_of(ptr))
                        };
                        if redeclare {
                            writeln!(out, "{pad}var {n}: Int := {rhs}").unwrap();
                        } else {
                            writeln!(out, "{pad}{n} := {rhs}").unwrap();
                        }
                    }
                }
                "mir.store" => {
                    let store_op = MirStoreOp::new(op);
                    let ptr = store_op.address_opd(self.ctx);
                    let val = store_op.value_opd(self.ctx);
                    let val_name = self.name_of(val);
                    if self.scalar_locals.contains(&ptr) {
                        let ptr_name = self.name_of(ptr);
                        writeln!(out, "{pad}{ptr_name} := {val_name}").unwrap();
                    } else if let Some(idx) = self.tile_index.get(&ptr).cloned() {
                        writeln!(out, "{pad}cells[{idx}].val := {val_name}").unwrap();
                    } else {
                        let ptr_name = self.name_of(ptr);
                        writeln!(out, "{pad}{ptr_name}.val := {val_name}").unwrap();
                    }
                }
                "verify.assert" => {
                    let a = VerifyAssertOp::new(op);
                    let cond = a.condition(self.ctx);
                    let n = self.name_of(cond);
                    writeln!(out, "{pad}assert {n}").unwrap();
                }
                "verify.invariant" => {
                    let a = VerifyInvariantOp::new(op);
                    let cond = a.condition(self.ctx);
                    let expr = self.atom(cond);
                    if redeclare {
                        invariants.push(expr);
                    }
                }
                "verify.invariant_perm" => {
                    let a = VerifyPermInvariantOp::new(op);
                    if redeclare {
                        invariants.push(a.clause(self.ctx));
                    }
                }
                "verify.barrier" => {
                    let b = VerifyBarrierOp::new(op);
                    let operand_names: Vec<String> = b
                        .operands(self.ctx)
                        .into_iter()
                        .map(|v| self.name_of(v))
                        .collect();
                    let subst = |tmpl: String| -> String {
                        let mut s = tmpl;
                        for (i, name) in operand_names.iter().enumerate() {
                            s = s.replace(&format!("${i}"), name);
                        }
                        s
                    };
                    writeln!(out, "{pad}exhale {}", subst(b.exhale_template(self.ctx))).unwrap();
                    writeln!(out, "{pad}inhale {}", subst(b.inhale_template(self.ctx))).unwrap();
                }
                other => panic!("translator: unhandled op {other}"),
            }
        }
        invariants
    }

    fn terminator(&self, block: Ptr<BasicBlock>) -> Ptr<Operation> {
        block
            .deref(self.ctx)
            .get_terminator(self.ctx)
            .expect("block has no terminator")
    }

    /// Does control starting at `from` ever reach `target`, following
    /// `mir.goto`/`mir.cond_br` successors? (`mir.return` has none.)
    fn reaches(&self, from: Ptr<BasicBlock>, target: Ptr<BasicBlock>) -> bool {
        let mut seen = HashSet::new();
        let mut stack = vec![from];
        while let Some(b) = stack.pop() {
            if b == target {
                return true;
            }
            if !seen.insert(b) {
                continue;
            }
            let term = self.terminator(b);
            let id = self.opid(term);
            match id.as_str() {
                "mir.goto" => stack.push(term.deref(self.ctx).get_successor(0)),
                "mir.cond_br" => {
                    stack.push(term.deref(self.ctx).get_successor(0));
                    stack.push(term.deref(self.ctx).get_successor(1));
                }
                _ => {}
            }
        }
        false
    }

    /// If `block`'s terminator is a plain `mir.goto` with no operands,
    /// return its target.
    fn plain_goto_target(&self, block: Ptr<BasicBlock>) -> Option<Ptr<BasicBlock>> {
        let term = self.terminator(block);
        if self.opid(term) != "mir.goto" {
            return None;
        }
        let o = term.deref(self.ctx);
        if o.get_num_operands() != 0 {
            return None;
        }
        Some(o.get_successor(0))
    }

    /// Translate the control flow starting at `block`, stopping (without
    /// emitting anything further) once control would reach `stop` -- the
    /// caller's known continuation point. `ancestors` is the set of blocks
    /// currently being translated higher up the recursion (used to detect
    /// back edges: a goto/branch target that's an ancestor just closes the
    /// enclosing `while` rather than being inlined again).
    fn translate_block(
        &mut self,
        out: &mut String,
        block: Ptr<BasicBlock>,
        ancestors: &mut Vec<Ptr<BasicBlock>>,
        stop: Option<Ptr<BasicBlock>>,
        indent: usize,
    ) {
        if Some(block) == stop {
            return;
        }
        ancestors.push(block);
        let invariants = self.translate_ops(out, block, indent, true);
        let term = self.terminator(block);
        let term_id = self.opid(term);
        match term_id.as_str() {
            "mir.return" => {}
            "mir.goto" => {
                let target = term.deref(self.ctx).get_successor(0);
                if Some(target) == stop || ancestors.contains(&target) {
                    // reached the continuation, or this IS the back edge
                } else {
                    self.translate_block(out, target, ancestors, stop, indent);
                }
            }
            "mir.cond_br" => {
                let o = term.deref(self.ctx);
                let cond = o.get_operand(0);
                let t = o.get_successor(0);
                let f = o.get_successor(1);
                drop(o);
                let cond_name = self.names.get(&cond).cloned().unwrap_or_else(|| {
                    panic!("cond_br condition value translated before use")
                });

                let pad = "  ".repeat(indent);

                // Simplest shape first: a guarded block with no `else` --
                // the true arm falls straight into the false arm's own
                // target (or vice versa). Must be checked before the
                // generic loop-back-edge test below, since inside a loop
                // body this shape's "true" arm *also* satisfies a naive
                // reachability-to-header check (it gets there via the
                // false arm's continuation), which would misfire.
                if self.plain_goto_target(t) == Some(f) {
                    writeln!(out, "{pad}if ({cond_name}) {{").unwrap();
                    self.translate_block(out, t, &mut vec![block], Some(f), indent + 1);
                    writeln!(out, "{pad}}}").unwrap();
                    self.translate_block(out, f, ancestors, stop, indent);
                    ancestors.pop();
                    return;
                }
                if self.plain_goto_target(f) == Some(t) {
                    writeln!(out, "{pad}if (!{cond_name}) {{").unwrap();
                    self.translate_block(out, f, &mut vec![block], Some(t), indent + 1);
                    writeln!(out, "{pad}}}").unwrap();
                    self.translate_block(out, t, ancestors, stop, indent);
                    ancestors.pop();
                    return;
                }

                let t_loops = self.reaches(t, block);
                let f_loops = self.reaches(f, block);

                if t_loops && !f_loops {
                    writeln!(out, "{pad}while ({cond_name})").unwrap();
                    for inv in &invariants {
                        writeln!(out, "{pad}  invariant {inv}").unwrap();
                    }
                    writeln!(out, "{pad}{{").unwrap();
                    let mut body_ancestors = vec![block];
                    self.translate_block(out, t, &mut body_ancestors, Some(block), indent + 1);
                    // The header's own ops (condition, invariants) genuinely
                    // re-execute every time control reaches it, including
                    // via this back edge -- re-emit them here, reassigning
                    // rather than redeclaring. See translate_ops' doc.
                    writeln!(out, "{pad}  // re-entering header (back edge): recompute its condition/invariants").unwrap();
                    self.translate_ops(out, block, indent + 1, false);
                    writeln!(out, "{pad}}}").unwrap();
                    self.translate_block(out, f, ancestors, stop, indent);
                } else if f_loops && !t_loops {
                    writeln!(out, "{pad}while (!{cond_name})").unwrap();
                    for inv in &invariants {
                        writeln!(out, "{pad}  invariant {inv}").unwrap();
                    }
                    writeln!(out, "{pad}{{").unwrap();
                    let mut body_ancestors = vec![block];
                    self.translate_block(out, f, &mut body_ancestors, Some(block), indent + 1);
                    writeln!(out, "{pad}  // re-entering header (back edge): recompute its condition/invariants").unwrap();
                    self.translate_ops(out, block, indent + 1, false);
                    writeln!(out, "{pad}}}").unwrap();
                    self.translate_block(out, t, ancestors, stop, indent);
                } else {
                    // Structured if/else: require both arms to converge via
                    // a plain `goto` to the same merge block.
                    let t_merge = self.plain_goto_target(t);
                    let f_merge = self.plain_goto_target(f);
                    let merge = match (t_merge, f_merge) {
                        (Some(m1), Some(m2)) if m1 == m2 => m1,
                        _ => panic!(
                            "translator: cond_br arms don't converge on a single plain-goto merge block (unsupported CFG shape)"
                        ),
                    };
                    writeln!(out, "{pad}if ({cond_name}) {{").unwrap();
                    self.translate_block(out, t, &mut vec![block], Some(merge), indent + 1);
                    writeln!(out, "{pad}}} else {{").unwrap();
                    self.translate_block(out, f, &mut vec![block], Some(merge), indent + 1);
                    writeln!(out, "{pad}}}").unwrap();
                    self.translate_block(out, merge, ancestors, stop, indent);
                }
            }
            other => panic!("translator: unhandled terminator {other}"),
        }
        ancestors.pop();
    }

    /// Translate a whole function's entry block (and everything reachable
    /// from it) into the body of a Viper `method`. `params` lists the
    /// function's block arguments paired with their intended Viper names
    /// and types (not translated automatically -- the caller states the
    /// method signature, since Viper's type/permission vocabulary has no
    /// counterpart to infer from `dialect-mir` types alone).
    pub fn translate_function_body(&mut self, func: MirFuncOp) -> String {
        let region = func.get_operation().deref(self.ctx).get_region(0);
        let entry: Ptr<BasicBlock> = region
            .deref(self.ctx)
            .iter(self.ctx)
            .next()
            .expect("function has an entry block");
        let mut out = String::new();
        let mut ancestors = Vec::new();
        self.translate_block(&mut out, entry, &mut ancestors, None, 1);
        out
    }
}
