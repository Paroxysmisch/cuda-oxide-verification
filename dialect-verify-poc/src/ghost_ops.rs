//! The ghost `verify.*` ops. Each erases to nothing; each carries exactly
//! the extra information a `dialect-mir` -> Viper translator needs that
//! `dialect-mir` itself has no way to express, because `dialect-mir` has no
//! concept of permissions/ownership at all -- that's precisely the gap
//! this whole project is bridging.
//!
//! `verify.assert` / `verify.invariant` carry a REAL SSA operand (a
//! `dialect-mir` i1 value) -- their condition is something `dialect-mir`
//! can already express (arithmetic + comparisons), so the translator
//! derives the Viper text for it structurally, the same way it derives
//! real ops' translations. No separate trust step for these.
//!
//! `verify.barrier` is different: a permission-redistribution across a
//! `sync_threads()` is a fact about *ownership*, a concept `dialect-mir`'s
//! SSA value system has no representation for at all. There is no SSA
//! expression to translate it from. So it carries its exhale/inhale
//! clauses as literal Viper assertion syntax, authored directly (by a
//! human or, in Phase 4, an LLM) in the target language. This is a
//! deliberate, named exception to "the translator derives everything
//! structurally" -- flagged here rather than smuggled in silently.

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    verify_err,
};
use pliron_derive::pliron_op;

#[pliron_op(
    name = "verify.assert",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<0>],
)]
pub struct VerifyAssertOp;

impl VerifyAssertOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        VerifyAssertOp { op }
    }

    pub fn build(ctx: &mut Context, condition: pliron::value::Value) -> Ptr<Operation> {
        Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![condition],
            vec![],
            0,
        )
    }

    pub fn condition(&self, ctx: &Context) -> pliron::value::Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

impl Verify for VerifyAssertOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 0 {
            return verify_err!(op.loc(), "verify.assert requires one operand, no results");
        }
        Ok(())
    }
}

/// A loop-invariant clause. Placed in a loop header block; the translator
/// collects every `verify.invariant` in a header into the Viper `while`'s
/// `invariant` clause list instead of emitting it as an inline `assert`.
#[pliron_op(
    name = "verify.invariant",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<0>],
)]
pub struct VerifyInvariantOp;

impl VerifyInvariantOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        VerifyInvariantOp { op }
    }

    pub fn build(ctx: &mut Context, condition: pliron::value::Value) -> Ptr<Operation> {
        Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![condition],
            vec![],
            0,
        )
    }

    pub fn condition(&self, ctx: &Context) -> pliron::value::Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

impl Verify for VerifyInvariantOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 0 {
            return verify_err!(op.loc(), "verify.invariant requires one operand, no results");
        }
        Ok(())
    }
}

/// A permission-only loop-invariant clause -- the same role as
/// `verify.invariant`, for the case the condition IS a permission
/// (`acc(...)`) rather than a value-level boolean, so there's no
/// `dialect-mir` SSA value to hang it on at all (same underlying reason
/// `verify.barrier`'s clauses are verbatim text, see its doc comment).
/// Needed in practice, not just in principle: the first real run of the
/// Phase 3 reduction kernel verified its value-level invariants fine but
/// then failed with "insufficient permission to access cells[tid].val" --
/// `acc(cells[tid].val)` must also be explicitly carried across the loop
/// boundary, exactly like any value-level fact, or Silicon's loop
/// treatment drops it at the loop's edge.
#[pliron_op(
    name = "verify.invariant_perm",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
    attributes = (clause: pliron::builtin::attributes::StringAttr)
)]
pub struct VerifyPermInvariantOp;

impl VerifyPermInvariantOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        VerifyPermInvariantOp { op }
    }

    pub fn build(ctx: &mut Context, clause: &str) -> Ptr<Operation> {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        VerifyPermInvariantOp::new(op)
            .set_attr_clause(ctx, pliron::builtin::attributes::StringAttr::new(clause.to_string()));
        op
    }

    pub fn clause(&self, ctx: &Context) -> String {
        self.get_attr_clause(ctx)
            .expect("verify.invariant_perm missing clause")
            .as_str()
            .to_string()
    }
}

impl Verify for VerifyPermInvariantOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        if op.get_num_operands() != 0 || op.get_num_results() != 0 {
            return verify_err!(op.loc(), "verify.invariant_perm takes no operands/results");
        }
        if self.get_attr_clause(ctx).is_none() {
            return verify_err!(op.loc(), "verify.invariant_perm requires a clause attribute");
        }
        Ok(())
    }
}

/// A `sync_threads()` permission redistribution. See the module doc for why
/// the clauses themselves are verbatim Viper text rather than translated
/// from an SSA expression -- `dialect-mir` has no permission concept to
/// translate FROM. But *which cells* those clauses talk about usually does
/// come from a real SSA value (e.g. `tid + stride`), so this op takes a
/// variadic list of such operands, and its templates reference them
/// positionally as `$0`, `$1`, ... -- substituted with each operand's real
/// translated Viper name at translation time, not guessed up front.
#[pliron_op(
    name = "verify.barrier",
    format,
    interfaces = [NResultsInterface<0>],
    attributes = (
        exhale_expr: pliron::builtin::attributes::StringAttr,
        inhale_expr: pliron::builtin::attributes::StringAttr
    )
)]
pub struct VerifyBarrierOp;

impl VerifyBarrierOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        VerifyBarrierOp { op }
    }

    pub fn build(
        ctx: &mut Context,
        operands: Vec<pliron::value::Value>,
        exhale_expr: &str,
        inhale_expr: &str,
    ) -> Ptr<Operation> {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], operands, vec![], 0);
        let wrapped = VerifyBarrierOp::new(op);
        wrapped.set_attr_exhale_expr(
            ctx,
            pliron::builtin::attributes::StringAttr::new(exhale_expr.to_string()),
        );
        wrapped.set_attr_inhale_expr(
            ctx,
            pliron::builtin::attributes::StringAttr::new(inhale_expr.to_string()),
        );
        op
    }

    pub fn exhale_template(&self, ctx: &Context) -> String {
        self.get_attr_exhale_expr(ctx)
            .expect("verify.barrier missing exhale_expr")
            .as_str()
            .to_string()
    }

    pub fn inhale_template(&self, ctx: &Context) -> String {
        self.get_attr_inhale_expr(ctx)
            .expect("verify.barrier missing inhale_expr")
            .as_str()
            .to_string()
    }

    pub fn operands(&self, ctx: &Context) -> Vec<pliron::value::Value> {
        self.get_operation().deref(ctx).operands().collect()
    }
}

impl Verify for VerifyBarrierOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        if op.get_num_results() != 0 {
            return verify_err!(op.loc(), "verify.barrier takes no results");
        }
        if self.get_attr_exhale_expr(ctx).is_none() || self.get_attr_inhale_expr(ctx).is_none() {
            return verify_err!(
                op.loc(),
                "verify.barrier requires exhale_expr and inhale_expr attributes"
            );
        }
        Ok(())
    }
}

/// Register every ghost op with the context. Mirrors the real dialects'
/// `register()` functions (e.g. `dialect_nvvm::register`).
pub fn register(ctx: &mut Context) {
    VerifyAssertOp::register(ctx);
    VerifyInvariantOp::register(ctx);
    VerifyPermInvariantOp::register(ctx);
    VerifyBarrierOp::register(ctx);
}

/// The erasure pass: delete every `verify.*` op reachable from `func`.
/// Walks all blocks of all regions, not just the entry block -- the real
/// pass (hosted in `mir-importer`, per the design) would run once per
/// function over the whole body.
pub fn erase_ghost_ops(ctx: &mut Context, func: dialect_mir::ops::MirFuncOp) {
    use pliron::linked_list::ContainsLinkedList;
    let region = func.get_operation().deref(ctx).get_region(0);
    let blocks: Vec<Ptr<pliron::basic_block::BasicBlock>> =
        region.deref(ctx).iter(ctx).collect();
    for block in blocks {
        let ops: Vec<Ptr<Operation>> = block.deref(ctx).iter(ctx).collect();
        for op in ops {
            let name = Operation::get_opid(op, ctx).to_string();
            if name.starts_with("verify.") {
                Operation::erase(op, ctx);
            }
        }
    }
}
