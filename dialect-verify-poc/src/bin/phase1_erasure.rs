//! Phase 1 POC: the ghost-operation mechanism, tested against the real
//! `dialect-mir` crate's real pliron types -- not a simulation. Builds one
//! tiny function containing BOTH a real operation (`mir.store`) and a new
//! ghost operation (`verify.assert`, defined here, standing in for what a
//! real `dialect-verify` crate would hold), in the same block, the same way
//! `#[requires(...)]`/`verify_assert!(...)` would sit next to real code in
//! the full design. Then runs a standalone erasure pass and shows the ghost
//! op is gone while the real op survives untouched.
//!
//! Deliberately standalone rather than patched into `mir-importer`: testing
//! that patch for real means rebuilding the actual `rustc-codegen-cuda`
//! backend, which needs the `rustc-dev` toolchain component (not installed)
//! and, for a full build, `llc`/`clang` (not installed either). This proves
//! the mechanical claim -- ghost op and real op coexist in one `dialect-mir`
//! function, erasure cleanly removes just the ghost one -- against the real
//! `MirStoreOp`/`Operation`/`BasicBlock` types, without that heavier
//! toolchain. See ../NOTES.md for the honest scope note.

use dialect_mir::{
    ops::{MirFuncOp, MirReturnOp, MirStoreOp},
    types::MirPtrType,
};
use dialect_verify::{self as ghost_ops, VerifyAssertOp};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::TypeAttr,
        types::{FunctionType, IntegerType, Signedness},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
};

fn op_name(ctx: &Context, op: Ptr<Operation>) -> String {
    Operation::get_opid(op, ctx).to_string()
}

fn main() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    ghost_ops::register(&mut ctx);

    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);

    // fn poc(p: *mut i32, cond: i1, v: i32) -> ()
    let func_ty = FunctionType::get(
        &ctx,
        vec![ptr_ty.into(), i1_ty.into(), i32_ty.into()],
        vec![],
    );
    let func_ty_attr = TypeAttr::new(func_ty.into());
    let func_op_ptr = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let mir_func = MirFuncOp::new(&mut ctx, func_op_ptr, func_ty_attr);

    let entry_block = BasicBlock::new(
        &mut ctx,
        None,
        vec![ptr_ty.into(), i1_ty.into(), i32_ty.into()],
    );
    let ptr_val = entry_block.deref(&ctx).get_argument(0);
    let cond_val = entry_block.deref(&ctx).get_argument(1);
    let stored_val = entry_block.deref(&ctx).get_argument(2);
    let region = mir_func.get_operation().deref(&ctx).get_region(0);
    entry_block.insert_at_front(region, &ctx);

    // The ghost op: verify.assert(cond) -- e.g. "idx < a.len()". Inserted
    // FIRST, immediately before the real memory operation it's guarding,
    // the same relative position `#[requires(...)]` would occupy in source.
    let assert_op = VerifyAssertOp::build(&mut ctx, cond_val);
    assert_op.insert_at_back(entry_block, &ctx);

    // The real op: a genuine mir.store, the same operation `TILE[tid] = v`
    // lowers to per the Phase-0 trace -- completely ordinary `dialect-mir`,
    // untouched by the ghost op sitting next to it. `stored_val` is just a
    // block argument; what it's bound to at runtime doesn't matter for this
    // test, only that the store op itself survives erasure unmodified.
    let store_op = Operation::new(
        &mut ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![ptr_val, stored_val],
        vec![],
        0,
    );
    store_op.insert_at_back(entry_block, &ctx);

    let ret_op = Operation::new(
        &mut ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    ret_op.insert_at_back(entry_block, &ctx);

    assert!(mir_func.verify(&ctx).is_ok(), "function should verify");
    assert!(
        MirStoreOp::new(store_op).verify(&ctx).is_ok(),
        "real mir.store op should verify"
    );
    assert!(
        VerifyAssertOp::new(assert_op).verify(&ctx).is_ok(),
        "ghost verify.assert op should verify"
    );

    println!("=== BEFORE erasure ===");
    let before: Vec<Ptr<Operation>> = entry_block.deref(&ctx).iter(&ctx).collect();
    for op in &before {
        println!("  {}", op_name(&ctx, *op));
    }
    println!("  ({} operations total)", before.len());

    // The erasure pass: delete every `verify.*` operation. In the full
    // design this runs once, between MIR import and `mem2reg`, over every
    // function in the module -- here it's the same logic, over one block.
    let to_erase: Vec<Ptr<Operation>> = before
        .iter()
        .copied()
        .filter(|op| op_name(&ctx, *op) == "verify.assert")
        .collect();
    for op in to_erase {
        Operation::erase(op, &mut ctx);
    }

    println!("=== AFTER erasure ===");
    let after: Vec<Ptr<Operation>> = entry_block.deref(&ctx).iter(&ctx).collect();
    for op in &after {
        println!("  {}", op_name(&ctx, *op));
    }
    println!("  ({} operations total)", after.len());

    assert!(
        after.iter().all(|op| op_name(&ctx, *op) != "verify.assert"),
        "ghost op must be gone after erasure"
    );
    assert!(
        after
            .iter()
            .any(|op| op_name(&ctx, *op) == "mir.store"),
        "real op must survive erasure untouched"
    );
    assert!(mir_func.verify(&ctx).is_ok(), "function should STILL verify after erasure");

    println!();
    println!("PASS: ghost op coexisted with the real op, erasure removed only the");
    println!("ghost op, and the real op + function remain valid afterward.");
}
