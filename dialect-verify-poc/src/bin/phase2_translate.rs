//! Phase 2: build a real `dialect-mir` function with a structured if/else
//! and real shared-memory stores through `mir.shared_alloc` +
//! `mir.ptr_offset` + `mir.store` -- the same op sequence Phase 0's Explore
//! agent traced for `TILE[tid] = v` -- then run it through the REAL
//! translator (`dialect_verify::translate`), not hand-written Viper,
//! and check the result with Silicon.
//!
//! Kernel being modeled:
//! ```text
//! fn poc(tid: i32, bound: i32, claim: i1) {
//!     static mut TILE: SharedArray<i32, 256> = ..;
//!     verify.assert(claim);
//!     if tid < bound {
//!         TILE[tid] = tid;
//!     } else {
//!         TILE[tid] = 0;
//!     }
//! }
//! ```
//! Emits two full `.vpr` files from the SAME translated body: one where the
//! method's own `requires` supplies `claim`, one where it doesn't. Prints
//! both, then (if `--run` is passed) shells out to Silicon on each, exactly
//! like Phase 0.

use dialect_mir::{ops::MirFuncOp, types::MirPtrType};
use dialect_verify::{self as ghost_ops, translate::Translator};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, TypeAttr},
        op_interfaces::OperandSegmentInterface,
        types::{FunctionType, IntegerType, Signedness},
    },
    common_traits::Verify,
    context::Context,
    op::Op,
    operation::Operation,
    utils::apint::APInt,
};

fn main() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    ghost_ops::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);

    let func_ty = FunctionType::get(&ctx, vec![i32_ty.into(), i32_ty.into(), i1_ty.into()], vec![]);
    let func_op_ptr = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let mir_func = MirFuncOp::new(&mut ctx, func_op_ptr, TypeAttr::new(func_ty.into()));
    let region = mir_func.get_operation().deref(&ctx).get_region(0);

    // entry(tid, bound, claim)
    let entry = BasicBlock::new(&mut ctx, None, vec![i32_ty.into(), i32_ty.into(), i1_ty.into()]);
    entry.insert_at_front(region, &ctx);
    let tid = entry.deref(&ctx).get_argument(0);
    let bound = entry.deref(&ctx).get_argument(1);
    let claim = entry.deref(&ctx).get_argument(2);

    let then_blk = BasicBlock::new(&mut ctx, None, vec![]);
    then_blk.insert_at_back(region, &ctx);
    let else_blk = BasicBlock::new(&mut ctx, None, vec![]);
    else_blk.insert_at_back(region, &ctx);
    let merge_blk = BasicBlock::new(&mut ctx, None, vec![]);
    merge_blk.insert_at_back(region, &ctx);

    let ptr_i32 = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);

    // entry: TILE = mir.shared_alloc; verify.assert(claim); %cond = tid < bound; cond_br
    let shared_alloc_op = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirSharedAllocOp::get_concrete_op_info(),
        vec![ptr_i32.into()],
        vec![],
        vec![],
        0,
    );
    let shared_alloc = dialect_mir::ops::MirSharedAllocOp::new(shared_alloc_op);
    shared_alloc.set_attr_elem_type(&mut ctx, TypeAttr::new(i32_ty.into()));
    shared_alloc.set_attr_size(
        &mut ctx,
        IntegerAttr::new(i32_ty, APInt::from_u32(256, std::num::NonZero::new(32).unwrap())),
    );
    shared_alloc_op.insert_at_back(entry, &ctx);
    let tile = shared_alloc_op.deref(&ctx).get_result(0);

    let assert_op = ghost_ops::VerifyAssertOp::build(&mut ctx, claim);
    assert_op.insert_at_back(entry, &ctx);

    let cond_op = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirLtOp::get_concrete_op_info(),
        vec![i1_ty.into()],
        vec![tid, bound],
        vec![],
        0,
    );
    cond_op.insert_at_back(entry, &ctx);
    let cond = cond_op.deref(&ctx).get_result(0);

    let (cond_flat, cond_sizes) =
        dialect_mir::ops::MirCondBranchOp::compute_segment_sizes(vec![vec![cond], vec![], vec![]]);
    let cond_br_op = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirCondBranchOp::get_concrete_op_info(),
        vec![],
        cond_flat,
        vec![then_blk, else_blk],
        0,
    );
    dialect_mir::ops::MirCondBranchOp::new(cond_br_op).set_operand_segment_sizes(&ctx, cond_sizes);
    cond_br_op.insert_at_back(entry, &ctx);

    // then: %p = ptr_offset(tile, tid); store(%p, tid); goto merge
    let off1 = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirPtrOffsetOp::get_concrete_op_info(),
        vec![ptr_i32.into()],
        vec![tile, tid],
        vec![],
        0,
    );
    off1.insert_at_back(then_blk, &ctx);
    let p1 = off1.deref(&ctx).get_result(0);
    let st1 = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![p1, tid],
        vec![],
        0,
    );
    st1.insert_at_back(then_blk, &ctx);
    let goto1 = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![merge_blk],
        0,
    );
    goto1.insert_at_back(then_blk, &ctx);

    // else: %z = mir.constant 0; %p2 = ptr_offset(tile, tid); store(%p2, %z); goto merge
    let zero_op = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirConstantOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    dialect_mir::ops::MirConstantOp::new(zero_op).set_attr_value(
        &mut ctx,
        IntegerAttr::new(i32_ty, APInt::from_u32(0, std::num::NonZero::new(32).unwrap())),
    );
    zero_op.insert_at_back(else_blk, &ctx);
    let zero = zero_op.deref(&ctx).get_result(0);
    let off2 = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirPtrOffsetOp::get_concrete_op_info(),
        vec![ptr_i32.into()],
        vec![tile, tid],
        vec![],
        0,
    );
    off2.insert_at_back(else_blk, &ctx);
    let p2 = off2.deref(&ctx).get_result(0);
    let st2 = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![p2, zero],
        vec![],
        0,
    );
    st2.insert_at_back(else_blk, &ctx);
    let goto2 = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![merge_blk],
        0,
    );
    goto2.insert_at_back(else_blk, &ctx);

    // merge: return
    let ret_op = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    ret_op.insert_at_back(merge_blk, &ctx);

    assert!(mir_func.verify(&ctx).is_ok(), "constructed function must verify");
    println!("Constructed a real dialect-mir function: entry -> {{then,else}} -> merge, with");
    println!("verify.assert + mir.shared_alloc + mir.ptr_offset + mir.store (if/else).\n");

    // --- Run the real translator ---
    let mut tr = Translator::new(&ctx);
    tr.bind(tid, "tid");
    tr.bind(bound, "bound");
    tr.bind(claim, "claim");
    let body = tr.translate_function_body(mir_func);

    println!("=== Translator output (body only, auto-generated) ===");
    print!("{body}");

    // Single-threaded here (ownership-under-concurrency is Phase 3's job):
    // the precondition just hands the whole tile's permission over via a
    // quantified `acc`, the same proven-working shape as Phase 0's
    // `bad_non_injective_redistribution` test.
    // Viper doesn't assume a Seq[Ref]'s elements are pairwise distinct on
    // its own (two different indices could alias the same Ref) -- Silicon
    // caught exactly this, correctly, on the first real run. Spelling out
    // distinctness explicitly is what makes `cells[i].val`'s quantified
    // permission well-formed.
    let signature = "method poc(cells: Seq[Ref], tid: Int, bound: Int, claim: Bool)\n  requires |cells| == 256\n  requires forall i: Int, j: Int :: 0 <= i && i < 256 && 0 <= j && j < 256 && i != j ==> cells[i] != cells[j]\n  requires forall i: Int :: 0 <= i && i < 256 ==> acc(cells[i].val)\n  requires 0 <= tid && tid < 256\n";

    let viper_dir = std::env::var("VIPER_POC_DIR").unwrap_or_else(|_| "../viper-poc".to_string());

    for (label, extra_requires, expect) in [
        ("pass", "  requires claim\n", "pass (claim is given)"),
        ("fail", "", "fail (claim is NOT given -- nothing justifies the assert)"),
    ] {
        let mut full = String::from("field val: Int\n\n");
        full.push_str(signature);
        full.push_str(extra_requires);
        full.push_str("{\n");
        full.push_str(&body);
        full.push_str("}\n");

        let path = format!("{viper_dir}/phase2_auto_{label}.vpr");
        std::fs::write(&path, &full).expect("write .vpr");
        println!("\n=== wrote {path} (expected: {expect}) ===");
    }

    if std::env::args().any(|a| a == "--run") {
        let z3 = std::env::var("VERUS_Z3")
            .unwrap_or_else(|_| format!("{}/.local/opt/verus-src/source/target-verus/release/z3", std::env::var("HOME").unwrap()));
        let jar = std::env::var("VIPER_JAR")
            .unwrap_or_else(|_| format!("{}/.local/opt/viper-tools/backends/viperserver.jar", std::env::var("HOME").unwrap()));
        for label in ["pass", "fail"] {
            let path = format!("{viper_dir}/phase2_auto_{label}.vpr");
            println!("\n--- running Silicon on {label} ---");
            let out = std::process::Command::new("java")
                .args(["-cp", &jar, "viper.silicon.SiliconRunner", "--z3Exe", &z3, &path])
                .output()
                .expect("run silicon");
            let stdout = String::from_utf8_lossy(&out.stdout);
            for line in stdout.lines().filter(|l| {
                l.contains("Verification successful")
                    || l.contains("error(s)")
                    || l.contains("might")
            }) {
                println!("{line}");
            }
        }
    } else {
        println!("\n(pass --run to actually invoke Silicon on the generated files)");
    }
}
