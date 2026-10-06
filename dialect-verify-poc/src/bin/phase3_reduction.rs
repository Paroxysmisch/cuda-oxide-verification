//! Phase 3: a real `dialect-mir` function for one generic thread's path
//! through the classic stride-halving block-sum reduction --
//! ```text
//! let mut stride = N / 2;
//! while stride >= 1 {
//!     if tid < stride { TILE[tid] += TILE[tid + stride]; }
//!     sync_threads();
//!     stride /= 2;
//! }
//! ```
//! with N fixed at 8 for tractability -- built with a REAL loop (a genuine
//! back edge in the CFG, not unrolled) and a real `mir.alloca`-backed
//! mutable local for `stride` (the honest pre-`mem2reg` shape, per
//! `MirAllocaOp`'s own doc comment: "every Rust MIR local is backed by an
//! alloca... after mem2reg these are erased").
//!
//! Scope, named plainly (see NOTES.md for the full discussion): this
//! verifies ONE generic thread's permission bookkeeping -- that thread
//! `tid` only ever touches cells it has justified access to, across every
//! loop iteration, with no unrolling. It does NOT re-derive the separate
//! argument (flagged as the crucial step all the way back in this
//! project's very first Verus-based plan) that combines N such per-thread
//! proofs via separating conjunction to get a whole-block race-freedom
//! guarantee -- nor does it prove the reduction's numeric correctness
//! (the strided/butterfly sum formula). Both are real further work, not
//! silently assumed here.
//!
//! The partner cell's permission (`cells[tid + stride]`, needed only for
//! the duration of the guarded add) is modeled as a local
//! acquire-then-release bracketing the add, via two `verify.barrier`
//! ghost ops -- the mechanical analogue of `sync_threads()`'s redistribution,
//! scoped down to what this one thread's proof actually needs.

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
    let ptr_i32 = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);

    let func_ty = FunctionType::get(&ctx, vec![i32_ty.into()], vec![]);
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

    let entry = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    entry.insert_at_front(region, &ctx);
    let tid = entry.deref(&ctx).get_argument(0);

    let header = BasicBlock::new(&mut ctx, None, vec![]);
    header.insert_at_back(region, &ctx);
    let body = BasicBlock::new(&mut ctx, None, vec![]);
    body.insert_at_back(region, &ctx);
    let do_add = BasicBlock::new(&mut ctx, None, vec![]);
    do_add.insert_at_back(region, &ctx);
    let after_add = BasicBlock::new(&mut ctx, None, vec![]);
    after_add.insert_at_back(region, &ctx);
    let exit = BasicBlock::new(&mut ctx, None, vec![]);
    exit.insert_at_back(region, &ctx);

    // ---- entry: tile = shared_alloc; stride_ptr = alloca; store(stride_ptr, 4); goto header
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
        IntegerAttr::new(i32_ty, APInt::from_u32(8, std::num::NonZero::new(32).unwrap())),
    );
    shared_alloc_op.insert_at_back(entry, &ctx);
    let tile = shared_alloc_op.deref(&ctx).get_result(0);

    let alloca_op = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirAllocaOp::get_concrete_op_info(),
        vec![ptr_i32.into()],
        vec![],
        vec![],
        0,
    );
    alloca_op.insert_at_back(entry, &ctx);
    let stride_ptr = alloca_op.deref(&ctx).get_result(0);

    let c4_op = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirConstantOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    dialect_mir::ops::MirConstantOp::new(c4_op).set_attr_value(
        &mut ctx,
        IntegerAttr::new(i32_ty, APInt::from_u32(4, std::num::NonZero::new(32).unwrap())),
    );
    c4_op.insert_at_back(entry, &ctx);
    let c4 = c4_op.deref(&ctx).get_result(0);

    let store_init = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![stride_ptr, c4],
        vec![],
        0,
    );
    store_init.insert_at_back(entry, &ctx);

    let goto_header = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![header],
        0,
    );
    goto_header.insert_at_back(entry, &ctx);

    // ---- header: s0 = load(stride_ptr); verify.invariant(s0 >= 0); cond = s0 >= 1; cond_br cond, body, exit
    let load_s0 = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirLoadOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![stride_ptr],
        vec![],
        0,
    );
    load_s0.insert_at_back(header, &ctx);
    let s0 = load_s0.deref(&ctx).get_result(0);

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
    zero_op.insert_at_back(header, &ctx);
    let zero = zero_op.deref(&ctx).get_result(0);

    let ge0_op = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirGeOp::get_concrete_op_info(),
        vec![i1_ty.into()],
        vec![s0, zero],
        vec![],
        0,
    );
    ge0_op.insert_at_back(header, &ctx);
    let ge0 = ge0_op.deref(&ctx).get_result(0);

    let inv_op = ghost_ops::VerifyInvariantOp::build(&mut ctx, ge0);
    inv_op.insert_at_back(header, &ctx);

    // The bound that actually makes the guarded partner access safe:
    // whenever `tid < stride` and `stride <= 4`, `tid + stride < 2*stride
    // <= 8` -- in bounds. Without this, Silicon correctly rejects the
    // partner access as a possible out-of-bounds index (confirmed for
    // real: the first run of this kernel, with only `stride >= 0` as the
    // invariant, failed with exactly that diagnostic).
    let four_op = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirConstantOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    dialect_mir::ops::MirConstantOp::new(four_op).set_attr_value(
        &mut ctx,
        IntegerAttr::new(i32_ty, APInt::from_u32(4, std::num::NonZero::new(32).unwrap())),
    );
    four_op.insert_at_back(header, &ctx);
    let four = four_op.deref(&ctx).get_result(0);

    let le4_op = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirLeOp::get_concrete_op_info(),
        vec![i1_ty.into()],
        vec![s0, four],
        vec![],
        0,
    );
    le4_op.insert_at_back(header, &ctx);
    let le4 = le4_op.deref(&ctx).get_result(0);

    let inv_op2 = ghost_ops::VerifyInvariantOp::build(&mut ctx, le4);
    inv_op2.insert_at_back(header, &ctx);

    // Permissions need to be carried across the loop boundary explicitly
    // too, exactly like any value-level fact -- confirmed for real: the
    // first run with only the two value invariants above failed with
    // "insufficient permission to access cells[tid].val" at the store.
    let perm_inv = ghost_ops::VerifyPermInvariantOp::build(&mut ctx, "acc(cells[tid].val)");
    perm_inv.insert_at_back(header, &ctx);

    let one_op = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirConstantOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    dialect_mir::ops::MirConstantOp::new(one_op).set_attr_value(
        &mut ctx,
        IntegerAttr::new(i32_ty, APInt::from_u32(1, std::num::NonZero::new(32).unwrap())),
    );
    one_op.insert_at_back(header, &ctx);
    let one = one_op.deref(&ctx).get_result(0);

    let cond_op = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirGeOp::get_concrete_op_info(),
        vec![i1_ty.into()],
        vec![s0, one],
        vec![],
        0,
    );
    cond_op.insert_at_back(header, &ctx);
    let cond = cond_op.deref(&ctx).get_result(0);

    let (cond_flat, cond_sizes) =
        dialect_mir::ops::MirCondBranchOp::compute_segment_sizes(vec![vec![cond], vec![], vec![]]);
    let cond_br_op = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirCondBranchOp::get_concrete_op_info(),
        vec![],
        cond_flat,
        vec![body, exit],
        0,
    );
    dialect_mir::ops::MirCondBranchOp::new(cond_br_op).set_operand_segment_sizes(&ctx, cond_sizes);
    cond_br_op.insert_at_back(header, &ctx);

    // ---- body: s2 = load(stride_ptr); active = tid < s2; cond_br active, do_add, after_add
    let load_s2 = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirLoadOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![stride_ptr],
        vec![],
        0,
    );
    load_s2.insert_at_back(body, &ctx);
    let s2 = load_s2.deref(&ctx).get_result(0);

    let active_op = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirLtOp::get_concrete_op_info(),
        vec![i1_ty.into()],
        vec![tid, s2],
        vec![],
        0,
    );
    active_op.insert_at_back(body, &ctx);
    let active = active_op.deref(&ctx).get_result(0);

    let (active_flat, active_sizes) = dialect_mir::ops::MirCondBranchOp::compute_segment_sizes(
        vec![vec![active], vec![], vec![]],
    );
    let active_br = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirCondBranchOp::get_concrete_op_info(),
        vec![],
        active_flat,
        vec![do_add, after_add],
        0,
    );
    dialect_mir::ops::MirCondBranchOp::new(active_br).set_operand_segment_sizes(&ctx, active_sizes);
    active_br.insert_at_back(body, &ctx);

    // ---- do_add: s3 = load(stride_ptr); partner = tid + s3;
    //              verify.barrier(acquire partner);
    //              a = load(tile+tid); b = load(tile+partner); c = a+b; store(tile+tid, c);
    //              verify.barrier(release partner); goto after_add
    let load_s3 = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirLoadOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![stride_ptr],
        vec![],
        0,
    );
    load_s3.insert_at_back(do_add, &ctx);
    let s3 = load_s3.deref(&ctx).get_result(0);

    let partner_op = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirAddOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![tid, s3],
        vec![],
        0,
    );
    partner_op.insert_at_back(do_add, &ctx);
    let mut partner = partner_op.deref(&ctx).get_result(0);

    // --broken: inject a one-line off-by-one bug -- partner = tid+stride+1
    // instead of tid+stride. At tid=stride-1 (the largest active thread
    // this round) with stride=4, that's cells[8]: one past the real tile.
    if std::env::args().any(|a| a == "--broken") {
        let one_bug_op = Operation::new(
            &mut ctx,
            dialect_mir::ops::MirConstantOp::get_concrete_op_info(),
            vec![i32_ty.into()],
            vec![],
            vec![],
            0,
        );
        dialect_mir::ops::MirConstantOp::new(one_bug_op).set_attr_value(
            &mut ctx,
            IntegerAttr::new(i32_ty, APInt::from_u32(1, std::num::NonZero::new(32).unwrap())),
        );
        one_bug_op.insert_at_back(do_add, &ctx);
        let one_bug = one_bug_op.deref(&ctx).get_result(0);
        let off_by_one_op = Operation::new(
            &mut ctx,
            dialect_mir::ops::MirAddOp::get_concrete_op_info(),
            vec![i32_ty.into()],
            vec![partner, one_bug],
            vec![],
            0,
        );
        off_by_one_op.insert_at_back(do_add, &ctx);
        partner = off_by_one_op.deref(&ctx).get_result(0);
        println!("*** --broken: injected off-by-one bug (partner = tid + stride + 1) ***\n");
    }

    let acquire = ghost_ops::VerifyBarrierOp::build(
        &mut ctx,
        vec![partner],
        "true",
        "acc(cells[$0].val)",
    );
    acquire.insert_at_back(do_add, &ctx);

    let a_ptr_op = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirPtrOffsetOp::get_concrete_op_info(),
        vec![ptr_i32.into()],
        vec![tile, tid],
        vec![],
        0,
    );
    a_ptr_op.insert_at_back(do_add, &ctx);
    let a_ptr = a_ptr_op.deref(&ctx).get_result(0);

    let a_load = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirLoadOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![a_ptr],
        vec![],
        0,
    );
    a_load.insert_at_back(do_add, &ctx);
    let a_val = a_load.deref(&ctx).get_result(0);

    let b_ptr_op = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirPtrOffsetOp::get_concrete_op_info(),
        vec![ptr_i32.into()],
        vec![tile, partner],
        vec![],
        0,
    );
    b_ptr_op.insert_at_back(do_add, &ctx);
    let b_ptr = b_ptr_op.deref(&ctx).get_result(0);

    let b_load = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirLoadOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![b_ptr],
        vec![],
        0,
    );
    b_load.insert_at_back(do_add, &ctx);
    let b_val = b_load.deref(&ctx).get_result(0);

    let sum_op = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirAddOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![a_val, b_val],
        vec![],
        0,
    );
    sum_op.insert_at_back(do_add, &ctx);
    let sum = sum_op.deref(&ctx).get_result(0);

    let store_sum = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![a_ptr, sum],
        vec![],
        0,
    );
    store_sum.insert_at_back(do_add, &ctx);

    let release = ghost_ops::VerifyBarrierOp::build(
        &mut ctx,
        vec![partner],
        "acc(cells[$0].val)",
        "true",
    );
    release.insert_at_back(do_add, &ctx);

    let goto_after_add_1 = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![after_add],
        0,
    );
    goto_after_add_1.insert_at_back(do_add, &ctx);

    // ---- after_add: s4 = load(stride_ptr); half = s4 / 2; store(stride_ptr, half); goto header (back edge)
    let load_s4 = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirLoadOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![stride_ptr],
        vec![],
        0,
    );
    load_s4.insert_at_back(after_add, &ctx);
    let s4 = load_s4.deref(&ctx).get_result(0);

    let two_op = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirConstantOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    dialect_mir::ops::MirConstantOp::new(two_op).set_attr_value(
        &mut ctx,
        IntegerAttr::new(i32_ty, APInt::from_u32(2, std::num::NonZero::new(32).unwrap())),
    );
    two_op.insert_at_back(after_add, &ctx);
    let two = two_op.deref(&ctx).get_result(0);

    let half_op = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirDivOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![s4, two],
        vec![],
        0,
    );
    half_op.insert_at_back(after_add, &ctx);
    let half = half_op.deref(&ctx).get_result(0);

    let store_half = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![stride_ptr, half],
        vec![],
        0,
    );
    store_half.insert_at_back(after_add, &ctx);

    let goto_back = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![header],
        0,
    );
    goto_back.insert_at_back(after_add, &ctx);

    // ---- exit: return
    let ret_op = Operation::new(
        &mut ctx,
        dialect_mir::ops::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    ret_op.insert_at_back(exit, &ctx);

    assert!(mir_func.verify(&ctx).is_ok(), "constructed function must verify");
    println!("Constructed a real dialect-mir function: entry -> header -> {{body -> {{do_add ->}} after_add -> header (back edge) }} -> exit");
    println!("A genuine loop (CFG back edge), mir.alloca-backed `stride` local, and two verify.barrier ops bracketing the guarded add.\n");

    let mut tr = Translator::new(&ctx);
    tr.bind(tid, "tid");
    let body_text = tr.translate_function_body(mir_func);

    println!("=== Translator output (auto-generated from the real IR above) ===");
    print!("{body_text}");

    let signature = "method poc(cells: Seq[Ref], tid: Int)\n  requires |cells| == 8\n  requires forall i: Int, j: Int :: 0 <= i && i < 8 && 0 <= j && j < 8 && i != j ==> cells[i] != cells[j]\n  requires 0 <= tid && tid < 8\n  requires acc(cells[tid].val)\n  ensures acc(cells[tid].val)\n";

    let mut full = String::from("field val: Int\n\n");
    full.push_str(signature);
    full.push_str("{\n");
    full.push_str(&body_text);
    full.push_str("}\n");

    let viper_dir = std::env::var("VIPER_POC_DIR").unwrap_or_else(|_| "../viper-poc".to_string());
    let suffix = if std::env::args().any(|a| a == "--broken") { "_broken" } else { "" };
    let path = format!("{viper_dir}/phase3_reduction{suffix}.vpr");
    std::fs::write(&path, &full).expect("write .vpr");
    println!("\n=== wrote {path} ===");

    if std::env::args().any(|a| a == "--run") {
        let z3 = std::env::var("VERUS_Z3").unwrap_or_else(|_| {
            format!(
                "{}/.local/opt/verus-src/source/target-verus/release/z3",
                std::env::var("HOME").unwrap()
            )
        });
        let jar = std::env::var("VIPER_JAR").unwrap_or_else(|_| {
            format!("{}/.local/opt/viper-tools/backends/viperserver.jar", std::env::var("HOME").unwrap())
        });
        println!("\n--- running Silicon ---");
        let out = std::process::Command::new("java")
            .args(["-cp", &jar, "viper.silicon.SiliconRunner", "--z3Exe", &z3, &path])
            .output()
            .expect("run silicon");
        let stdout = String::from_utf8_lossy(&out.stdout);
        println!("{stdout}");
    } else {
        println!("(pass --run to actually invoke Silicon)");
    }
}
