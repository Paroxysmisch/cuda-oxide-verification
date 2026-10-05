//! Stage 0 — the kernel exactly as cuda-oxide/cuda-rust compiles it today.
//!
//! This is a faithful, line-for-line reproduction of the real `shared_test`
//! kernel from
//! `cuda-rust/cuda-oxide/crates/rustc-codegen-cuda/examples/sharedmem/src/main.rs`.
//! It has **zero** verification annotations — it is exactly what a cuda-oxide
//! author writes and the real compiler accepts, `unsafe` blocks and all.
//!
//! The types below (`SharedArray`, `ThreadIndex`, `DisjointSlice`, `thread::*`)
//! are local stand-ins that mirror the real `cuda-device` crate's public
//! signatures (including its `#[inline(never)] unreachable!()` stub
//! convention for device intrinsics — see
//! `cuda-rust/cuda-oxide/crates/cuda-device/src/thread.rs`). This file is
//! never executed: it exists purely as the unmodified "before" snapshot that
//! stages 1 and 2 build on.

#![allow(dead_code, unused_variables, unreachable_code, non_snake_case)]

/// Stand-in for `cuda_device::SharedArray<T, N>` (see `cuda-device/src/shared.rs:116`).
pub struct SharedArray<T, const N: usize> {
    _marker: core::marker::PhantomData<core::cell::UnsafeCell<[T; N]>>,
}

impl<T, const N: usize> SharedArray<T, N> {
    pub const UNINIT: Self = SharedArray { _marker: core::marker::PhantomData };
}

impl<T, const N: usize> core::ops::Index<usize> for SharedArray<T, N> {
    type Output = T;
    fn index(&self, _i: usize) -> &T {
        unreachable!("indexing replaced by the real backend's addrspace(3) load")
    }
}
impl<T, const N: usize> core::ops::IndexMut<usize> for SharedArray<T, N> {
    fn index_mut(&mut self, _i: usize) -> &mut T {
        unreachable!("indexing replaced by the real backend's addrspace(3) store")
    }
}

/// Stand-in for `cuda_device::ThreadIndex` (see `cuda-device/src/thread.rs:294`).
pub struct ThreadIndex;
impl ThreadIndex {
    pub fn get(&self) -> usize {
        unreachable!("resolved by the real backend from threadIdx/blockIdx")
    }
}

/// Stand-in for `cuda_device::DisjointSlice<'a, T>` (see `cuda-device/src/disjoint.rs:179`).
pub struct DisjointSlice<'a, T> {
    _marker: core::marker::PhantomData<&'a mut [T]>,
}
impl<'a, T> DisjointSlice<'a, T> {
    pub fn get_mut(&mut self, _idx: ThreadIndex) -> Option<&mut T> {
        unreachable!("bounds-checked by the real backend")
    }
}

/// Stand-in for `cuda_device::thread` (see `cuda-device/src/thread.rs`).
pub mod thread {
    use super::ThreadIndex;

    #[inline(never)]
    pub fn threadIdx_x() -> u32 {
        unreachable!("threadIdx_x called outside CUDA kernel context")
    }

    #[inline(never)]
    pub fn index_1d<'kernel>() -> ThreadIndex {
        unreachable!("index_1d called outside CUDA kernel context")
    }

    #[inline(never)]
    pub fn sync_threads() {
        // Replaced by the generated CTA barrier (`bar.sync 0`) during device compilation.
        unreachable!("sync_threads called outside CUDA kernel context")
    }
}

/// Verbatim reproduction of `shared_test` from the real `sharedmem` example.
/// No `#[kernel]`/`#[cuda_module]` macros here (they're cuda-oxide-specific
/// proc macros with no bearing on verification) -- just the kernel body,
/// unchanged.
pub fn shared_test(data: &[f32], mut out: DisjointSlice<f32>) {
    static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;

    let tid = thread::threadIdx_x() as usize;
    let gid = thread::index_1d().get();

    // Write to shared memory.
    unsafe {
        TILE[tid] = data[gid];
    }

    thread::sync_threads();

    // Read from shared memory (neighbor).
    unsafe {
        let neighbor_idx = (tid + 1) % 256;
        if let Some(out_elem) = out.get_mut(thread::index_1d()) {
            *out_elem = TILE[neighbor_idx];
        }
    }
}
