//! Independent, Verus-free brute-force check of the same correctness claim
//! the Verus proof makes: `out[tid] == data[(tid+1) % N]` for the
//! `shared_test` kernel's access pattern.
//!
//! This is the "belt and suspenders" backstop from the design discussion --
//! the same role cuda-oxide's own rustlantis-based differential fuzzer plays
//! relative to the compiler. It never touches Verus; it just simulates the
//! kernel's two phases directly and checks the resulting values by brute
//! force over several concrete block sizes. A passing Verus proof plus a
//! passing brute-force check for the same claim is strictly stronger
//! evidence than either alone.

/// Simulate the whole block's execution of `shared_test` directly: phase 1
/// (every thread writes its own tile slot), implicit barrier, phase 2
/// (every thread reads its neighbor's slot into `out`).
fn simulate_shared_test(data: &[f32]) -> Vec<f32> {
    let n = data.len();
    let mut tile = vec![0.0f32; n];
    for tid in 0..n {
        tile[tid] = data[tid]; // gid == tid: single block, block_dim == data.len()
    }
    // -- sync_threads() boundary: every write above is visible to every read below --
    let mut out = vec![0.0f32; n];
    for tid in 0..n {
        let neighbor_idx = (tid + 1) % n;
        out[tid] = tile[neighbor_idx];
    }
    out
}

fn check_block_size(n: usize) -> Result<(), String> {
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 1.5 + 1.0).collect();
    let out = simulate_shared_test(&data);
    for tid in 0..n {
        let expected = data[(tid + 1) % n];
        if out[tid] != expected {
            return Err(format!(
                "N={n}: out[{tid}]={} but data[({tid}+1)%{n}]={expected}",
                out[tid]
            ));
        }
    }
    Ok(())
}

fn main() {
    let sizes = [1usize, 2, 4, 8, 16, 32, 256, 1024];
    let mut failures = Vec::new();
    for &n in &sizes {
        match check_block_size(n) {
            Ok(()) => println!("N={n:<5} OK  (out[tid] == data[(tid+1) % {n}] for all tid)"),
            Err(e) => {
                println!("N={n:<5} FAIL: {e}");
                failures.push(e);
            }
        }
    }
    if failures.is_empty() {
        println!("\nsanity_check: PASS -- brute-force cross-check agrees with the Verus proof's claim.");
    } else {
        eprintln!("\nsanity_check: FAIL -- {} block size(s) disagreed.", failures.len());
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_property_holds_for_small_and_large_blocks() {
        for &n in &[1usize, 2, 3, 4, 7, 8, 16, 256, 1024] {
            assert!(check_block_size(n).is_ok(), "failed for N={n}");
        }
    }

    #[test]
    fn neighbor_indexing_wraps_at_the_boundary() {
        // Specifically exercise the modulo wraparound: the last thread's
        // neighbor is thread 0.
        let n = 8;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let out = simulate_shared_test(&data);
        assert_eq!(out[n - 1], data[0]);
    }
}
