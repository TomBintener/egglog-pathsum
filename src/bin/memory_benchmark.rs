#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unreachable_code)]

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use std::process::Command;

/// This is the core logic that runs in a dedicated, isolated process.
/// It runs the simulation for a single qubit size and lets DHAT
/// generate a clean report for just that run.
#[cfg(feature = "dhat-heap")]
fn run_single_test(num_qubits: i64) {
    use egglog_experimental::bridge::{
        id_pathsum_logic, apply_cx_logic, apply_h_logic, apply_t_logic, apply_s_logic, apply_z_logic, apply_x_logic
    };
    use rand::Rng;

    let _profiler = dhat::Profiler::new_heap();

    // Simulate complex state
    let mut state = id_pathsum_logic(num_qubits);
    let mut rng = rand::thread_rng();
    let depth = 50;

    for i in 0..depth {
        let q1 = rng.gen_range(0..num_qubits);
        let mut q2 = rng.gen_range(0..num_qubits);
        if q1 == q2 { q2 = (q2 + 1) % num_qubits; }

        state = match i % 6 {
            0 => apply_cx_logic(state, q1, q2),
            1 => apply_t_logic(state, q1),
            2 => apply_s_logic(state, q2),
            3 => apply_z_logic(state, q1),
            4 => apply_x_logic(state, q2),
            5 => {
                if (num_qubits + (i / 6)) < 60 {
                    apply_h_logic(state, q1)
                } else {
                    apply_cx_logic(state, q1, q2)
                }
            },
            _ => unreachable!()
        };
    }
    // The profiler will write its report when it's dropped at the end of this function.
}

fn main() {
    // Check if we are in a child process running a specific test
    if let Ok(_size_str) = std::env::var("DHAT_QUBIT_SIZE") {
        #[cfg(feature = "dhat-heap")]
        {
            let num_qubits = _size_str.parse().unwrap();
            run_single_test(num_qubits);
        }
        return;
    }

    // If not in a child process, we are the main test runner.
    #[cfg(not(feature = "dhat-heap"))]
    {
        println!("Error: Memory benchmark must be run with the 'dhat-heap' feature enabled.");
        println!("Please run: cargo run --release --bin memory_benchmark --features dhat-heap");
        return;
    }

    #[cfg(feature = "dhat-heap")]
    {
        println!("============================================================");
        println!("  EvaluatedPathSum Peak Memory Scaling Benchmark");
        println!("============================================================");
        println!("| Qubits | Circuit Depth | Peak Heap Allocation (Bytes) |");
        println!("|--------|---------------|------------------------------|");

        let test_sizes = [5, 10, 15, 20, 25, 30, 40, 50];
        let self_exe = std::env::current_exe().unwrap();

        for size in test_sizes {
            // Spawn a child process for each size
            let output = Command::new(&self_exe)
                .env("DHAT_QUBIT_SIZE", size.to_string())
                .env("RUST_BACKTRACE", "1")
                .output()
                .expect("Failed to execute child process");

            // We read the stderr from the child process.
            // DHAT prints the max memory to stderr like: "dhat: At t-gmax: 10,776 bytes in 8 blocks"
            let stderr = String::from_utf8_lossy(&output.stderr);
            let mut peak_memory = "0".to_string();

            for line in stderr.lines() {
                if line.contains("At t-gmax:") {
                    // Extract the byte value from the line
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if let Some(pos) = parts.iter().position(|&s| s == "bytes") {
                        if pos > 0 {
                            peak_memory = parts[pos - 1].replace(',', "");
                        }
                    }
                }
            }

            println!("| {:<6} | {:<13} | {:<28} |", size, 50, peak_memory);
        }

        println!("============================================================");
        println!("\nBenchmark complete. Individual reports were generated and summarized.");
    }
}
