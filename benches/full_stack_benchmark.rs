use criterion::{criterion_group, criterion_main, Criterion};
use std::process::Command;
use std::path::PathBuf;

/// Finds the release binary for the `egglog-experimental` CLI.
fn get_cli_binary_path() -> PathBuf {
    let mut path = std::env::current_exe().unwrap();
    path.pop(); // remove test exe name
    if path.ends_with("deps") {
        path.pop();
    }
    path.join("egglog-experimental.exe")
}

fn benchmark_full_stack(c: &mut Criterion) {
    let mut group = c.benchmark_group("Full Stack (Python + Egglog + Rust)");
    let cli_path = get_cli_binary_path();
    let python_script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("run_egglog_test.py");

    // We use fewer gates here because each one invokes the full egglog engine,
    // which is much slower than the isolated boilerplate test.
    for size in [10, 100, 500].iter() {
        group.bench_function(format!("{} gates", size), |b| {
            b.iter(|| {
                let status = Command::new("python")
                    .arg(&python_script_path)
                    .arg("--size")
                    .arg(size.to_string())
                    .arg("--bin")
                    .arg(&cli_path)
                    .status()
                    .expect("Failed to execute Python script");

                assert!(status.success(), "Python script failed");
            });
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_full_stack);
criterion_main!(benches);
