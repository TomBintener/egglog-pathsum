use criterion::{black_box, criterion_group, criterion_main, Criterion};
use egglog_experimental::app::OracleApp;
use rand::Rng;

/// Generates a dummy circuit string of a given size.
fn generate_dummy_circuit_string(size: usize) -> String {
    let mut payload = String::with_capacity(size * 10); // Pre-allocate
    let mut rng = rand::thread_rng();
    let gates = ["h,0", "cx,0,1", "t,1", "s,2", "z,3"];

    for _ in 0..size {
        payload.push_str(gates[rng.gen_range(0..gates.len())]);
        payload.push(',');
    }
    payload
}

fn benchmark_boilerplate_setup(c: &mut Criterion) {
    let mut group = c.benchmark_group("Rust Boilerplate Setup");

    // Test the 5 specific sizes
    for size in [100, 1_000, 25_000, 100_000, 1_000_000].iter() {
        // 1. Generate a dummy payload string of 'size' gates
        let payload = generate_dummy_circuit_string(*size);

        group.bench_function(format!("Setup {} gates", size), |b| {
            b.iter(|| {
                // 2. Measure ONLY the App initialization and payload prep
                let app = OracleApp::new();
                let configured_engine = app.orchestrate_setup(black_box(&payload));

                // IMPORTANT: Do NOT call .run() or .parse_and_run_program() here.
                // We are only benchmarking the Rust setup, not the execution.
                black_box(configured_engine);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_boilerplate_setup);
criterion_main!(benches);
