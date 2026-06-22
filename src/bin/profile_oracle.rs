use criterion::Criterion;
use egglog_experimental::bridge::{PSum64, PSum128};
use egglog_experimental::engine::{engine_64, engine_128};
use std::env;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::str::FromStr;
use std::time::Duration;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <path_to_file>", args[0]);
        return;
    }
    let path = &args[1];
    let file = File::open(path).unwrap();
    let lines: Vec<String> = io::BufReader::new(file)
        .lines()
        .map(|l| l.unwrap())
        .collect();
    let mut app = OracleApp::new(lines);

    let mut criterion = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(10);

    criterion.bench_function("oracle", |b| {
        b.iter(|| {
            app.run();
        })
    });

    criterion.final_summary();
}

pub struct OracleApp {
    lines: Vec<String>,
    state: PSum64,
}

impl OracleApp {
    pub fn new(lines: Vec<String>) -> Self {
        Self {
            lines,
            state: PSum64::new(engine_64::EvaluatedPathSum::new_id(0)),
        }
    }

    pub fn run(&mut self) {
        for line in &self.lines {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let gate = parts[0];
            let args: Vec<i64> = parts[1..].iter().map(|s| s.parse().unwrap()).collect();

            match gate {
                "h" => {
                    self.state = apply_h(self.state.clone(), args[0]);
                }
                "cx" => {
                    self.state = apply_cx(self.state.clone(), args[0], args[1]);
                }
                "t" => {
                    self.state = apply_t(self.state.clone(), args[0]);
                }
                "s" => {
                    self.state = apply_s(self.state.clone(), args[0]);
                }
                "z" => {
                    self.state = apply_z(self.state.clone(), args[0]);
                }
                "x" => {
                    self.state = apply_x(self.state.clone(), args[0]);
                }
                "sdg" => {
                    self.state = apply_sdg(self.state.clone(), args[0]);
                }
                "tdg" => {
                    self.state = apply_tdg(self.state.clone(), args[0]);
                }
                "sx" => {
                    self.state = apply_sx(self.state.clone(), args[0]);
                }
                "rz" => {
                    let theta = f64::from_str(parts[2]).unwrap();
                    self.state = apply_rz(self.state.clone(), args[0], theta.to_bits() as i64);
                }
                _ => {}
            }
        }
    }
}

fn apply_h(state: PSum64, q: i64) -> PSum64 {
    let mut new_state = (*state).clone();
    new_state.apply_h(q as usize);
    new_state.reduce();
    PSum64::new(new_state)
}

fn apply_cx(state: PSum64, qc: i64, qt: i64) -> PSum64 {
    let mut new_state = (*state).clone();
    new_state.apply_cx(qc as usize, qt as usize);
    new_state.reduce();
    PSum64::new(new_state)
}

fn apply_t(state: PSum64, q: i64) -> PSum64 {
    let mut new_state = (*state).clone();
    new_state.apply_t(q as usize);
    new_state.reduce();
    PSum64::new(new_state)
}

fn apply_s(state: PSum64, q: i64) -> PSum64 {
    let mut new_state = (*state).clone();
    new_state.apply_s(q as usize);
    new_state.reduce();
    PSum64::new(new_state)
}

fn apply_z(state: PSum64, q: i64) -> PSum64 {
    let mut new_state = (*state).clone();
    new_state.apply_z(q as usize);
    new_state.reduce();
    PSum64::new(new_state)
}

fn apply_x(state: PSum64, q: i64) -> PSum64 {
    let mut new_state = (*state).clone();
    new_state.apply_x(q as usize);
    new_state.reduce();
    PSum64::new(new_state)
}

fn apply_sdg(state: PSum64, q: i64) -> PSum64 {
    let mut new_state = (*state).clone();
    new_state.apply_sdg(q as usize);
    new_state.reduce();
    PSum64::new(new_state)
}

fn apply_tdg(state: PSum64, q: i64) -> PSum64 {
    let mut new_state = (*state).clone();
    new_state.apply_tdg(q as usize);
    new_state.reduce();
    PSum64::new(new_state)
}

fn apply_sx(state: PSum64, q: i64) -> PSum64 {
    let mut new_state = (*state).clone();
    new_state.apply_sx(q as usize);
    new_state.reduce();
    PSum64::new(new_state)
}

fn apply_rz(state: PSum64, q: i64, theta_bits: i64) -> PSum64 {
    let mut new_state = (*state).clone();
    new_state.apply_rz(q as usize, f64::from_bits(theta_bits as u64));
    PSum64::new(new_state)
}
