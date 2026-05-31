use crate::canonical_phase_poly::EvaluatedPathSum;
use egglog::prelude::BaseSort;
use egglog::sort::{BaseValues, Boxed};
use egglog::{add_primitive, EGraph, Value};
use egglog::ast::Literal;
use egglog::{TermId, TermDag};

/// The memory-safe wrapper for EvaluatedPathSum that is passed to egglog.
pub type PSum = Boxed<EvaluatedPathSum>;

/// The anchor struct that registers the "PathSum" sort with the egglog engine.
#[derive(Debug)]
pub struct PathSumSort;

// Helper functions for primitives.
// Extracting these avoids `add_primitive!` macro parsing errors with semicolons.
fn id_pathsum_logic(num_qubits: i64) -> PSum {
    PSum::new(EvaluatedPathSum::new_id(num_qubits as u32))
}

fn apply_x_logic(state: PSum, q: i64) -> PSum {
    let mut new_state = (*state).clone();
    new_state.apply_x(q as usize);
    PSum::new(new_state)
}

fn apply_z_logic(state: PSum, q: i64) -> PSum {
    let mut new_state = (*state).clone();
    new_state.apply_z(q as usize);
    PSum::new(new_state)
}

fn apply_s_logic(state: PSum, q: i64) -> PSum {
    let mut new_state = (*state).clone();
    new_state.apply_s(q as usize);
    PSum::new(new_state)
}

fn apply_t_logic(state: PSum, q: i64) -> PSum {
    let mut new_state = (*state).clone();
    new_state.apply_t(q as usize);
    PSum::new(new_state)
}

fn apply_cx_logic(state: PSum, qc: i64, qt: i64) -> PSum {
    let mut new_state = (*state).clone();
    new_state.apply_cx(qc as usize, qt as usize);
    PSum::new(new_state)
}

fn apply_h_logic(state: PSum, q: i64) -> PSum {
    let mut new_state = (*state).clone();
    new_state.apply_h(q as usize);
    new_state.reduce(); // Eagerly integrate out path variables
    PSum::new(new_state)
}

fn add_global_phase_logic(state: PSum, phase: i64) -> PSum {
    let mut new_state = (*state).clone();
    let phase_term = crate::canonical_phase_poly::PackedPhaseTerm::create(0, phase as u8);
    let phase_poly = crate::canonical_phase_poly::CanonicalPhasePoly { terms: smallvec::smallvec![phase_term] };
    new_state.phase_poly.add_assign(&phase_poly);
    PSum::new(new_state)
}

impl BaseSort for PathSumSort {
    type Base = PSum;

    fn name(&self) -> &str {
        "PathSum"
    }

    fn register_primitives(&self, eg: &mut EGraph) {
        add_primitive!(eg, "rust_id_pathsum" = |num_qubits: i64| -> PSum {
            id_pathsum_logic(num_qubits)
        });

        add_primitive!(eg, "rust_apply_x" = |state: PSum, q: i64| -> PSum {
            apply_x_logic(state, q)
        });

        add_primitive!(eg, "rust_apply_z" = |state: PSum, q: i64| -> PSum {
            apply_z_logic(state, q)
        });

        add_primitive!(eg, "rust_apply_s" = |state: PSum, q: i64| -> PSum {
            apply_s_logic(state, q)
        });

        add_primitive!(eg, "rust_apply_t" = |state: PSum, q: i64| -> PSum {
            apply_t_logic(state, q)
        });

        add_primitive!(eg, "rust_apply_cx" = |state: PSum, qc: i64, qt: i64| -> PSum {
            apply_cx_logic(state, qc, qt)
        });

        add_primitive!(eg, "rust_apply_h" = |state: PSum, q: i64| -> PSum {
            apply_h_logic(state, q)
        });

        add_primitive!(eg, "rust_add_global_phase" = |state: PSum, phase: i64| -> PSum {
            add_global_phase_logic(state, phase)
        });
    }

    fn reconstruct_termdag(&self, _base_values: &BaseValues, _value: Value, termdag: &mut TermDag) -> TermId {
        // Deferring full serialization for now. This allows the engine to compile
        // and run without crashing if extraction is triggered.
        termdag.lit(Literal::String("<Unextracted PathSum State>".into()))
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_bridge_initialization() {
        let mut eg = crate::new_experimental_egraph();

        let script = r#"
            (let state (rust_id_pathsum 2))
        "#;

        let result = eg.parse_and_run_program(None, script);
        assert!(result.is_ok(), "Failed to run rust_id_pathsum through egglog: {:?}", result);
    }

    #[test]
    fn test_bridge_clifford_gates() {
        let mut eg = crate::new_experimental_egraph();

        let script = r#"
            (let s0 (rust_id_pathsum 2))
            (let s1 (rust_apply_x s0 0))
            (let s2 (rust_apply_cx s1 0 1))
            (let s3 (rust_apply_z s2 1))
            (let s4 (rust_apply_s s3 0))
            (let s5 (rust_apply_t s4 1))
        "#;

        let result = eg.parse_and_run_program(None, script);
        assert!(result.is_ok(), "Failed to run Clifford sequence through egglog: {:?}", result);
    }

    #[test]
    fn test_hzh_is_x_in_egraph() {
        let mut eg = crate::new_experimental_egraph();

        let script = r#"
            (let s0 (rust_id_pathsum 1))

            ;; Path 1: H -> Z -> H
            (let s_hzh (rust_apply_h (rust_apply_z (rust_apply_h s0 0) 0) 0))

            ;; Path 2: Apply X directly
            (let s_x (rust_apply_x s0 0))

            ;; Check that the e-graph proves them equal.
            (check (= s_hzh s_x))
        "#;

        let result = eg.parse_and_run_program(None, script);
        assert!(result.is_ok(), "Failed to prove HZH = X in egraph: {:?}", result);
    }
}
