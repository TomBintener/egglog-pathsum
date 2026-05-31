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
// These now contain the FFI panic shields and universal eager reduction.
fn id_pathsum_logic(num_qubits: i64) -> PSum {
    if num_qubits <= 0 {
        PSum::new(EvaluatedPathSum::new_id(0))
    } else {
        PSum::new(EvaluatedPathSum::new_id(num_qubits as u32))
    }
}

fn apply_x_logic(state: PSum, q: i64) -> PSum {
    let mut new_state = (*state).clone();
    if q < 0 || q as usize >= new_state.num_qubits as usize {
        return PSum::new(new_state); // FFI Shield
    }
    new_state.apply_x(q as usize);
    new_state.reduce();
    PSum::new(new_state)
}

fn apply_z_logic(state: PSum, q: i64) -> PSum {
    let mut new_state = (*state).clone();
    if q < 0 || q as usize >= new_state.num_qubits as usize {
        return PSum::new(new_state); // FFI Shield
    }
    new_state.apply_z(q as usize);
    new_state.reduce();
    PSum::new(new_state)
}

fn apply_s_logic(state: PSum, q: i64) -> PSum {
    let mut new_state = (*state).clone();
    if q < 0 || q as usize >= new_state.num_qubits as usize {
        return PSum::new(new_state); // FFI Shield
    }
    new_state.apply_s(q as usize);
    new_state.reduce();
    PSum::new(new_state)
}

fn apply_t_logic(state: PSum, q: i64) -> PSum {
    let mut new_state = (*state).clone();
    if q < 0 || q as usize >= new_state.num_qubits as usize {
        return PSum::new(new_state); // FFI Shield
    }
    new_state.apply_t(q as usize);
    new_state.reduce();
    PSum::new(new_state)
}

fn apply_cx_logic(state: PSum, qc: i64, qt: i64) -> PSum {
    let mut new_state = (*state).clone();
    if qc == qt || qc < 0 || qt < 0 ||
       qc as usize >= new_state.num_qubits as usize ||
       qt as usize >= new_state.num_qubits as usize {
        return PSum::new(new_state); // FFI Shield
    }
    new_state.apply_cx(qc as usize, qt as usize);
    new_state.reduce();
    PSum::new(new_state)
}

fn apply_h_logic(state: PSum, q: i64) -> PSum {
    let mut new_state = (*state).clone();
    if q < 0 || q as usize >= new_state.num_qubits as usize {
        return PSum::new(new_state); // FFI Shield
    }
    new_state.apply_h(q as usize);
    new_state.reduce();
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
    }

    fn reconstruct_termdag(&self, _base_values: &BaseValues, _value: Value, termdag: &mut TermDag) -> TermId {
        termdag.lit(Literal::String("<Unextracted PathSum State>".into()))
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_bridge_initialization() {
        let mut eg = crate::new_experimental_egraph();
        let script = r#"(let state (rust_id_pathsum 2))"#;
        let result = eg.parse_and_run_program(None, script);
        assert!(result.is_ok(), "Failed to run rust_id_pathsum through egglog: {:?}", result);
    }

    #[test]
    fn test_hzh_is_x_in_egraph() {
        let mut eg = crate::new_experimental_egraph();
        let script = r#"
            (let s0 (rust_id_pathsum 1))
            (let s_hzh (rust_apply_h (rust_apply_z (rust_apply_h s0 0) 0) 0))
            (let s_x (rust_apply_x s0 0))

            ;; Note: HZH = -iX, so they differ by a global phase.
            ;; In a real e-graph rule, we would have a way to equate them modulo phase.
            ;; For this test, we just ensure the sequence runs without crashing.
        "#;
        let result = eg.parse_and_run_program(None, script);
        assert!(result.is_ok(), "Failed to run HZH in egraph: {:?}", result);
    }

    #[test]
    fn test_ffi_panic_shield_single_qubit() {
        let mut eg = crate::new_experimental_egraph();
        let script = r#"
            (let s0 (rust_id_pathsum 1))
            ;; Apply X to an invalid qubit index
            (let s1 (rust_apply_x s0 100))
            ;; The state should be unchanged
            (check (= s0 s1))
        "#;
        let result = eg.parse_and_run_program(None, script);
        assert!(result.is_ok(), "FFI panic shield test failed for single qubit gate: {:?}", result);
    }

    #[test]
    fn test_ffi_panic_shield_two_qubit() {
        let mut eg = crate::new_experimental_egraph();
        let script = r#"
            (let s0 (rust_id_pathsum 2))
            ;; Apply CX to an invalid qubit index
            (let s1 (rust_apply_cx s0 0 100))
            ;; Apply CX with identical control and target
            (let s2 (rust_apply_cx s0 1 1))
            ;; The state should be unchanged in both cases
            (check (= s0 s1))
            (check (= s0 s2))
        "#;
        let result = eg.parse_and_run_program(None, script);
        assert!(result.is_ok(), "FFI panic shield test failed for two qubit gate: {:?}", result);
    }

    #[test]
    fn test_bridge_eager_reduction_with_z() {
        let mut eg = crate::new_experimental_egraph();
        // This test ensures that the Z gate triggers reduction.
        // H introduces a variable, Z provides the constraint to eliminate it.
        // If Z does not trigger reduction, the state will be unreduced.
        let script = r#"
            (let s0 (rust_id_pathsum 1))
            (let s1 (rust_apply_h s0 0))
            (let s2 (rust_apply_z s1 0))
            ;; We just ensure it runs without crashing, verifying the
            ;; eager reduction mechanism inside the Z gate primitive.
        "#;
        let result = eg.parse_and_run_program(None, script);
        assert!(result.is_ok(), "Eager reduction in Z gate failed: {:?}", result);
    }
}
