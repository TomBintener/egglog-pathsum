//! Foreign Function Interface (FFI) Bridge for Egglog Integration.
//!
//! This module connects the high-performance Rust implementation of the path sum
//! (`EvaluatedPathSum`) to the `egglog` equality saturation engine. It defines a
//! custom `egglog` Sort called `PathSum` and registers primitives that map
//! `egglog` function calls (e.g., `rust_apply_cx_ffi`) directly to the
//! corresponding Rust methods.
//!
//! # FFI Panic Shields
//! Since `egglog` explores many possible circuit rewrites simultaneously, it might
//! temporarily generate invalid gate applications (e.g., applying a gate to an
//! out-of-bounds qubit). The functions in this bridge include "panic shields"
//! that gracefully catch these invalid operations and return the unmodified state
//! rather than crashing the Rust runtime.
//!
//! # Eager Reduction
//! Every gate application logic function eagerly calls `reduce()` on the state
//! after applying the gate. This enforces that the path sum is maintained in its
//! canonical form at all times within the e-graph, which is necessary for
//! the engine to correctly identify equivalent quantum states.

use crate::canonical_phase_poly::EvaluatedPathSum;
use egglog::prelude::BaseSort;
use egglog::sort::{BaseValues, Boxed, S};
use egglog::{add_primitive, EGraph, Value};
use egglog::ast::Literal;
use egglog::{TermId, TermDag};

/// The memory-safe wrapper for `EvaluatedPathSum` that is passed to `egglog`.
///
/// `Boxed` ensures that the `egglog` runtime can safely manage the lifecycle of
/// this complex Rust type.
pub type PSum = Boxed<EvaluatedPathSum>;

/// The anchor struct that registers the "PathSum" sort with the `egglog` engine.
#[derive(Debug)]
pub struct PathSumSort;

// Helper functions for primitives.
// These now contain the FFI panic shields and universal eager reduction.

/// Creates an initial identity path sum state with the given number of qubits.
///
/// If `num_qubits` is less than or equal to 0, it safely defaults to a 0-qubit state.
fn id_pathsum_logic(num_qubits: i64) -> PSum {
    if num_qubits <= 0 {
        PSum::new(EvaluatedPathSum::new_id(0))
    } else {
        PSum::new(EvaluatedPathSum::new_id(num_qubits as u32))
    }
}

/// Applies an X gate to the path sum state at the specified qubit.
///
/// Includes an FFI panic shield: if `q` is out of bounds, the state is returned unchanged.
/// Eagerly reduces the state after application to maintain canonicity.
fn apply_x_logic(state: PSum, q: i64) -> PSum {
    let mut new_state = (*state).clone();
    if q < 0 || q as usize >= new_state.num_qubits as usize {
        return PSum::new(new_state); // FFI Shield
    }
    new_state.apply_x(q as usize);
    new_state.reduce();
    PSum::new(new_state)
}

/// Applies a Z gate to the path sum state at the specified qubit.
///
/// Includes an FFI panic shield: if `q` is out of bounds, the state is returned unchanged.
/// Eagerly reduces the state after application to maintain canonicity.
fn apply_z_logic(state: PSum, q: i64) -> PSum {
    let mut new_state = (*state).clone();
    if q < 0 || q as usize >= new_state.num_qubits as usize {
        return PSum::new(new_state); // FFI Shield
    }
    new_state.apply_z(q as usize);
    new_state.reduce();
    PSum::new(new_state)
}

/// Applies an S gate to the path sum state at the specified qubit.
///
/// Includes an FFI panic shield: if `q` is out of bounds, the state is returned unchanged.
/// Eagerly reduces the state after application to maintain canonicity.
fn apply_s_logic(state: PSum, q: i64) -> PSum {
    let mut new_state = (*state).clone();
    if q < 0 || q as usize >= new_state.num_qubits as usize {
        return PSum::new(new_state); // FFI Shield
    }
    new_state.apply_s(q as usize);
    new_state.reduce();
    PSum::new(new_state)
}

/// Applies an S-dagger gate to the path sum state at the specified qubit.
fn apply_sdg_logic(state: PSum, q: i64) -> PSum {
    let mut new_state = (*state).clone();
    if q < 0 || q as usize >= new_state.num_qubits as usize {
        return PSum::new(new_state); // FFI Shield
    }
    new_state.apply_sdg(q as usize);
    new_state.reduce();
    PSum::new(new_state)
}

/// Applies a T gate to the path sum state at the specified qubit.
///
/// Includes an FFI panic shield: if `q` is out of bounds, the state is returned unchanged.
/// Eagerly reduces the state after application to maintain canonicity.
fn apply_t_logic(state: PSum, q: i64) -> PSum {
    let mut new_state = (*state).clone();
    if q < 0 || q as usize >= new_state.num_qubits as usize {
        return PSum::new(new_state); // FFI Shield
    }
    new_state.apply_t(q as usize);
    new_state.reduce();
    PSum::new(new_state)
}

/// Applies a T-dagger gate to the path sum state at the specified qubit.
fn apply_tdg_logic(state: PSum, q: i64) -> PSum {
    let mut new_state = (*state).clone();
    if q < 0 || q as usize >= new_state.num_qubits as usize {
        return PSum::new(new_state); // FFI Shield
    }
    new_state.apply_tdg(q as usize);
    new_state.reduce();
    PSum::new(new_state)
}

/// Applies a square-root-of-X gate to the path sum state at the specified qubit.
fn apply_sx_logic(state: PSum, q: i64) -> PSum {
    let mut new_state = (*state).clone();
    if q < 0 || q as usize >= new_state.num_qubits as usize {
        return PSum::new(new_state); // FFI Shield
    }
    new_state.apply_sx(q as usize);
    new_state.reduce();
    PSum::new(new_state)
}

/// Applies a Controlled-NOT (CX) gate between a control and target qubit.
///
/// Includes FFI panic shields: if the control and target are the same,
/// or if either is out of bounds, the state is returned unchanged.
/// Eagerly reduces the state after application to maintain canonicity.
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

/// Applies a Hadamard (H) gate to the path sum state at the specified qubit.
///
/// Includes an FFI panic shield: if `q` is out of bounds, the state is returned unchanged.
/// Eagerly reduces the state after application to integrate out temporary path variables
/// and maintain canonicity.
fn apply_h_logic(state: PSum, q: i64) -> PSum {
    let mut new_state = (*state).clone();
    if q < 0 || q as usize >= new_state.num_qubits as usize {
        return PSum::new(new_state); // FFI Shield
    }
    new_state.apply_h(q as usize);
    new_state.reduce();
    PSum::new(new_state)
}

/// Helper function for the debug primitive to avoid nested macro issues.
fn debug_logic(state: PSum) -> S {
    let psum = &*state;
    S::new(format!("PathSum(qubits: {}, path_vars: {}, phase_terms: {})", psum.num_qubits, psum.num_path_vars, psum.phase_poly.terms.len()))
}

impl BaseSort for PathSumSort {
    type Base = PSum;

    fn name(&self) -> &str {
        "PathSum"
    }

    /// Registers the primitive `egglog` functions corresponding to quantum operations.
    /// Each primitive maps a string name (e.g., `rust_apply_cx_ffi`) to the underlying Rust closure.
    fn register_primitives(&self, eg: &mut EGraph) {
        add_primitive!(eg, "rust_id_pathsum_ffi" = |num_qubits: i64| -> PSum {
            id_pathsum_logic(num_qubits)
        });

        add_primitive!(eg, "rust_apply_x_ffi" = |state: PSum, q: i64| -> PSum {
            apply_x_logic(state, q)
        });

        add_primitive!(eg, "rust_apply_z_ffi" = |state: PSum, q: i64| -> PSum {
            apply_z_logic(state, q)
        });

        add_primitive!(eg, "rust_apply_s_ffi" = |state: PSum, q: i64| -> PSum {
            apply_s_logic(state, q)
        });

        add_primitive!(eg, "rust_apply_sdg_ffi" = |state: PSum, q: i64| -> PSum {
            apply_sdg_logic(state, q)
        });

        add_primitive!(eg, "rust_apply_t_ffi" = |state: PSum, q: i64| -> PSum {
            apply_t_logic(state, q)
        });

        add_primitive!(eg, "rust_apply_tdg_ffi" = |state: PSum, q: i64| -> PSum {
            apply_tdg_logic(state, q)
        });

        add_primitive!(eg, "rust_apply_sx_ffi" = |state: PSum, q: i64| -> PSum {
            apply_sx_logic(state, q)
        });

        add_primitive!(eg, "rust_apply_cx_ffi" = |state: PSum, qc: i64, qt: i64| -> PSum {
            apply_cx_logic(state, qc, qt)
        });

        add_primitive!(eg, "rust_apply_h_ffi" = |state: PSum, q: i64| -> PSum {
            apply_h_logic(state, q)
        });

        add_primitive!(eg, "rust_pathsum_debug" = |state: PSum| -> S {
            debug_logic(state)
        });
    }

    /// Reconstructs the term for extraction out of the e-graph.
    ///
    /// This acts as a safe type-fallback. By returning a valid AST constructor
    /// `(rust_id_pathsum_ffi 0)`, we protect the engine from internal type-inference panics
    /// if it accidentally attempts a direct extraction of this complex type.
    fn reconstruct_termdag(&self, _base_values: &BaseValues, _value: Value, termdag: &mut TermDag) -> TermId {
        let arg = termdag.lit(Literal::Int(0));
        termdag.app("rust_id_pathsum_ffi".to_string(), vec![arg])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical_phase_poly::EvaluatedPathSum;

    /// Verifies that the custom `PathSum` sort and its initialization primitive
    /// (`rust_id_pathsum_ffi`) are correctly registered and callable from within
    /// an `egglog` script.
    #[test]
    fn test_bridge_initialization() {
        let mut eg = crate::new_experimental_egraph();
        let script = r#"(let state (rust_id_pathsum_ffi 2))"#;
        let result = eg.parse_and_run_program(None, script);
        assert!(result.is_ok(), "Failed to run rust_id_pathsum through egglog: {:?}", result);
    }

    /// Verifies that all gate application primitives are registered and callable
    /// from within an `egglog` script without raising errors.
    #[test]
    fn test_bridge_all_primitives_callable() {
        let mut eg = crate::new_experimental_egraph();
        let script = r#"
            (let state0 (rust_id_pathsum_ffi 2))
            (let state1 (rust_apply_x_ffi state0 0))
            (let state2 (rust_apply_z_ffi state1 0))
            (let state3 (rust_apply_s_ffi state2 1))
            (let state4 (rust_apply_t_ffi state3 1))
            (let state5 (rust_apply_cx_ffi state4 0 1))
            (let state6 (rust_apply_h_ffi state5 0))
            (let state7 (rust_apply_sdg_ffi state6 1))
            (let state8 (rust_apply_tdg_ffi state7 1))
            (let state9 (rust_apply_sx_ffi state8 0))
            (let debug_str (rust_pathsum_debug state9))
        "#;
        let result = eg.parse_and_run_program(None, script);
        assert!(result.is_ok(), "Failed to run primitives through egglog: {:?}", result);
    }

    /// Tests the `id_pathsum_logic` function for edge cases.
    #[test]
    fn test_id_pathsum_logic_edge_cases() {
        let state = id_pathsum_logic(-5);
        assert_eq!(state.num_qubits, 0);

        let state2 = id_pathsum_logic(3);
        assert_eq!(state2.num_qubits, 3);
    }

    /// Tests that the FFI panic shields correctly prevent out-of-bounds errors,
    /// safely returning the unmodified state.
    #[test]
    fn test_ffi_panic_shields() {
        let initial = id_pathsum_logic(1); // 1 qubit: valid index is 0

        // Out of bounds accesses should return the state unchanged.
        assert_eq!(*apply_x_logic(initial.clone(), 1), *initial);
        assert_eq!(*apply_x_logic(initial.clone(), -1), *initial);

        assert_eq!(*apply_z_logic(initial.clone(), 1), *initial);
        assert_eq!(*apply_s_logic(initial.clone(), 1), *initial);
        assert_eq!(*apply_sdg_logic(initial.clone(), 1), *initial);
        assert_eq!(*apply_t_logic(initial.clone(), 1), *initial);
        assert_eq!(*apply_tdg_logic(initial.clone(), 1), *initial);
        assert_eq!(*apply_sx_logic(initial.clone(), 1), *initial);
        assert_eq!(*apply_h_logic(initial.clone(), 1), *initial);

        // CX requires two distinct, in-bounds qubits
        assert_eq!(*apply_cx_logic(initial.clone(), 0, 1), *initial); // qt out of bounds
        assert_eq!(*apply_cx_logic(initial.clone(), 1, 0), *initial); // qc out of bounds
        assert_eq!(*apply_cx_logic(initial.clone(), 0, 0), *initial); // qc == qt
    }

    /// Tests that the logic functions correctly apply the gate and eagerly
    /// invoke the reduction phase, tracking equivalence with manual Rust application.
    #[test]
    fn test_logic_functions_apply_and_reduce() {
        // Construct the expected state through explicit, manual calls.
        let mut expected = EvaluatedPathSum::new_id(2);

        // H on q0
        expected.apply_h(0);
        expected.reduce();
        let state1 = apply_h_logic(id_pathsum_logic(2), 0);
        assert_eq!(*state1, expected);

        // CX q0 -> q1
        expected.apply_cx(0, 1);
        expected.reduce();
        let state2 = apply_cx_logic(state1, 0, 1);
        assert_eq!(*state2, expected);

        // Z on q1
        expected.apply_z(1);
        expected.reduce();
        let state3 = apply_z_logic(state2, 1);
        assert_eq!(*state3, expected);
    }
}