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
use egglog::sort::{BaseValues, Boxed};
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

        add_primitive!(eg, "rust_apply_t_ffi" = |state: PSum, q: i64| -> PSum {
            apply_t_logic(state, q)
        });

        add_primitive!(eg, "rust_apply_cx_ffi" = |state: PSum, qc: i64, qt: i64| -> PSum {
            apply_cx_logic(state, qc, qt)
        });

        add_primitive!(eg, "rust_apply_h_ffi" = |state: PSum, q: i64| -> PSum {
            apply_h_logic(state, q)
        });
    }

    /// Reconstructs the term for extraction out of the e-graph.
    ///
    /// Since the full internal representation of the path sum is too complex to be
    /// usefully extracted back into an `egglog` AST directly, we return a placeholder string literal.
    fn reconstruct_termdag(&self, _base_values: &BaseValues, _value: Value, termdag: &mut TermDag) -> TermId {
        termdag.lit(Literal::String("<Unextracted PathSum State>".into()))
    }
}

#[cfg(test)]
mod tests {

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
}