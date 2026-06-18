//! Path Sum Evaluator for Quantum Circuits.
//!
//! This module provides methods to evaluate quantum gates on the `EvaluatedPathSum`
//! structure. It simulates the action of Clifford+T gates by tracking the transformations
//! of the output state (represented as boolean polynomials) and the accumulated
//! phase polynomial (represented in a canonical form).

use crate::canonical_phase_poly::{BooleanPoly, CanonicalPhasePoly, EvaluatedPathSum, PackedPhaseTerm};
use crate::continuous_poly::ContinuousPhasePoly;

impl EvaluatedPathSum {
    /// Creates a new identity `EvaluatedPathSum` for a given number of qubits.
    ///
    /// The initial state maps each qubit to its corresponding input variable
    /// (e.g., qubit `i` corresponds to the monomial `1 << i`). The phase polynomial
    /// is initially empty, and there are zero path variables.
    pub fn new_id(num_qubits: u32) -> Self {
        let mut out_state = Vec::with_capacity(num_qubits as usize);
        for i in 0..num_qubits {
            out_state.push(BooleanPoly::from_terms(smallvec::smallvec![1 << i]));
        }

        Self {
            num_qubits,
            num_path_vars: 0,
            out_state,
            phase_poly: CanonicalPhasePoly { terms: smallvec::smallvec![] },
            continuous_poly: ContinuousPhasePoly::new(),
        }
    }

    /// Applies a Pauli X gate (NOT gate) to the specified qubit.
    ///
    /// The X gate flips the boolean state of the qubit, which corresponds to
    /// adding the constant `1` (represented as the monomial `0`) in GF(2)
    /// to the target qubit's boolean polynomial.
    pub fn apply_x(&mut self, q: usize) {
        let constant_one = BooleanPoly::from_terms(smallvec::smallvec![0]);
        self.out_state[q].add_assign(&constant_one);
    }

    /// Applies a Controlled-NOT (CX) gate between a control qubit and a target qubit.
    ///
    /// The CX gate adds the boolean state of the control qubit to the target qubit
    /// in GF(2), which corresponds to an XOR operation on their polynomial representations.
    pub fn apply_cx(&mut self, qc: usize, qt: usize) {
        assert!(qc != qt, "CX control and target must be distinct");

        // ZERO-ALLOCATION BORROWING: Safely extract two disjoint slices
        let (ctrl_poly, tgt_poly) = if qc < qt {
            let (left, right) = self.out_state.split_at_mut(qt);
            (&left[qc], &mut right[0])
        } else {
            let (left, right) = self.out_state.split_at_mut(qc);
            (&right[0], &mut left[qt])
        };

        tgt_poly.add_assign(ctrl_poly);
    }

    /// Applies a Pauli Z gate to the specified qubit.
    ///
    /// The Z gate applies a phase of pi (-1) if the qubit is in the |1> state.
    /// This adds a phase of pi (phase value 4) to the phase polynomial for every
    /// term currently present in the qubit's boolean state.
    pub fn apply_z(&mut self, q: usize) {
        let terms = &self.out_state[q].terms;
        let mut batch = Vec::with_capacity(terms.len());

        for &t in terms {
            batch.push(PackedPhaseTerm::create(t, 4));
        }
        self.phase_poly.merge_unsorted_batch(batch);
    }

    /// Applies an S gate (Phase gate) to the specified qubit.
    ///
    /// The S gate applies a phase of pi/2 (i) if the qubit is in the |1> state.
    /// Because the state is a sum of terms in GF(2), the phase distributes over the terms.
    /// This introduces individual phases of pi/2 (value 2) for each term, and
    /// cross-terms (bitwise OR of term pairs) with a phase of pi (value 4).
    pub fn apply_s(&mut self, q: usize) {
        let terms = &self.out_state[q].terms;
        let n = terms.len();

        // Exact combinatorial pre-allocation: N + (N choose 2)
        let capacity = n + (n * n.saturating_sub(1)) / 2;
        let mut batch = Vec::with_capacity(capacity);

        for i in 0..n {
            batch.push(PackedPhaseTerm::create(terms[i], 2)); // +pi/2
            for j in (i + 1)..n {
                batch.push(PackedPhaseTerm::create(terms[i] | terms[j], 4)); // cross-term +pi
            }
        }
        self.phase_poly.merge_unsorted_batch(batch);
    }

    /// Applies an S-dagger gate (inverse Phase gate) to the specified qubit.
    pub fn apply_sdg(&mut self, q: usize) {
        let terms = &self.out_state[q].terms;
        let n = terms.len();
        let capacity = n + (n * n.saturating_sub(1)) / 2;
        let mut batch = Vec::with_capacity(capacity);

        for i in 0..n {
            batch.push(PackedPhaseTerm::create(terms[i], 6)); // -pi/2
            for j in (i + 1)..n {
                batch.push(PackedPhaseTerm::create(terms[i] | terms[j], 4)); // cross-term +pi
            }
        }
        self.phase_poly.merge_unsorted_batch(batch);
    }

    /// Applies a T gate (pi/4 Phase gate) to the specified qubit.
    ///
    /// The T gate applies a phase of pi/4 if the qubit is in the |1> state.
    /// The phase distribution generates individual pi/4 phases (value 1) for each term,
    /// pair-wise cross-terms with -pi/2 mod 2pi (value 6), and triplet cross-terms
    /// with pi (value 4).
    pub fn apply_t(&mut self, q: usize) {
        let terms = &self.out_state[q].terms;
        let n = terms.len();

        // Exact combinatorial pre-allocation: N + (N choose 2) + (N choose 3)
        let pairs = (n * n.saturating_sub(1)) / 2;
        let triplets = (n * n.saturating_sub(1) * n.saturating_sub(2)) / 6;
        let capacity = n + pairs + triplets;

        let mut batch = Vec::with_capacity(capacity);

        for i in 0..n {
            batch.push(PackedPhaseTerm::create(terms[i], 1)); // +pi/4
            for j in (i + 1)..n {
                batch.push(PackedPhaseTerm::create(terms[i] | terms[j], 6)); // -pi/2 mod 8
                for k in (j + 1)..n {
                    batch.push(PackedPhaseTerm::create(terms[i] | terms[j] | terms[k], 4)); // +pi
                }
            }
        }
        self.phase_poly.merge_unsorted_batch(batch);
    }

    /// Applies a T-dagger gate (inverse pi/4 Phase gate) to the specified qubit.
    pub fn apply_tdg(&mut self, q: usize) {
        let terms = &self.out_state[q].terms;
        let n = terms.len();
        let pairs = (n * n.saturating_sub(1)) / 2;
        let triplets = (n * n.saturating_sub(1) * n.saturating_sub(2)) / 6;
        let capacity = n + pairs + triplets;
        let mut batch = Vec::with_capacity(capacity);

        for i in 0..n {
            batch.push(PackedPhaseTerm::create(terms[i], 7)); // -pi/4
            for j in (i + 1)..n {
                batch.push(PackedPhaseTerm::create(terms[i] | terms[j], 2)); // +pi/2 mod 8
                for k in (j + 1)..n {
                    batch.push(PackedPhaseTerm::create(terms[i] | terms[j] | terms[k], 4)); // +pi
                }
            }
        }
        self.phase_poly.merge_unsorted_batch(batch);
    }

    /// Applies a square-root-of-X gate to the specified qubit.
    /// This is equivalent to H S H.
    pub fn apply_sx(&mut self, q: usize) {
        self.apply_h(q);
        self.apply_s(q);
        self.apply_h(q);
    }

    /// Applies a Hadamard (H) gate to the specified qubit.
    ///
    /// The Hadamard gate introduces a new path variable representing the quantum superposition.
    /// It updates the target qubit's state to this new path variable, and adds
    /// phase cross-terms (phase pi, value 4) between the old state terms and the new path variable
    /// into the phase polynomial.
    pub fn apply_h(&mut self, q: usize) {
        let var_index = self.num_qubits + self.num_path_vars;
        assert!(var_index < 61, "Exceeded 61-bit limit for path variables");

        let v_mask = 1u64 << var_index;
        self.num_path_vars += 1;

        let old_state = std::mem::replace(
            &mut self.out_state[q],
            BooleanPoly::from_terms(smallvec::smallvec![v_mask]),
        );

        let mut batch = Vec::with_capacity(old_state.terms.len());
        for t in old_state.terms {
            batch.push(PackedPhaseTerm::create(t | v_mask, 4));
        }
        self.phase_poly.merge_unsorted_batch(batch);
    }

    /// Applies a continuous Rz gate to the specified qubit.
    pub fn apply_rz(&mut self, q: usize, theta: f64) {
        // 1. Capture the exact Boolean parity of the target qubit at this moment
        let current_parity = self.out_state[q].clone();

        // 2. Append the parity and the rotation angle to the lazy continuous pipeline
        self.continuous_poly.apply_phase(current_parity, theta);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical_phase_poly::PackedPhaseTerm;

    /// Verifies that creating a new identity state initializes the qubits
    /// to their corresponding input variables and starts with an empty phase polynomial.
    #[test]
    fn test_new_id() {
        let state = EvaluatedPathSum::new_id(3);
        assert_eq!(state.num_qubits, 3);
        assert_eq!(state.num_path_vars, 0);
        assert_eq!(state.out_state.len(), 3);
        assert_eq!(state.out_state[0].terms.as_slice(), &[1 << 0]);
        assert_eq!(state.out_state[1].terms.as_slice(), &[1 << 1]);
        assert_eq!(state.out_state[2].terms.as_slice(), &[1 << 2]);
        assert!(state.phase_poly.terms.is_empty());
        assert!(state.continuous_poly.parities.is_empty());
    }

    /// Tests that the Pauli X gate correctly flips the state of a qubit
    /// by adding the constant `1` (monomial `0`) to its boolean polynomial.
    #[test]
    fn test_apply_x() {
        let mut state = EvaluatedPathSum::new_id(1);
        state.apply_x(0);
        // State is now x_0 + 1
        assert_eq!(state.out_state[0].terms.as_slice(), &[0, 1]);
    }

    /// Verifies that applying the Pauli X gate twice results in the identity operation.
    #[test]
    fn test_apply_x_identity() {
        let mut state = EvaluatedPathSum::new_id(1);
        let initial_state = state.out_state[0].clone();
        state.apply_x(0);
        state.apply_x(0); // Applying twice should be an identity
        assert_eq!(state.out_state[0], initial_state);
    }

    /// Tests the Controlled-NOT (CX) gate by ensuring it adds the control qubit's
    /// state to the target qubit's state in GF(2) (XOR).
    #[test]
    fn test_apply_cx() {
        let mut state = EvaluatedPathSum::new_id(2);
        state.apply_cx(0, 1);
        // Target state (q1) becomes x_1 + x_0
        assert_eq!(state.out_state[1].terms.as_slice(), &[1 << 0, 1 << 1]);
        // Control state (q0) is unchanged
        assert_eq!(state.out_state[0].terms.as_slice(), &[1 << 0]);
    }

    /// Tests the CX gate with target and control reversed to ensure
    /// that memory aliasing and splitting works correctly regardless of index order.
    #[test]
    fn test_apply_cx_reversed() {
        let mut state = EvaluatedPathSum::new_id(2);
        state.apply_cx(1, 0);
        // Target state (q0) becomes x_0 + x_1
        assert_eq!(state.out_state[0].terms.as_slice(), &[1 << 0, 1 << 1]);
        // Control state (q1) is unchanged
        assert_eq!(state.out_state[1].terms.as_slice(), &[1 << 1]);
    }

    /// Verifies that applying the CX gate twice with the same control and target
    /// results in the identity operation.
    #[test]
    fn test_apply_cx_identity() {
        let mut state = EvaluatedPathSum::new_id(2);
        let initial_state = state.out_state.clone();
        state.apply_cx(0, 1);
        state.apply_cx(0, 1); // Applying twice should be an identity
        assert_eq!(state.out_state, initial_state);
    }

    /// Tests the Pauli Z gate to ensure it correctly maps a boolean state to a phase polynomial
    /// where every monomial receives a phase of pi (value 4).
    #[test]
    fn test_apply_z() {
        let mut state = EvaluatedPathSum::new_id(1);
        state.out_state[0] = BooleanPoly::from_terms(smallvec::smallvec![1, 2, 4]); // x0 + x1 + x2
        state.apply_z(0);
        // Phase poly should contain Z(x0+x1+x2) = Z(x0)Z(x1)Z(x2)
        // = exp(i*pi*(x0+x1+x2))
        // = phase(4) on monomials 1, 2, 4
        let mut expected_phases = vec![
            PackedPhaseTerm::create(1, 4),
            PackedPhaseTerm::create(2, 4),
            PackedPhaseTerm::create(4, 4),
        ];
        expected_phases.sort_unstable();
        assert_eq!(state.phase_poly.terms.as_slice(), expected_phases.as_slice());
    }

    /// Tests the S (Phase) gate to ensure it distributes a pi/2 phase correctly
    /// over a sum of terms, generating pi/2 phases for individual terms
    /// and pi phases for cross-terms.
    #[test]
    fn test_apply_s() {
        let mut state = EvaluatedPathSum::new_id(1);
        state.out_state[0] = BooleanPoly::from_terms(smallvec::smallvec![1, 2]); // x0 + x1
        state.apply_s(0);
        // S(x0+x1) = S(x0)S(x1)Z(x0)Z(x1)
        // Phases:
        // S(x0) -> mono 1, phase 2
        // S(x1) -> mono 2, phase 2
        // Z(x0)Z(x1) -> mono 1|2, phase 4
        let mut expected_phases = vec![
            PackedPhaseTerm::create(1, 2),
            PackedPhaseTerm::create(2, 2),
            PackedPhaseTerm::create(1 | 2, 4),
        ];
        expected_phases.sort_unstable();
        assert_eq!(state.phase_poly.terms.as_slice(), expected_phases.as_slice());
    }

    /// Tests the T (pi/4 Phase) gate to ensure it correctly distributes a pi/4 phase
    /// over a sum of terms, generating the correct individual, pair-wise, and triplet phases.
    #[test]
    fn test_apply_t() {
        let mut state = EvaluatedPathSum::new_id(1);
        state.out_state[0] = BooleanPoly::from_terms(smallvec::smallvec![1, 2, 4]); // x0 + x1 + x2
        state.apply_t(0);
        // T(x0+x1+x2) = T(x0)T(x1)T(x2) S(x0)S(x1)S(x2)Z(x0x1x2)
        // Phases:
        // T(xi) -> mono i, phase 1
        // S(xixj) -> mono i|j, phase 2
        // Z(x0x1x2) -> mono 1|2|4, phase 4
        // The formula in apply_t is a more optimized version of this.
        let mut expected_phases = vec![
            PackedPhaseTerm::create(1, 1),
            PackedPhaseTerm::create(2, 1),
            PackedPhaseTerm::create(4, 1),
            PackedPhaseTerm::create(1 | 2, 6),
            PackedPhaseTerm::create(1 | 4, 6),
            PackedPhaseTerm::create(2 | 4, 6),
            PackedPhaseTerm::create(1 | 2 | 4, 4),
        ];
        expected_phases.sort_unstable();
        assert_eq!(state.phase_poly.terms.as_slice(), expected_phases.as_slice());
    }

    /// Tests the Hadamard (H) gate to ensure it introduces a new path variable
    /// to represent superposition, sets the target qubit to this variable,
    /// and correctly adds phase cross-terms between the old state and the new variable.
    #[test]
    fn test_apply_h() {
        let mut state = EvaluatedPathSum::new_id(1);
        state.out_state[0] = BooleanPoly::from_terms(smallvec::smallvec![1, 2]); // x0 + x1
        state.apply_h(0);

        // New path variable `v` is introduced at index 1 (since num_qubits=1)
        let v_mask = 1 << 1;
        assert_eq!(state.num_path_vars, 1);
        // The output state for q0 is now just `v`
        assert_eq!(state.out_state[0].terms.as_slice(), &[v_mask]);

        // The phase polynomial gets the cross terms Z(v*x0)Z(v*x1)
        let mut expected_phases = vec![
            PackedPhaseTerm::create( (1<<0) | v_mask, 4),
            PackedPhaseTerm::create( (1<<1) | v_mask, 4),
        ];
        expected_phases.sort_unstable();
        assert_eq!(state.phase_poly.terms.as_slice(), expected_phases.as_slice());
    }

    /// Integration test for the composition of H, Z, and H gates.
    /// Verifies that multiple path variables are allocated correctly
    /// and that the phases accumulate as expected for a sequence of gates.
    #[test]
    fn test_hzh_composition() {
        let mut state = EvaluatedPathSum::new_id(1);
        state.apply_h(0);
        state.apply_z(0);
        state.apply_h(0);

        // Verify that the exact un-integrated path sum representation is constructed correctly.
        // x0 is 1<<0 = 1
        // first H introduces v1 at 1<<1 = 2. out_state is v1. phase gets Z(v1*x0) = (1|2, 4)
        // Z adds Z(v1) = (2, 4) to phase poly
        // second H introduces v2 at 1<<2 = 4. out_state is v2. phase gets Z(v2*v1) = (2|4, 4)

        assert_eq!(state.out_state[0].terms.as_slice(), &[4]);

        let mut expected_phases = vec![
            PackedPhaseTerm::create(1 | 2, 4), // from first H
            PackedPhaseTerm::create(2, 4),     // from Z
            PackedPhaseTerm::create(2 | 4, 4), // from second H
        ];
        expected_phases.sort_unstable();
        assert_eq!(state.phase_poly.terms.as_slice(), expected_phases.as_slice());
    }

    /// Verifies that applying a phase gate (S) to a constant state (monomial 0)
    /// correctly adds a global phase to the polynomial.
    #[test]
    fn test_phase_on_constant() {
        let mut state = EvaluatedPathSum::new_id(1);
        // Set the state to a constant 1 (monomial 0)
        state.out_state[0] = BooleanPoly::from_terms(smallvec::smallvec![0]);
        state.apply_s(0);

        // S gate on a constant 1 should add a global phase of pi/2
        let expected_phases = vec![PackedPhaseTerm::create(0, 2)];
        assert_eq!(state.phase_poly.terms.as_slice(), expected_phases.as_slice());
    }

    #[test]
    fn test_s_sdg_identity() {
        let mut state = EvaluatedPathSum::new_id(1);
        let initial_state = state.clone();
        state.apply_s(0);
        state.apply_sdg(0);
        assert_eq!(state, initial_state);

        let mut state2 = EvaluatedPathSum::new_id(1);
        let initial_state2 = state2.clone();
        state2.apply_sdg(0);
        state2.apply_s(0);
        assert_eq!(state2, initial_state2);
    }

    #[test]
    fn test_t_tdg_identity() {
        let mut state = EvaluatedPathSum::new_id(1);
        let initial_state = state.clone();
        state.apply_t(0);
        state.apply_tdg(0);
        assert_eq!(state, initial_state);

        let mut state2 = EvaluatedPathSum::new_id(1);
        let initial_state2 = state2.clone();
        state2.apply_tdg(0);
        state2.apply_t(0);
        assert_eq!(state2, initial_state2);
    }

    #[test]
    fn test_sx_is_h_s_h() {
        let mut state_sx = EvaluatedPathSum::new_id(1);
        state_sx.apply_sx(0);
        state_sx.reduce();

        let mut state_hsh = EvaluatedPathSum::new_id(1);
        state_hsh.apply_h(0);
        state_hsh.apply_s(0);
        state_hsh.apply_h(0);
        state_hsh.reduce();

        assert_eq!(state_sx, state_hsh);
    }
}