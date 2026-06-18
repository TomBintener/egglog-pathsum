//! Path Sum Reduction via Gaussian Elimination.
//!
//! This module implements the "reduction" or "integration" phase of the path sum
//! evaluation. The core logic resides in the `reduce` method for `EvaluatedPathSum`.
//! This method identifies and solves for internal "path variables" that appear
//! linearly with a phase of pi in the phase polynomial.
//!
//! The process uses Gaussian elimination over GF(2) to substitute these solved
//! variables back into the path sum, effectively integrating them out. This simplifies
//! the representation and is a key step in canonicalizing the path sum. After
//! elimination, any remaining path variables are re-indexed to form a compact,
//! contiguous set.

use crate::canonical_phase_poly::{BooleanPoly, EvaluatedPathSum, PackedPhaseTerm};
use smallvec::SmallVec;

impl BooleanPoly {
    /// Creates a `BooleanPoly` from a bitmask representation.
    ///
    /// This is a helper for the reduction algorithm. The bitmask is interpreted as follows:
    /// - Bit `i` (for `i` < 63) corresponds to a monomial `1 << i`.
    /// - Bit 63 is a special flag representing the constant `1` (monomial `0`).
    /// The resulting terms are sorted to maintain the canonical form of `BooleanPoly`.
    fn from_mask(mask: u64) -> Self {
        let mut terms = SmallVec::new();
        if (mask & (1u64 << 63)) != 0 {
            terms.push(0); // The constant 1
        }
        let mut var_mask = mask & !(1u64 << 63);
        while var_mask != 0 {
            let bit = var_mask.trailing_zeros();
            terms.push(1u64 << bit);
            var_mask &= var_mask - 1; // Clear lowest set bit
        }
        BooleanPoly::from_terms(terms)
    }
}

impl EvaluatedPathSum {
    /// Reduces the path sum by integrating out internal variables using Gaussian elimination.
    ///
    /// This method implements the "Integrator" phase of the path sum formalism. It works
    /// by repeatedly scanning for path variables that can be "solved for". A variable `v`
    /// can be solved for if it meets two criteria:
    ///
    /// 1. It does not appear in the final output state (`out_state`). This means it is
    ///    a purely internal, temporary variable.
    /// 2. It appears only *linearly* and with a phase of *pi* (phase value 4) in the
    ///    phase polynomial. This corresponds to an equation of the form `v * P = pi`,
    ///    which can be rewritten in GF(2) as `v = P`.
    ///
    /// When such a variable `v` is found, we identify a suitable pivot variable `u` from
    /// the polynomial `P`. We then perform a substitution `u = P / u` throughout the entire
    /// path sum (both `out_state` and `phase_poly`). This eliminates both `v` and `u`
    /// from the set of path variables.
    ///
    /// This process is repeated until no more variables can be eliminated. Finally, the
    /// remaining ("surviving") path variables are re-indexed to be contiguous, and the
    /// total `num_path_vars` is updated.
    pub fn reduce(&mut self) {
        let mut dead_vars = 0u64;
        let original_num_path_vars = self.num_path_vars;
        let path_var_mask = if original_num_path_vars == 0 {
            0
        } else {
            u64::MAX.checked_shl(self.num_qubits as u32).unwrap_or(0)
                & u64::MAX.checked_shr(64 - (self.num_qubits + original_num_path_vars) as u32).unwrap_or(0)
        };

        let mut changed = true;
        while changed {
            changed = false;
            for v_idx in self.num_qubits..self.num_qubits + original_num_path_vars {
                let v_mask = 1u64 << v_idx;
                if (dead_vars & v_mask) != 0 {
                    continue;
                }

                if self.out_state.iter().any(|poly| (poly.variable_mask & v_mask) != 0) {
                    continue;
                }

                let mut p_mask = 0u64;
                let mut is_linear_and_phase_pi = true;
                for term in &self.phase_poly.terms {
                    let mono = term.monomial();
                    if (mono & v_mask) != 0 {
                        if term.phase() != 4 {
                            is_linear_and_phase_pi = false;
                            break;
                        }
                        let remaining = mono & !v_mask;
                        if remaining == 0 {
                            p_mask ^= 1u64 << 63; // Constant 1
                        } else if remaining.count_ones() == 1 {
                            p_mask ^= remaining;
                        } else {
                            is_linear_and_phase_pi = false;
                            break;
                        }
                    }
                }

                if !is_linear_and_phase_pi {
                    continue;
                }

                let valid_pivots = p_mask & path_var_mask & !dead_vars;
                if valid_pivots == 0 {
                    continue;
                }

                changed = true;
                let pivot_index = valid_pivots.trailing_zeros();
                let u_mask = 1u64 << pivot_index;
                let e_mask = p_mask ^ u_mask;

                let e_poly = BooleanPoly::from_mask(e_mask);

                for poly in &mut self.out_state {
                    if (poly.variable_mask & u_mask) == 0 {
                        continue;
                    }
                    let mut b_poly = BooleanPoly::from_terms(SmallVec::new());
                    poly.terms.retain(|t| {
                        if (*t & u_mask) != 0 {
                            b_poly.terms.push(*t & !u_mask);
                            false
                        } else {
                            true
                        }
                    });
                    b_poly.variable_mask = b_poly.terms.iter().fold(0, |acc, &x| acc | x);


                    if !b_poly.terms.is_empty() {
                        let mut eb_poly = BooleanPoly::from_terms(SmallVec::new());
                        for e_term in &e_poly.terms {
                            let mut shifted_b = b_poly.clone();
                            if *e_term != 0 {
                                for b in &mut shifted_b.terms {
                                    *b |= *e_term;
                                }
                            }
                            eb_poly.add_assign(&shifted_b);
                        }
                        poly.add_assign(&eb_poly);
                    }
                }

                let mut next_gen_terms = Vec::new();
                for term in self.phase_poly.terms.iter() {
                    let mono = term.monomial();
                    if (mono & v_mask) != 0 {
                        continue;
                    }
                    if (mono & u_mask) != 0 {
                        let base = mono & !u_mask;
                        for &e_term in &e_poly.terms {
                            let new_mono = if e_term == 0 { base } else { base | e_term };
                            next_gen_terms.push(PackedPhaseTerm::create(new_mono, term.phase()));
                        }
                    } else {
                        next_gen_terms.push(*term);
                    }
                }
                self.phase_poly.terms.clear();
                self.phase_poly.merge_unsorted_batch(next_gen_terms);
                dead_vars |= v_mask | u_mask;
            }
        }

        let surviving_mask = path_var_mask & !dead_vars;
        if surviving_mask.count_ones() < self.num_path_vars as u32 {
            let mut remapping = [0u32; 64];
            for i in 0..self.num_qubits {
                remapping[i as usize] = i;
            }
            let mut current_new_idx = 0;
            for i in self.num_qubits..64 {
                if (surviving_mask & (1u64 << i)) != 0 {
                    remapping[i as usize] = self.num_qubits + current_new_idx;
                    current_new_idx += 1;
                }
            }

            let remap_mono = |mono: u64| -> u64 {
                let mut new_mono = 0u64;
                let mut temp = mono;
                while temp != 0 {
                    let bit_idx = temp.trailing_zeros() as usize;
                    new_mono |= 1u64 << remapping[bit_idx];
                    temp &= temp - 1;
                }
                new_mono
            };

            for poly in &mut self.out_state {
                let new_terms = poly.terms.iter().map(|t| remap_mono(*t)).collect();
                *poly = BooleanPoly::from_terms(new_terms);
            }

            let mut new_phase_terms = Vec::with_capacity(self.phase_poly.terms.len());
            for term in &self.phase_poly.terms {
                new_phase_terms.push(PackedPhaseTerm::create(remap_mono(term.monomial()), term.phase()));
            }
            self.phase_poly.terms.clear();
            self.phase_poly.merge_unsorted_batch(new_phase_terms);
            self.num_path_vars = surviving_mask.count_ones();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical_phase_poly::EvaluatedPathSum;

    /// Tests that the sequence H-Z-H, which is equivalent to an X gate,
    /// correctly reduces to the expected state. This is a classic integration test
    /// for the reduction algorithm.
    #[test]
    fn test_reduce_hzh_to_x() {
        let mut state = EvaluatedPathSum::new_id(1);
        state.apply_h(0);
        state.apply_z(0);
        state.apply_h(0);
        assert_eq!(state.num_path_vars, 2);
        state.reduce();
        assert_eq!(state.num_path_vars, 0);
        assert_eq!(state.out_state[0].terms.as_slice(), &[0, 1]);
        assert!(state.phase_poly.terms.is_empty());
    }

    /// Verifies that applying reduction to a state with an unsolvable path variable
    /// (from a single H gate) does not change the state.
    #[test]
    fn test_reduce_h_does_not_eliminate() {
        let mut state = EvaluatedPathSum::new_id(1);
        state.apply_h(0);
        let original_state = state.clone();
        state.reduce();
        assert_eq!(state, original_state);
    }

    /// Tests that a sequence of two Hadamard gates, which should be an identity operation,
    /// correctly reduces to the initial state, eliminating all intermediate path variables.
    #[test]
    fn test_double_h_is_identity() {
        let mut state = EvaluatedPathSum::new_id(1);
        state.apply_h(0);
        state.apply_h(0);
        assert_eq!(state.num_path_vars, 2);
        state.reduce();
        assert_eq!(state.num_path_vars, 0);
        assert_eq!(state.out_state[0].terms.as_slice(), &[1 << 0]);
        assert!(state.phase_poly.terms.is_empty());
    }

    /// A specific test to verify the variable re-indexing (repacking) logic.
    /// It creates a scenario where path variables are eliminated out of order,
    /// forcing the remapping of surviving variables.
    #[test]
    fn test_repacking_logic() {
        let mut state = EvaluatedPathSum::new_id(1); // q0 is bit 0
        state.apply_h(0); // v1 is bit 1
        state.apply_h(0); // v2 is bit 2

        // out_state[0] is {v2}
        // phase_poly has Z(v1*x0), Z(v2*v1)
        // This gives P_v1 = x0 + v2. We can pivot on v2, so v2=x0.
        // We can eliminate v1.
        // We substitute v2=x0 into the output state.
        // The output becomes {x0}.
        // v1 and v2 are eliminated. No path vars should survive.

        state.reduce();

        assert_eq!(state.num_path_vars, 0);
        assert_eq!(state.out_state[0].terms.as_slice(), &[1 << 0]);
    }

    /// Verifies that calling `reduce` on a circuit with no path variables
    /// is a no-op and does not alter the state.
    #[test]
    fn test_noop_reduction() {
        let mut state = EvaluatedPathSum::new_id(2);
        state.apply_cx(0, 1);
        let initial_state = state.clone();
        state.reduce();
        // A circuit with no path variables (or no solvable equations) should not be altered.
        assert_eq!(state, initial_state);
    }

    /// Tests the reduction of a SWAP circuit composed of three CX gates.
    /// The circuit creates no path variables and should reduce to the expected
    /// swapped state.
    #[test]
    fn test_swap_circuit() {
        let mut state = EvaluatedPathSum::new_id(2); // q0, q1
        state.apply_cx(0, 1);
        state.apply_cx(1, 0);
        state.apply_cx(0, 1);
        state.reduce();

        // SWAP gate should map q0 -> x1 and q1 -> x0. No path variables created.
        assert_eq!(state.num_path_vars, 0);
        assert_eq!(state.out_state[0].terms.as_slice(), &[1 << 1]);
        assert_eq!(state.out_state[1].terms.as_slice(), &[1 << 0]);
        assert!(state.phase_poly.terms.is_empty());
    }

    /// Tests a more complex scenario where a variable is eliminated by equating it
    /// to a constant (1). This forces a non-trivial substitution back into the
    /// output state.
    #[test]
    fn test_elimination_to_constant() {
        let mut state = EvaluatedPathSum::new_id(1);
        state.apply_h(0); // v1 (bit 1), phase: Z(v1*x0)

        // Let's manually add a phase term Z(v1) which means v1*1 -> v1=1
        state.phase_poly.merge_unsorted_batch(vec![
            PackedPhaseTerm::create(1 << 1, 4) // Z(v1)
        ]);

        state.apply_h(0); // v2 (bit 2), out_state is v2. phase gets Z(v2*v1)

        state.reduce();

        assert_eq!(state.num_path_vars, 0);
        // out_state was v2, now it is x0 + 1
        assert_eq!(state.out_state[0].terms.as_slice(), &[0, 1 << 0]);
    }
}