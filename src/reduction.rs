use crate::canonical_phase_poly::{BooleanPoly, EvaluatedPathSum, PackedPhaseTerm};
use smallvec::{smallvec, SmallVec};

impl BooleanPoly {
    /// Creates a BooleanPoly from a bitmask, where bit 63 represents the constant 1.
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
        terms.sort_unstable();
        BooleanPoly { terms }
    }
}

impl EvaluatedPathSum {
    /// Reduces the path sum by integrating out internal variables
    /// using Gaussian elimination over GF(2). This is the "Integrator" phase.
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

                if self.out_state.iter().any(|poly| poly.terms.iter().any(|&t| t == v_mask)) {
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
                    let mut b_poly = BooleanPoly { terms: SmallVec::new() };
                    poly.terms.retain(|t| {
                        if (*t & u_mask) != 0 {
                            b_poly.terms.push(*t & !u_mask);
                            false
                        } else {
                            true
                        }
                    });

                    if !b_poly.terms.is_empty() {
                        let mut eb_poly = BooleanPoly { terms: SmallVec::new() };
                        for &e_term in &e_poly.terms {
                            let mut shifted_b = b_poly.clone();
                            if e_term != 0 {
                                for b in &mut shifted_b.terms {
                                    *b |= e_term;
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
                let mut new_terms = smallvec![];
                for term in &poly.terms {
                    new_terms.push(remap_mono(*term));
                }
                new_terms.sort_unstable();
                poly.terms = new_terms;
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

    #[test]
    fn test_reduce_h_does_not_eliminate() {
        let mut state = EvaluatedPathSum::new_id(1);
        state.apply_h(0);
        let original_state = state.clone();
        state.reduce();
        assert_eq!(state, original_state);
    }

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

    #[test]
    fn test_noop_reduction() {
        let mut state = EvaluatedPathSum::new_id(2);
        state.apply_cx(0, 1);
        let initial_state = state.clone();
        state.reduce();
        // A circuit with no path variables (or no solvable equations) should not be altered.
        assert_eq!(state, initial_state);
    }

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

    #[test]
    fn test_elimination_to_constant() {
        let mut state = EvaluatedPathSum::new_id(1);
        state.apply_h(0); // v1 (bit 1), phase: Z(v1*x0)

        // Let's manually add a phase term Z(v1) which means v1*1 -> v1=1
        state.phase_poly.merge_unsorted_batch(vec![
            PackedPhaseTerm::create(1 << 1, 4) // Z(v1)
        ]);

        // So let's create a state where a path variable pivots to a constant.
        // Add a dummy path variable v2 (bit 2) that isn't in out_state.
        state.apply_h(0); // v2 (bit 2), out_state is v2. phase gets Z(v2*v1)

        // Current phase poly: Z(v1*x0), Z(v1), Z(v2*v1)
        // Grouping by v1: Z(v1 * (x0 + 1 + v2))
        // So P_v1 = x0 + 1 + v2.
        // We can pivot on v2! So v2 = x0 + 1.
        // We substitute v2 -> x0 + 1 everywhere.
        // v1 and v2 are eliminated.

        state.reduce();

        assert_eq!(state.num_path_vars, 0);
        // out_state was v2, now it is x0 + 1
        assert_eq!(state.out_state[0].terms.as_slice(), &[0, 1 << 0]);
    }

    #[test]
    fn test_stress_maximal_variables() {
        // The implementation allows up to 61 bits for monomials (qubits + path vars).
        // Let's test the upper bound: 1 qubit and 60 path variables.
        let mut state = EvaluatedPathSum::new_id(1);

        // Applying 30 pairs of H-Z-H gates.
        // Each pair introduces 2 path variables.
        // Total = 60 path variables.
        for _ in 0..30 {
            state.apply_h(0);
            state.apply_z(0);
            state.apply_h(0);
        }

        assert_eq!(state.num_qubits, 1);
        assert_eq!(state.num_path_vars, 60);

        // Reduce the entire 60-variable system.
        state.reduce();

        // Since HZH = X (ignoring global phase), repeating it 30 times
        // corresponds to X^30 = Identity.
        // The reducer should be able to eliminate ALL 60 path variables.
        assert_eq!(state.num_path_vars, 0);

        // Final state should be perfectly returned to the initial x0.
        assert_eq!(state.out_state[0].terms.as_slice(), &[1 << 0]);
        // Global phases are accumulated but all actual path variables are integrated out.
    }
}
