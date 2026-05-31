use std::cmp::Ordering;
use smallvec::SmallVec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PackedPhaseTerm(pub u64);

impl PackedPhaseTerm {
    pub const PHASE_MASK: u64 = 0xE000_0000_0000_0000;
    pub const MONOMIAL_MASK: u64 = !Self::PHASE_MASK;

    #[inline(always)]
    pub fn monomial(&self) -> u64 { self.0 & Self::MONOMIAL_MASK }

    #[inline(always)]
    pub fn phase(&self) -> u8 { (self.0 >> 61) as u8 }

    #[inline(always)]
    pub fn create(monomial: u64, phase: u8) -> Self {
        // Silently enforce the mathematical boundaries to prevent memory corruption.
        // This is safe and has zero overhead in release builds.
        let safe_monomial = monomial & Self::MONOMIAL_MASK;
        let safe_phase = (phase & 0b111) as u64; // Mask to keep only the bottom 3 bits

        Self(safe_monomial | (safe_phase << 61))
    }
}

impl PartialOrd for PackedPhaseTerm {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PackedPhaseTerm {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.monomial().cmp(&other.monomial())
            .then_with(|| self.phase().cmp(&other.phase()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CanonicalPhasePoly {
    pub terms: SmallVec<[PackedPhaseTerm; 16]>,
}

impl CanonicalPhasePoly {
    pub fn add_assign(&mut self, other: &Self) {
        let mut result = SmallVec::with_capacity(self.terms.len() + other.terms.len());
        let mut i = 0;
        let mut j = 0;

        while i < self.terms.len() && j < other.terms.len() {
            let a = self.terms[i];
            let b = other.terms[j];

            match a.monomial().cmp(&b.monomial()) {
                Ordering::Less => { result.push(a); i += 1; }
                Ordering::Greater => { result.push(b); j += 1; }
                Ordering::Equal => {
                    let new_phase = (a.phase() + b.phase()) % 8;
                    if new_phase != 0 {
                        result.push(PackedPhaseTerm::create(a.monomial(), new_phase));
                    }
                    i += 1;
                    j += 1;
                }
            }
        }
        result.extend_from_slice(&self.terms[i..]);
        result.extend_from_slice(&other.terms[j..]);
        self.terms = result;
    }

    /// Ingests a raw, unsorted batch of terms, compacts them, and merges them in O(K log K + M)
    pub fn merge_unsorted_batch(&mut self, mut batch: Vec<PackedPhaseTerm>) {
        if batch.is_empty() { return; }

        // 1. Sort using Rust's fastest unstable sort
        batch.sort_unstable();

        // 2. Compact duplicates modulo 8 in-place
        let mut compacted = SmallVec::<[PackedPhaseTerm; 16]>::new();
        if batch.is_empty() {
            return;
        }
        let mut current_mono = batch[0].monomial();
        let mut current_phase = batch[0].phase();

        for term in batch.into_iter().skip(1) {
            if term.monomial() == current_mono {
                current_phase = (current_phase + term.phase()) % 8;
            } else {
                if current_phase != 0 {
                    compacted.push(PackedPhaseTerm::create(current_mono, current_phase));
                }
                current_mono = term.monomial();
                current_phase = term.phase();
            }
        }
        if current_phase != 0 {
            compacted.push(PackedPhaseTerm::create(current_mono, current_phase));
        }

        // 3. Single O(N+M) merge into the main polynomial
        let batch_poly = CanonicalPhasePoly { terms: compacted };
        self.add_assign(&batch_poly);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BooleanPoly {
    pub terms: SmallVec<[u64; 8]>,
}

impl BooleanPoly {
    pub fn add_assign(&mut self, other: &Self) {
        let mut result = SmallVec::with_capacity(self.terms.len() + other.terms.len());
        let mut i = 0;
        let mut j = 0;

        while i < self.terms.len() && j < other.terms.len() {
            let a = self.terms[i];
            let b = other.terms[j];

            match a.cmp(&b) {
                Ordering::Less => {
                    result.push(a);
                    i += 1;
                }
                Ordering::Greater => {
                    result.push(b);
                    j += 1;
                }
                Ordering::Equal => {
                    // In GF(2), a + a = 0, so we drop both terms.
                    i += 1;
                    j += 1;
                }
            }
        }
        result.extend_from_slice(&self.terms[i..]);
        result.extend_from_slice(&other.terms[j..]);
        self.terms = result;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EvaluatedPathSum {
    pub num_qubits: u32,
    pub num_path_vars: u32,
    pub out_state: Vec<BooleanPoly>,
    pub phase_poly: CanonicalPhasePoly,
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    /// Verifies that the `create` function correctly masks out-of-bounds inputs
    /// to guarantee the struct's invariants in both debug and release builds.
    #[test]
    fn test_create_masks_out_of_bounds_inputs() {
        // An oversized monomial that would bleed into the phase bits.
        let oversized_monomial = (1 << 62) | 5;
        // A phase value that exceeds the 3-bit limit (0-7).
        let oversized_phase = 9; // Binary 1001

        let term = PackedPhaseTerm::create(oversized_monomial, oversized_phase);

        // The monomial should be masked to its valid 61 bits.
        assert_eq!(term.monomial(), 5);
        // The phase should be masked to its valid 3 bits (9 becomes 1).
        assert_eq!(term.phase(), 1);
    }

    /// Verifies that our custom `Ord` implementation correctly ignores the phase bits.
    /// A term with a smaller monomial but a larger phase should sort BEFORE a term
    /// with a larger monomial but smaller phase. This is critical for canonicity.
    #[test]
    fn test_custom_ord() {
        let term1 = PackedPhaseTerm::create(1, 7);
        let term2 = PackedPhaseTerm::create(2, 1);

        // Even though phase is larger in term1, monomial is smaller, so it should be Less
        assert_eq!(term1.cmp(&term2), Ordering::Less);
        assert!(term1 < term2);

        // Monomials are equal, so phase acts as a tie-breaker.
        let term3 = PackedPhaseTerm::create(1, 2);
        let term4 = PackedPhaseTerm::create(1, 7);
        assert_eq!(term3.cmp(&term4), Ordering::Less);
    }

    /// Verifies that our custom `Ord` implementation agrees with the derived `PartialEq`.
    /// In Rust, `a.cmp(&b) == Ordering::Equal` if and only if `a == b`.
    #[test]
    fn test_ord_eq_agreement() {
        let term1 = PackedPhaseTerm::create(1, 2);
        let term2 = PackedPhaseTerm::create(1, 3);
        
        assert_ne!(term1.cmp(&term2), Ordering::Equal);
        assert_ne!(term1, term2);
    }

    /// Tests the crucial property that identical terms with opposing phases
    /// (phases that sum to 0 mod 8, e.g., 4 + 4 or 2 + 6) cancel each other out
    /// entirely, leaving no trace in the resulting polynomial container.
    #[test]
    fn test_add_assign_cancels_out() {
        let mut poly_a = CanonicalPhasePoly {
            terms: smallvec![
                PackedPhaseTerm::create(1, 4),
                PackedPhaseTerm::create(3, 2),
            ],
        };

        let poly_b = CanonicalPhasePoly {
            terms: smallvec![
                PackedPhaseTerm::create(1, 4), // 4 + 4 = 8 (0 mod 8) -> drops
                PackedPhaseTerm::create(3, 6), // 2 + 6 = 8 (0 mod 8) -> drops
            ],
        };

        poly_a.add_assign(&poly_b);

        assert!(poly_a.terms.is_empty());
    }

    /// Tests that a typical O(N+M) merge correctly interleaves distinct monomials
    /// and appropriately adds the phases of colliding monomials modulo 8.
    #[test]
    fn test_add_assign_merges() {
        let mut poly_a = CanonicalPhasePoly {
            terms: smallvec![
                PackedPhaseTerm::create(1, 2),
                PackedPhaseTerm::create(3, 1),
            ],
        };

        let poly_b = CanonicalPhasePoly {
            terms: smallvec![
                PackedPhaseTerm::create(2, 5),
                PackedPhaseTerm::create(3, 2),
            ],
        };

        poly_a.add_assign(&poly_b);

        let expected = vec![
            PackedPhaseTerm::create(1, 2), // From A
            PackedPhaseTerm::create(2, 5), // From B
            PackedPhaseTerm::create(3, 3), // Collides: 1 (from A) + 2 (from B) = 3
        ];

        assert_eq!(poly_a.terms.as_slice(), expected.as_slice());
    }

    /// Tests phase wraparound when adding two terms where their sum exceeds 8 but is not a multiple of 8.
    #[test]
    fn test_phase_poly_add_assign_wraparound() {
        let mut poly_a = CanonicalPhasePoly {
            terms: smallvec![PackedPhaseTerm::create(1, 5)],
        };
        let poly_b = CanonicalPhasePoly {
            terms: smallvec![PackedPhaseTerm::create(1, 6)],
        };

        poly_a.add_assign(&poly_b);

        // 5 + 6 = 11, 11 % 8 = 3
        let expected = vec![PackedPhaseTerm::create(1, 3)];
        assert_eq!(poly_a.terms.as_slice(), expected.as_slice());
    }

    /// Tests that trailing terms from one polynomial are correctly appended when the other exhausts early.
    #[test]
    fn test_phase_poly_add_assign_trailing_terms() {
        let mut poly_a = CanonicalPhasePoly {
            terms: smallvec![PackedPhaseTerm::create(1, 2)],
        };
        let poly_b = CanonicalPhasePoly {
            terms: smallvec![
                PackedPhaseTerm::create(2, 5),
                PackedPhaseTerm::create(3, 1),
            ],
        };

        poly_a.add_assign(&poly_b);

        let expected = vec![
            PackedPhaseTerm::create(1, 2),
            PackedPhaseTerm::create(2, 5),
            PackedPhaseTerm::create(3, 1),
        ];
        assert_eq!(poly_a.terms.as_slice(), expected.as_slice());
    }

    /// Tests that adding a boolean polynomial with identical terms correctly cancels
    /// those terms out, mirroring GF(2) addition (XOR) where a + a = 0.
    #[test]
    fn test_boolean_poly_add_assign_cancels() {
        let mut poly_a = BooleanPoly {
            terms: smallvec![1, 3, 5],
        };
        let poly_b = BooleanPoly {
            terms: smallvec![1, 4, 5],
        };

        poly_a.add_assign(&poly_b);

        // 1 and 5 should cancel out. 3 and 4 should remain, sorted.
        let expected = BooleanPoly {
            terms: smallvec![3, 4],
        };
        assert_eq!(poly_a, expected);
    }

    /// Tests that adding two disjoint boolean polynomials results in a merged polynomial
    /// that contains all terms from both, correctly sorted.
    #[test]
    fn test_boolean_poly_add_assign_disjoint() {
        let mut poly_a = BooleanPoly {
            terms: smallvec![1, 3, 5],
        };
        let poly_b = BooleanPoly {
            terms: smallvec![2, 4, 6],
        };

        poly_a.add_assign(&poly_b);

        // Should just be the sorted merge of the two.
        let expected = BooleanPoly {
            terms: smallvec![1, 2, 3, 4, 5, 6],
        };
        assert_eq!(poly_a, expected);
    }

    /// Tests that adding an empty boolean polynomial to an existing one acts as an
    /// identity operation, leaving the original polynomial unchanged.
    #[test]
    fn test_boolean_poly_add_assign_identity() {
        let mut poly_a = BooleanPoly {
            terms: smallvec![1, 2, 3],
        };
        let poly_b = BooleanPoly {
            terms: smallvec![],
        };

        poly_a.add_assign(&poly_b);

        let expected = BooleanPoly {
            terms: smallvec![1, 2, 3],
        };
        assert_eq!(poly_a, expected);
    }

    /// Verifies the basic construction and field assignment of the overarching
    /// EvaluatedPathSum struct, ensuring it correctly holds all required state.
    #[test]
    fn test_evaluated_path_sum_construction() {
        let phase_poly = CanonicalPhasePoly {
            terms: smallvec![PackedPhaseTerm::create(1, 2)],
        };

        let out_state = vec![
            BooleanPoly { terms: smallvec![1, 2] },
            BooleanPoly { terms: smallvec![3] },
        ];

        let path_sum = EvaluatedPathSum {
            num_qubits: 2,
            num_path_vars: 1,
            out_state: out_state.clone(),
            phase_poly: phase_poly.clone(),
        };

        assert_eq!(path_sum.num_qubits, 2);
        assert_eq!(path_sum.num_path_vars, 1);
        assert_eq!(path_sum.out_state, out_state);
        assert_eq!(path_sum.phase_poly, phase_poly);
    }

    /// Tests that the add_assign merge logic for CanonicalPhasePoly correctly handles
    /// edge cases where either the receiver or the argument is completely empty.
    #[test]
    fn test_phase_poly_add_assign_empty() {
        let mut poly_a = CanonicalPhasePoly {
            terms: smallvec![],
        };

        let poly_b = CanonicalPhasePoly {
            terms: smallvec![PackedPhaseTerm::create(1, 2)],
        };

        poly_a.add_assign(&poly_b);
        assert_eq!(poly_a.terms.as_slice(), &[PackedPhaseTerm::create(1, 2)]);

        let mut poly_c = CanonicalPhasePoly {
            terms: smallvec![PackedPhaseTerm::create(1, 2)],
        };

        let poly_d = CanonicalPhasePoly {
            terms: smallvec![],
        };

        poly_c.add_assign(&poly_d);
        assert_eq!(poly_c.terms.as_slice(), &[PackedPhaseTerm::create(1, 2)]);
    }

    /// Tests that the add_assign merge logic for BooleanPoly correctly handles
    /// edge cases where either the receiver or the argument is completely empty.
    #[test]
    fn test_boolean_poly_add_assign_empty() {
        let mut poly_a = BooleanPoly {
            terms: smallvec![],
        };

        let poly_b = BooleanPoly {
            terms: smallvec![1, 2],
        };

        poly_a.add_assign(&poly_b);
        assert_eq!(poly_a.terms.as_slice(), &[1, 2]);

        let mut poly_c = BooleanPoly {
            terms: smallvec![1, 2],
        };

        let poly_d = BooleanPoly {
            terms: smallvec![],
        };

        poly_c.add_assign(&poly_d);
        assert_eq!(poly_c.terms.as_slice(), &[1, 2]);
    }

    /// Tests the `merge_unsorted_batch` method to ensure it correctly sorts,
    /// compacts, and merges a raw vector of phase terms.
    #[test]
    fn test_merge_unsorted_batch() {
        let mut poly = CanonicalPhasePoly {
            terms: smallvec![
                PackedPhaseTerm::create(1, 1),
                PackedPhaseTerm::create(4, 3),
            ],
        };

        let batch = vec![
            PackedPhaseTerm::create(10, 1), // New term
            PackedPhaseTerm::create(1, 2),  // Collides with existing
            PackedPhaseTerm::create(5, 7),  // New term, out of order
            PackedPhaseTerm::create(1, 5),  // Collides with existing and self
        ];

        poly.merge_unsorted_batch(batch);

        // Expected result:
        // Original: (1, 1), (4, 3)
        // Batch compacts to: (1, 7), (5, 7), (10, 1)
        // Merged:
        // mono 1: 1 + 7 = 8 -> 0 (drops)
        // mono 4: 3
        // mono 5: 7
        // mono 10: 1
        let expected = vec![
            PackedPhaseTerm::create(4, 3),
            PackedPhaseTerm::create(5, 7),
            PackedPhaseTerm::create(10, 1),
        ];

        assert_eq!(poly.terms.as_slice(), expected.as_slice());
    }
}
