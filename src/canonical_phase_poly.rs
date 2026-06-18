//! Canonical Phase Polynomial and Boolean Polynomial structures for Path Sum evaluation.
//!
//! This module provides the foundational data structures used to represent and manipulate
//! phase polynomials and boolean polynomials (over GF(2)) in a canonical form. These
//! structures are crucial in quantum compilation and path sum evaluation for efficiently
//! tracking the accumulated phases and output states of qubits.

use crate::continuous_poly::ContinuousPhasePoly;
use std::cmp::Ordering;
use smallvec::SmallVec;

/// A packed representation of a phase term in a phase polynomial.
///
/// It stores both the monomial (a bitset of variables, up to 61 bits)
/// and the phase (a 3-bit value representing a multiple of pi/4, i.e., 0 to 7)
/// in a single 64-bit integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PackedPhaseTerm(pub u64);

impl PackedPhaseTerm {
    /// Bitmask used to extract the phase bits (the top 3 bits).
    pub const PHASE_MASK: u64 = 0xE000_0000_0000_0000;
    /// Bitmask used to extract the monomial bits (the bottom 61 bits).
    pub const MONOMIAL_MASK: u64 = !Self::PHASE_MASK;

    /// Retrieves the monomial part of the packed term.
    #[inline(always)]
    pub fn monomial(&self) -> u64 { self.0 & Self::MONOMIAL_MASK }

    /// Retrieves the phase part of the packed term as a value from 0 to 7.
    #[inline(always)]
    pub fn phase(&self) -> u8 { (self.0 >> 61) as u8 }

    /// Creates a new `PackedPhaseTerm` from a monomial and a phase.
    ///
    /// The monomial is masked to 61 bits and the phase is masked to 3 bits
    /// to ensure the mathematical boundaries are enforced.
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
    /// Custom ordering that prioritizes the monomial, followed by the phase.
    /// This ensures that terms with the same monomial are grouped together.
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.monomial().cmp(&other.monomial())
            .then_with(|| self.phase().cmp(&other.phase()))
    }
}

/// Represents a phase polynomial in a canonical form.
///
/// The terms are stored in a small vector, sorted by their monomial.
/// Multiple terms with the same monomial are compacted, and terms with
/// a phase of 0 (modulo 8) are removed to maintain canonicity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CanonicalPhasePoly {
    /// The sorted, compacted list of phase terms.
    pub terms: SmallVec<[PackedPhaseTerm; 16]>,
}

impl CanonicalPhasePoly {
    /// Adds another canonical phase polynomial to this one in place.
    ///
    /// This performs a linear-time merge of the two sorted term lists,
    /// adding the phases of identical monomials modulo 8. Any resulting
    /// terms with a phase of 0 are discarded.
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

/// Represents a boolean polynomial over GF(2).
///
/// It stores a sorted list of variables or monomials (represented as `u64`).
/// Addition of polynomials behaves like XOR (since a + a = 0 in GF(2)).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BooleanPoly {
    /// The sorted list of terms (monomials) present in the polynomial.
    pub terms: SmallVec<[u64; 8]>,
    /// A bitmask representing the union of all variables present in `terms`.
    pub variable_mask: u64,
}

impl BooleanPoly {
    /// Creates a new `BooleanPoly` from a set of terms.
    /// The terms are automatically sorted and the variable mask is computed.
    pub fn from_terms(mut terms: SmallVec<[u64; 8]>) -> Self {
        terms.sort_unstable();
        let variable_mask = terms.iter().fold(0, |acc, &x| acc | x);
        Self { terms, variable_mask }
    }

    /// Adds another boolean polynomial to this one in place over GF(2).
    ///
    /// This performs a linear-time merge of the two sorted term lists.
    /// Because addition is in GF(2), if a term appears in both polynomials,
    /// they cancel each other out and are removed from the result.
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
        // Recalculate the mask after the merge.
        self.variable_mask = self.terms.iter().fold(0, |acc, &x| acc | x);
    }
}

/// Represents the final evaluation of a path sum.
///
/// Encapsulates the number of qubits, the number of path variables,
/// the boolean polynomials representing the output state of each qubit,
/// and the overall phase polynomial.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EvaluatedPathSum {
    /// The number of qubits in the system.
    pub num_qubits: u32,
    /// The number of path variables introduced during evaluation.
    pub num_path_vars: u32,
    /// The output state for each qubit, represented as a boolean polynomial.
    pub out_state: Vec<BooleanPoly>,
    /// The canonical phase polynomial representing the accumulated phases.
    pub phase_poly: CanonicalPhasePoly,
    /// The passenger pipeline for continuous phase parameters.
    pub continuous_poly: ContinuousPhasePoly,
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
        let mut poly_a = BooleanPoly::from_terms(smallvec![1, 3, 5]);
        let poly_b = BooleanPoly::from_terms(smallvec![1, 4, 5]);

        poly_a.add_assign(&poly_b);

        // 1 and 5 should cancel out. 3 and 4 should remain, sorted.
        let expected = BooleanPoly::from_terms(smallvec![3, 4]);
        assert_eq!(poly_a, expected);
    }

    /// Tests that adding two disjoint boolean polynomials results in a merged polynomial
    /// that contains all terms from both, correctly sorted.
    #[test]
    fn test_boolean_poly_add_assign_disjoint() {
        let mut poly_a = BooleanPoly::from_terms(smallvec![1, 3, 5]);
        let poly_b = BooleanPoly::from_terms(smallvec![2, 4, 6]);

        poly_a.add_assign(&poly_b);

        // Should just be the sorted merge of the two.
        let expected = BooleanPoly::from_terms(smallvec![1, 2, 3, 4, 5, 6]);
        assert_eq!(poly_a, expected);
    }

    /// Tests that adding an empty boolean polynomial to an existing one acts as an
    /// identity operation, leaving the original polynomial unchanged.
    #[test]
    fn test_boolean_poly_add_assign_identity() {
        let mut poly_a = BooleanPoly::from_terms(smallvec![1, 2, 3]);
        let poly_b = BooleanPoly::from_terms(smallvec![]);

        poly_a.add_assign(&poly_b);

        let expected = BooleanPoly::from_terms(smallvec![1, 2, 3]);
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
            BooleanPoly::from_terms(smallvec![1, 2]),
            BooleanPoly::from_terms(smallvec![3]),
        ];

        let path_sum = EvaluatedPathSum {
            num_qubits: 2,
            num_path_vars: 1,
            out_state: out_state.clone(),
            phase_poly: phase_poly.clone(),
            continuous_poly: ContinuousPhasePoly::new(),
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
        let mut poly_a = BooleanPoly::from_terms(smallvec![]);
        let poly_b = BooleanPoly::from_terms(smallvec![1, 2]);

        poly_a.add_assign(&poly_b);
        assert_eq!(poly_a.terms.as_slice(), &[1, 2]);
        assert_eq!(poly_a.variable_mask, 1 | 2);

        let mut poly_c = BooleanPoly::from_terms(smallvec![1, 2]);
        let poly_d = BooleanPoly::from_terms(smallvec![]);

        poly_c.add_assign(&poly_d);
        assert_eq!(poly_c.terms.as_slice(), &[1, 2]);
        assert_eq!(poly_c.variable_mask, 1 | 2);
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

    #[test]
    fn test_variable_mask_creation() {
        let poly = BooleanPoly::from_terms(smallvec![(1 << 2) | (1 << 5), 1 << 3]);
        // The mask should be the bitwise OR of all terms
        let expected_mask = (1 << 2) | (1 << 5) | (1 << 3);
        assert_eq!(poly.variable_mask, expected_mask);
    }

    #[test]
    fn test_variable_mask_cancellation() {
        let mut poly_a = BooleanPoly::from_terms(smallvec![(1 << 1) | (1 << 2), 1 << 3]);
        let poly_b = BooleanPoly::from_terms(smallvec![(1 << 1) | (1 << 2), 1 << 4]);

        // The term (1 << 1) | (1 << 2) should cancel out
        poly_a.add_assign(&poly_b);

        let expected_poly = BooleanPoly::from_terms(smallvec![1 << 3, 1 << 4]);
        assert_eq!(poly_a, expected_poly);
        // The mask should only contain the remaining variables
        assert_eq!(poly_a.variable_mask, (1 << 3) | (1 << 4));
    }
}