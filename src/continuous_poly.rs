//! Continuous Phase Polynomial for tracking arbitrary rotations.
//!
//! This module provides the `ContinuousPhasePoly` data structure, designed to manage
//! continuous-valued phase parameters in a quantum circuit. It acts as a "passenger"
//! to the main GF(2) reduction engine, applying substitutions lazily and using a
//! Structure of Arrays (SoA) layout for cache efficiency.

use crate::canonical_phase_poly::BooleanPoly;
use std::f64::consts::TAU; // TAU is 2 * PI
use std::hash::{Hash, Hasher};

const EPSILON: f64 = 1e-10;

/// A data structure to track continuous phase parameters, designed to work
/// alongside the discrete `CanonicalPhasePoly`.
///
/// It uses a Structure of Arrays (SoA) layout (`parities` and `phases` are separate)
/// to improve cache performance and enable vectorization during evaluation.
#[derive(Debug, Clone, Default)]
pub struct ContinuousPhasePoly {
    /// The GF(2) parity associated with each continuous phase.
    pub parities: Vec<BooleanPoly>,
    /// The continuous phase value (angle in radians) for each parity.
    pub phases: Vec<f64>,
}

/// Quantizes a phase into a discrete bucket for consistent hashing and equality.
fn quantize_phase(phase: f64) -> i64 {
    // Normalize to [0, TAU), which is the standard interval.
    let norm_phase = phase.rem_euclid(TAU);
    // If the phase is extremely close to TAU, it's equivalent to 0.
    if (TAU - norm_phase) <= EPSILON {
        return 0;
    }
    // Snap the phase to a grid defined by EPSILON.
    (norm_phase / EPSILON).round() as i64
}

impl ContinuousPhasePoly {
    /// Creates a new, empty `ContinuousPhasePoly`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Applies a continuous phase rotation.
    ///
    /// The phase `theta` is normalized modulo 2*pi. If the normalized phase is
    /// negligibly small, it is discarded to conserve memory.
    pub fn apply_phase(&mut self, parity: BooleanPoly, theta: f64) {
        let normalized_theta = theta.rem_euclid(TAU);

        if normalized_theta <= EPSILON || (TAU - normalized_theta) <= EPSILON {
            return;
        }

        // Use Binary Search to maintain strict sorting order instantly.
        // This requires ZERO allocations and runs in O(log N) time.
        match self.parities.binary_search_by(|p| p.terms.cmp(&parity.terms)) {
            Ok(idx) => {
                // The parity exists. Merge the phases in-place.
                let new_phase = (self.phases[idx] + normalized_theta).rem_euclid(TAU);

                if new_phase <= EPSILON || (TAU - new_phase) <= EPSILON {
                    // Phases canceled out perfectly; remove the terms
                    self.parities.remove(idx);
                    self.phases.remove(idx);
                } else {
                    self.phases[idx] = new_phase;
                }
            }
            Err(idx) => {
                // The parity does NOT exist.
                // `idx` is the exact sorted position where it should be inserted.
                self.parities.insert(idx, parity);
                self.phases.insert(idx, normalized_theta);
            }
        }
    }

    /// Forces the polynomial into a strict canonical form.
    /// Sorts parities and merges overlapping continuous phases without HashMap allocations.
    pub fn compact(&mut self) {
        if self.parities.is_empty() {
            return;
        }

        // 1. Zip into a single vector for fast sorting
        let mut combined: Vec<_> = self.parities.drain(..).zip(self.phases.drain(..)).collect();

        // 2. Sort deterministically by terms
        combined.sort_unstable_by(|a, b| a.0.terms.cmp(&b.0.terms));

        // 3. In-place deduplication and phase merging
        let mut i = 0;
        while i < combined.len() {
            let mut j = i + 1;
            let mut accumulated_phase = combined[i].1;

            // Look ahead for identical parities and merge their phases
            while j < combined.len() && combined[j].0.terms == combined[i].0.terms {
                accumulated_phase += combined[j].1;
                j += 1;
            }

            // Normalize and check against EPSILON threshold
            let norm = accumulated_phase.rem_euclid(TAU);
            if norm > EPSILON && (TAU - norm) > EPSILON {
                self.parities.push(combined[i].0.clone());
                self.phases.push(norm);
            }

            // Skip past the duplicates we just merged
            i = j;
        }
    }

    /// The integration hook called by the main reduction engine.
    ///
    /// This method performs a substitution for a pivot variable (`u_mask`) with a
    /// substitution expression (`e_poly`) across all stored parities.
    ///
    /// # Arguments
    /// * `u_mask`: The bitmask of the pivot variable to be eliminated.
    /// * `e_poly`: The boolean polynomial to substitute in place of the pivot.
    pub fn substitute(&mut self, u_mask: u64, e_poly: &BooleanPoly) {
        for parity in self.parities.iter_mut() {
            // The check is now a single, fast bitwise AND operation.
            if (parity.variable_mask & u_mask) == 0 {
                continue;
            }

            // Create a new polynomial `b_poly` containing only the terms with `u_mask`.
            let mut b_poly = BooleanPoly::from_terms(Default::default());
            parity.terms.retain(|term| {
                if (*term & u_mask) != 0 {
                    // Add the term without `u_mask` to `b_poly`.
                    b_poly.terms.push(*term & !u_mask);
                    false // Remove the original term from the parity.
                } else {
                    true // Keep the term.
                }
            });
            // The terms in b_poly were derived from a sorted list, but the filtering
            // and mapping might have disordered them. We must re-sort to maintain
            // the canonical form required by `add_assign`.
            b_poly.terms.sort_unstable();
            b_poly.variable_mask = b_poly.terms.iter().fold(0, |acc, &x| acc | x);

            // If b_poly is not empty, perform the substitution: parity += e_poly * b_poly
            if !b_poly.terms.is_empty() {
                let mut eb_poly = BooleanPoly::from_terms(Default::default());
                for e_term in &e_poly.terms {
                    let mut shifted_b = b_poly.clone();
                    if *e_term != 0 { // If e_term is not the constant one
                        for b in &mut shifted_b.terms {
                            *b |= *e_term;
                        }
                        shifted_b.terms.sort_unstable();
                    }
                    eb_poly.add_assign(&shifted_b);
                }
                parity.add_assign(&eb_poly);
            }
        }
    }
}

impl PartialEq for ContinuousPhasePoly {
    fn eq(&self, other: &Self) -> bool {
        if self.parities != other.parities || self.phases.len() != other.phases.len() {
            return false;
        }

        self.phases
            .iter()
            .zip(other.phases.iter())
            .all(|(&a, &b)| quantize_phase(a) == quantize_phase(b))
    }
}

impl Eq for ContinuousPhasePoly {}

impl Hash for ContinuousPhasePoly {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.parities.hash(state);
        for &phase in &self.phases {
            quantize_phase(phase).hash(state);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical_phase_poly::BooleanPoly;
    use smallvec::smallvec;
    use std::collections::hash_map::DefaultHasher;

    #[test]
    fn test_apply_phase_merges_terms() {
        let mut poly = ContinuousPhasePoly::new();
        let parity = BooleanPoly::from_terms(smallvec![1]);
        poly.apply_phase(parity.clone(), 1.0);
        poly.apply_phase(parity.clone(), 0.5);

        assert_eq!(poly.parities.len(), 1);
        assert_eq!(poly.phases.len(), 1);
        assert!((poly.phases[0] - 1.5).abs() < EPSILON);
    }

    #[test]
    fn test_apply_phase_cancels_terms() {
        let mut poly = ContinuousPhasePoly::new();
        let parity = BooleanPoly::from_terms(smallvec![1]);
        poly.apply_phase(parity.clone(), 1.0);
        poly.apply_phase(parity.clone(), -1.0);

        assert!(poly.parities.is_empty());
        assert!(poly.phases.is_empty());
    }

    #[test]
    fn test_compact_merges_and_sorts() {
        let mut poly = ContinuousPhasePoly::new();
        let p1 = BooleanPoly::from_terms(smallvec![1]);
        let p2 = BooleanPoly::from_terms(smallvec![2]);
        let p3 = BooleanPoly::from_terms(smallvec![3]);

        // Add terms out of order and with duplicates
        poly.parities = vec![p3.clone(), p1.clone(), p2.clone(), p1.clone()];
        poly.phases = vec![0.5, 0.2, 0.8, 0.3];

        poly.compact();

        assert_eq!(poly.parities.len(), 3);
        assert_eq!(poly.phases.len(), 3);

        // Check that p1 was merged correctly
        assert_eq!(poly.parities[0], p1);
        assert!((poly.phases[0] - 0.5).abs() < EPSILON);

        // Check that p2 is in the correct sorted position
        assert_eq!(poly.parities[1], p2);
        assert!((poly.phases[1] - 0.8).abs() < EPSILON);

        // Check that p3 is in the correct sorted position
        assert_eq!(poly.parities[2], p3);
        assert!((poly.phases[2] - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_compact_with_cancellation() {
        let mut poly = ContinuousPhasePoly::new();
        let p1 = BooleanPoly::from_terms(smallvec![1]);
        let p2 = BooleanPoly::from_terms(smallvec![2]);

        poly.parities = vec![p1.clone(), p2.clone(), p1.clone()];
        poly.phases = vec![0.5, 1.0, -0.5];

        poly.compact();

        // p1 should have been completely cancelled out and removed.
        assert_eq!(poly.parities.len(), 1);
        assert_eq!(poly.phases.len(), 1);
        assert_eq!(poly.parities[0], p2);
        assert!((poly.phases[0] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_substitute_causes_compaction() {
        let mut poly = ContinuousPhasePoly::new();
        let u_mask = 1 << 60;

        // p1 = u
        let p1 = BooleanPoly::from_terms(smallvec![u_mask]);
        // p2 = x1
        let p2 = BooleanPoly::from_terms(smallvec![1 << 1]);

        poly.apply_phase(p1, 1.0);
        poly.apply_phase(p2, 2.0);

        // Substitute u -> x1. This will make the first parity identical to the second.
        let e_poly = BooleanPoly::from_terms(smallvec![1 << 1]);
        poly.substitute(u_mask, &e_poly);

        // The two parities should now have been merged into one.
        assert_eq!(poly.parities.len(), 1);
        assert_eq!(poly.phases.len(), 1);
        assert_eq!(poly.parities[0], BooleanPoly::from_terms(smallvec![1 << 1]));
        assert!((poly.phases[0] - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_full_compaction_after_substitution() {
        let mut poly = ContinuousPhasePoly::new();
        let u_mask = 1 << 60;

        // p1 = u
        let p1 = BooleanPoly::from_terms(smallvec![u_mask]);
        // p2 = x1 + u
        let p2 = BooleanPoly::from_terms(smallvec![1 << 1, u_mask]);

        poly.apply_phase(p1, 1.0);
        poly.apply_phase(p2, 2.0);

        // Substitute u -> x1.
        // p1 becomes x1.
        // p2 becomes x1 + x1 = 0.
        let e_poly = BooleanPoly::from_terms(smallvec![1 << 1]);
        poly.substitute(u_mask, &e_poly);

        // The substitution results in two terms:
        // 1. A phase of 1.0 with parity x1
        // 2. A phase of 2.0 with parity 0 (a global phase)
        assert_eq!(poly.parities.len(), 2);
        assert_eq!(poly.phases.len(), 2);

        // The list is sorted, so the empty polynomial (global phase) comes first.
        assert_eq!(poly.parities[0], BooleanPoly::from_terms(smallvec![]));
        assert!((poly.phases[0] - 2.0).abs() < EPSILON);

        assert_eq!(poly.parities[1], BooleanPoly::from_terms(smallvec![1 << 1]));
        assert!((poly.phases[1] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_hash_quantization_for_equality() {
        let p1 = BooleanPoly::from_terms(smallvec![1]);

        let mut poly1 = ContinuousPhasePoly::new();
        poly1.apply_phase(p1.clone(), 1.0);

        let mut poly2 = ContinuousPhasePoly::new();
        // This difference is too small to change the quantization bucket
        poly2.apply_phase(p1.clone(), 1.0 + EPSILON / 4.0);

        // The two polynomials should be considered equal because they are in the same bucket
        assert_eq!(poly1, poly2);

        // Their hashes MUST be equal
        let mut hasher1 = DefaultHasher::new();
        poly1.hash(&mut hasher1);
        let hash1 = hasher1.finish();

        let mut hasher2 = DefaultHasher::new();
        poly2.hash(&mut hasher2);
        let hash2 = hasher2.finish();

        assert_eq!(hash1, hash2, "Hashes of numbers in the same bucket must be equal");

        // This difference IS large enough to cross a bucket boundary
        let mut poly3 = ContinuousPhasePoly::new();
        poly3.apply_phase(p1.clone(), 1.0 + EPSILON * 0.6);
        assert_ne!(poly1, poly3);
    }

    #[test]
    fn test_invariants_are_maintained_after_mutation() {
        // Part 1: apply_phase maintains sort order
        let mut poly = ContinuousPhasePoly::new();
        let p2 = BooleanPoly::from_terms(smallvec![2]);
        let p1 = BooleanPoly::from_terms(smallvec![1]);

        poly.apply_phase(p2, 2.0);
        // This would break the sort order if compact() wasn't called
        poly.apply_phase(p1, 1.0);

        assert_eq!(poly.parities[0], BooleanPoly::from_terms(smallvec![1]));
        assert_eq!(poly.parities[1], BooleanPoly::from_terms(smallvec![2]));

        // Part 2: substitute maintains compaction
        let mut poly2 = ContinuousPhasePoly::new();
        let u_mask = 1 << 60;
        let p_u = BooleanPoly::from_terms(smallvec![u_mask]);
        let p_x1 = BooleanPoly::from_terms(smallvec![1 << 1]);

        poly2.apply_phase(p_u, 1.0);
        poly2.apply_phase(p_x1.clone(), 2.0);

        // Substitute u -> x1, which should cause the first term to merge with the second
        let e_poly = BooleanPoly::from_terms(smallvec![1 << 1]);
        poly2.substitute(u_mask, &e_poly);

        assert_eq!(poly2.parities.len(), 1, "Substitute should trigger compaction");
        assert_eq!(poly2.parities[0], p_x1);
        assert!((poly2.phases[0] - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_partial_eq_with_modular_wrap_around() {
        let p1 = BooleanPoly::from_terms(smallvec![1]);

        let mut poly1 = ContinuousPhasePoly::new();
        poly1.apply_phase(p1.clone(), EPSILON / 4.0);

        let mut poly2 = ContinuousPhasePoly::new();
        // This phase is close to TAU, which is equivalent to being close to 0
        poly2.apply_phase(p1.clone(), TAU - EPSILON / 4.0);

        // Because of the wrap-around logic in quantize_phase, these should be equal
        assert_eq!(poly1, poly2);

        // Their hashes must also be equal
        let mut hasher1 = DefaultHasher::new();
        poly1.hash(&mut hasher1);
        let hash1 = hasher1.finish();

        let mut hasher2 = DefaultHasher::new();
        poly2.hash(&mut hasher2);
        let hash2 = hasher2.finish();
        assert_eq!(hash1, hash2);
    }
}