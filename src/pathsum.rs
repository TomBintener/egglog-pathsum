use egglog::prelude::BaseSort;
use egglog::sort::{BaseValues, Boxed};
use crate::ast::Literal;
use smallvec::{smallvec, SmallVec};
use super::*;

// A large prime for our Finite Field (2^61 - 1)
const PRIME: u64 = 2305843009213693951; 

// Helper function for ultra-fast modulo arithmetic for Mersenne Prime 2^61 - 1
// Replaces slow hardware division (%) with lightning-fast bitwise shifts
#[inline(always)]
fn fast_mod(mut v: u128) -> u64 {
    const PRIME_U128: u128 = 2305843009213693951;
    v = (v & PRIME_U128) + (v >> 61);
    v = (v & PRIME_U128) + (v >> 61);
    if v >= PRIME_U128 {
        v -= PRIME_U128;
    }
    v as u64
}

// A complex number over our Finite Field
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComplexModP {
    pub real: u64,
    pub imag: u64,
}

impl ComplexModP {
    pub const ZERO: Self = Self { real: 0, imag: 0 };
    pub const ONE: Self = Self { real: 1, imag: 0 };

    // Helper for exact modular multiplication of two complex numbers
    // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    #[inline(always)]
    fn mul_mod(self, other: Self) -> Self {
        let ac = self.real as u128 * other.real as u128;
        let bd = self.imag as u128 * other.imag as u128;
        let ad_bc = self.real as u128 * other.imag as u128 + self.imag as u128 * other.real as u128;
        
        let r_pos = fast_mod(ac);
        let r_neg = fast_mod(bd);
        
        let real = if r_pos >= r_neg { r_pos - r_neg } else { PRIME + r_pos - r_neg };
        
        ComplexModP { real, imag: fast_mod(ad_bc) }
    }

    #[inline(always)]
    fn add_mod(self, other: Self) -> Self {
        let mut real = self.real + other.real;
        if real >= PRIME { real -= PRIME; }
        
        let mut imag = self.imag + other.imag;
        if imag >= PRIME { imag -= PRIME; }
        
        ComplexModP { real, imag }
    }
}

impl std::ops::Add for ComplexModP {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output { self.add_mod(rhs) }
}

impl std::ops::Mul for ComplexModP {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output { self.mul_mod(rhs) }
}

// 1. Define the exact integer data structure
// Deriving Eq and Hash is what enables the O(N) Hash-Join in egglog!
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EvaluatedPathSum {
    pub rows: usize,
    pub cols: usize,
    pub qubits: SmallVec<[i64; 4]>,
    pub data: SmallVec<[ComplexModP; 16]>,
}

pub type PSum = Boxed<EvaluatedPathSum>;

#[derive(Debug)]
pub struct PathSumSort;

// Helper function for matrix multiplication over F_p
// Extracting this avoids `add_primitive!` macro parsing errors with semicolons
fn combine_pathsum_logic(a: PSum, b: PSum) -> PSum {
    assert_eq!(a.cols, b.rows, "Matrix dimension mismatch: {}x{} * {}x{}", a.rows, a.cols, b.rows, b.cols);
    
    // Safety check: Ensure we are multiplying matrices on the exact same qubits!
    assert_eq!(a.qubits, b.qubits, "Cannot directly multiply matrices on different qubit sets! Qubits must be aligned first.");

    let mut result_data = smallvec![ComplexModP::ZERO; a.rows * b.cols];
    for i in 0..a.rows {
        for j in 0..b.cols {
            let mut real_pos: u128 = 0;
            let mut real_neg: u128 = 0;
            let mut imag_pos: u128 = 0;
            
            for k in 0..a.cols {
                let val_a = a.data[i * a.cols + k];
                let val_b = b.data[k * b.cols + j];
                
                real_pos += val_a.real as u128 * val_b.real as u128;
                real_neg += val_a.imag as u128 * val_b.imag as u128;
                imag_pos += val_a.real as u128 * val_b.imag as u128 + val_a.imag as u128 * val_b.real as u128;
                
                // Prevent u128 overflow for matrices larger than 16x16 (4 qubits)
                if (k & 15) == 15 {
                    real_pos = fast_mod(real_pos) as u128;
                    real_neg = fast_mod(real_neg) as u128;
                    imag_pos = fast_mod(imag_pos) as u128;
                }
            }
            
            let r_pos = fast_mod(real_pos);
            let r_neg = fast_mod(real_neg);
            let real = if r_pos >= r_neg { r_pos - r_neg } else { PRIME + r_pos - r_neg };
            
            result_data[i * b.cols + j] = ComplexModP { real, imag: fast_mod(imag_pos) };
        }
    }

    PSum::new(EvaluatedPathSum {
        rows: a.rows,
        cols: b.cols,
        qubits: a.qubits.clone(),
        data: result_data,
    })
}

// Helper function for Tensor (Kronecker) Product of two matrices
// Used when gates are applied in parallel on different qubits
fn tensor_pathsum_logic(a: PSum, b: PSum) -> PSum {
    // Enforce a canonical order to ensure that tensor(A, B) == tensor(B, A)
    // This is critical for the e-graph to recognize equivalent parallel operations.
    let (first, second) = if a.qubits.iter().min() < b.qubits.iter().min() {
        (a, b)
    } else {
        (b, a)
    };

    // Safety check: Ensure qubit sets are disjoint.
    for q_a in first.qubits.iter() {
        assert!(!second.qubits.contains(q_a), "Cannot tensor matrices with overlapping qubits: {:?} and {:?}", first.qubits, second.qubits);
    }

    let rows = first.rows * second.rows;
    let cols = first.cols * second.cols;
    
    let mut new_qubits = first.qubits.clone();
    new_qubits.extend_from_slice(&second.qubits);

    let mut result_data = smallvec![ComplexModP::ZERO; rows * cols];

    for i in 0..first.rows {
        for j in 0..first.cols {
            let val_a = first.data[i * first.cols + j];
            for k in 0..second.rows {
                for l in 0..second.cols {
                    let val_b = second.data[k * second.cols + l];
                    result_data[(i * second.rows + k) * cols + (j * second.cols + l)] = val_a * val_b;
                }
            }
        }
    }
    PSum::new(EvaluatedPathSum {
        rows,
        cols,
        qubits: new_qubits,
        data: result_data,
    })
}

// Helper function for quickly creating constant gate matrices
fn constant_gate(rows: usize, cols: usize, qubits: &[i64], data: &[u64]) -> PSum {
    let complex_data: SmallVec<[ComplexModP; 16]> = data.iter().map(|&val| ComplexModP {
        real: val,
        imag: 0,
    }).collect();

    PSum::new(EvaluatedPathSum {
        rows,
        cols,
        qubits: SmallVec::from_slice(qubits),
        data: complex_data,
    })
}

// 2. Teach the database about the data structure
impl BaseSort for PathSumSort {
    type Base = PSum;

    fn name(&self) -> &str {
        "PathSum"
    }

    fn register_primitives(&self, eg: &mut EGraph) {
        // The exact modular arithmetic primitive for combining sequences
        // IMPORTANT: We flip `a` and `b` because a sequence `then(A, B)` evaluates as B * A
        add_primitive!(eg, "combine-pathsum" = |a: PSum, b: PSum| -> PSum { 
            combine_pathsum_logic(b, a) 
        });

        // Primitive for parallel gates
        add_primitive!(eg, "tensor-pathsum" = |a: PSum, b: PSum| -> PSum { 
            tensor_pathsum_logic(a, b) 
        });
        
        // Constant Base Cases
        add_primitive!(eg, "id-pathsum" = |q: i64| -> PSum {
            constant_gate(2, 2, &[q], &[1, 0, 0, 1])
        });

        add_primitive!(eg, "x-pathsum" = |q: i64| -> PSum {
            constant_gate(2, 2, &[q], &[0, 1, 1, 0])
        });

        add_primitive!(eg, "z-pathsum" = |q: i64| -> PSum {
            // In a Finite Field, -1 is represented as (PRIME - 1)
            constant_gate(2, 2, &[q], &[1, 0, 0, PRIME - 1])
        });

        add_primitive!(eg, "cx-pathsum" = |qc: i64, qt: i64| -> PSum {
            constant_gate(4, 4, &[qc, qt], &[
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 0, 1,
                0, 0, 1, 0,
            ])
        });
    }

    fn reconstruct_termdag(&self, _base_values: &BaseValues, _value: Value, termdag: &mut TermDag) -> TermId {
        termdag.lit(Literal::String("<EvaluatedPathSum>".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_mod_p_arithmetic() {
        let a = ComplexModP { real: 2, imag: 3 };
        let b = ComplexModP { real: 4, imag: 5 };
        
        // Addition: (2 + 3i) + (4 + 5i) = 6 + 8i
        let sum = a.add_mod(b);
        assert_eq!(sum, ComplexModP { real: 6, imag: 8 });

        // Multiplication: (2 + 3i) * (4 + 5i) = (8 - 15) + (10 + 12)i = -7 + 22i
        // In our finite field, -7 is represented as PRIME - 7
        let prod = a.mul_mod(b);
        assert_eq!(prod, ComplexModP { real: PRIME - 7, imag: 22 });

        // The Imaginary Unit: i * i = -1 (which is PRIME - 1)
        let i = ComplexModP { real: 0, imag: 1 };
        let i_squared = i.mul_mod(i);
        assert_eq!(i_squared, ComplexModP { real: PRIME - 1, imag: 0 });
    }

    #[test]
    fn test_gate_matrix_multiplication() {
        // Generate our base gates
        let id = constant_gate(2, 2, &[0], &[1, 0, 0, 1]);
        let x = constant_gate(2, 2, &[0], &[0, 1, 1, 0]);
        let z = constant_gate(2, 2, &[0], &[1, 0, 0, PRIME - 1]);
        let cx = constant_gate(4, 4, &[0, 1], &[
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 0, 1,
            0, 0, 1, 0,
        ]);
        let id4 = constant_gate(4, 4, &[0, 1], &[
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ]);

        // Test 1: X * X = Identity (2x2)
        let x_squared = combine_pathsum_logic(x.clone(), x.clone());
        assert_eq!(x_squared.data, id.data, "X * X did not equal Identity");
        assert_eq!(x_squared.rows, 2);
        assert_eq!(x_squared.cols, 2);

        // Test 2: Z * Z = Identity (2x2)
        let z_squared = combine_pathsum_logic(z.clone(), z.clone());
        assert_eq!(z_squared.data, id.data, "Z * Z did not equal Identity");

        // Test 3: X * Identity (2x2) = X
        let x_id = combine_pathsum_logic(x.clone(), id.clone());
        assert_eq!(x_id.data, x.data, "X * I did not equal X");

        // Test 4: CX * CX = Identity (4x4)
        let cx_squared = combine_pathsum_logic(cx.clone(), cx.clone());
        assert_eq!(cx_squared.data, id4.data, "CX * CX did not equal 4x4 Identity");
        assert_eq!(cx_squared.rows, 4);
        assert_eq!(cx_squared.cols, 4);

        // Test 5: Pauli Anti-Commutation (X * Z = -Z * X)
        let x_z = combine_pathsum_logic(x.clone(), z.clone());
        let z_x = combine_pathsum_logic(z.clone(), x.clone());
        
        // X*Z = [0, -1; 1, 0] (Using PRIME - 1 for -1)
        let expected_xz = constant_gate(2, 2, &[0], &[0, PRIME - 1, 1, 0]);
        assert_eq!(x_z.data, expected_xz.data, "X * Z matrix is mathematically incorrect");

        // Z*X = [0, 1; -1, 0] (Using PRIME - 1 for -1)
        let expected_zx = constant_gate(2, 2, &[0], &[0, 1, PRIME - 1, 0]);
        assert_eq!(z_x.data, expected_zx.data, "Z * X matrix is mathematically incorrect");
        
        // Ensure they correctly evaluated to different matrices!
        assert_ne!(x_z.data, z_x.data, "X*Z should NOT equal Z*X");
    }

    #[test]
    fn test_canonical_tensor_product() {
        let x0 = constant_gate(2, 2, &[0], &[0, 1, 1, 0]);
        let z1 = constant_gate(2, 2, &[1], &[1, 0, 0, PRIME - 1]);

        // Test that tensor(X(0), Z(1)) produces the same result as tensor(Z(1), X(0))
        let xz_tensor = tensor_pathsum_logic(x0.clone(), z1.clone());
        let zx_tensor = tensor_pathsum_logic(z1.clone(), x0.clone());

        assert_eq!(xz_tensor.qubits, zx_tensor.qubits, "Canonical qubit ordering failed");
        assert_eq!(xz_tensor.data, zx_tensor.data, "Tensor product result should be identical regardless of argument order");
        assert_eq!(xz_tensor.qubits.as_slice(), &[0, 1], "Qubits should be sorted");
    }

    #[test]
    fn test_matrix_multiplication_throughput() {
        use std::time::Instant;
        use std::hint::black_box;

        let iterations = 100_000;

        // 1. Benchmark 2x2 Matrices (e.g., X gate)
        let x_gate = constant_gate(2, 2, &[0], &[0, 1, 1, 0]);
        let mut current_2x2 = x_gate.clone();
        
        let start_2x2 = Instant::now();
        for _ in 0..iterations {
            // black_box prevents the compiler from optimizing the loop away
            current_2x2 = combine_pathsum_logic(black_box(current_2x2), black_box(x_gate.clone()));
        }
        let duration_2x2 = start_2x2.elapsed();
        let ops_per_sec_2x2 = (iterations as f64) / duration_2x2.as_secs_f64();

        // 2. Benchmark 4x4 Matrices (e.g., CX gate)
        let cx_gate = constant_gate(4, 4, &[0, 1], &[
            1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 0, 1,  0, 0, 1, 0,
        ]);
        let mut current_4x4 = cx_gate.clone();

        let start_4x4 = Instant::now();
        for _ in 0..iterations {
            current_4x4 = combine_pathsum_logic(black_box(current_4x4), black_box(cx_gate.clone()));
        }
        let duration_4x4 = start_4x4.elapsed();
        let ops_per_sec_4x4 = (iterations as f64) / duration_4x4.as_secs_f64();

        // Print results to stdout
        println!("\n--- Performance Test Results ({} iterations) ---", iterations);
        println!("2x2 Matrix Throughput: {:.2} ops/sec", ops_per_sec_2x2);
        println!("2x2 Time per mult:     {:?}", duration_2x2 / iterations);
        println!("--------------------------------------------------");
        println!("4x4 Matrix Throughput: {:.2} ops/sec", ops_per_sec_4x4);
        println!("4x4 Time per mult:     {:?}", duration_4x4 / iterations);
        println!("--------------------------------------------------\n");

        // Use the final values so the compiler doesn't throw them out
        assert_eq!(current_2x2.rows, 2);
        assert_eq!(current_4x4.rows, 4);
    }

    #[test]
    fn test_qubit_awareness() {
        // 1. Create two identical gates on different qubits
        let x_on_q0 = constant_gate(2, 2, &[0], &[0, 1, 1, 0]);
        let x_on_q1 = constant_gate(2, 2, &[1], &[0, 1, 1, 0]);

        // 2. Assert they are NOT equal because their qubit lists differ.
        // This proves the `qubits` field is part of the Hash and Eq implementations.
        assert_ne!(x_on_q0, x_on_q1, "Gates on different qubits should not be equal");

        // 3. Assert that attempting to multiply gates on different qubits panics.
        // This verifies the safety check in `combine_pathsum_logic`.
        
        // Temporarily suppress the panic output so it doesn't pollute our test logs
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        let result = std::panic::catch_unwind(|| {
            combine_pathsum_logic(x_on_q0, x_on_q1);
        });

        // Restore the standard panic hook immediately after
        std::panic::set_hook(prev_hook);

        assert!(result.is_err(), "Multiplying matrices on different qubits should panic");
    }
}