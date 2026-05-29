use egglog::prelude::BaseSort;
use egglog::sort::{BaseValues, Boxed};
use egglog::ast::Literal;
use smallvec::{smallvec, SmallVec};
use egglog::*;

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
    #[allow(dead_code)]
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
        add_primitive!(eg, "rust_combine_pathsum" = |a: PSum, b: PSum| -> PSum { 
            combine_pathsum_logic(b, a) 
        });

        // Primitive for parallel gates
        add_primitive!(eg, "rust_tensor_pathsum" = |a: PSum, b: PSum| -> PSum { 
            tensor_pathsum_logic(a, b) 
        });
        
        // Constant Base Cases
        add_primitive!(eg, "rust_id_pathsum" = |q: i64| -> PSum {
            constant_gate(2, 2, &[q], &[1, 0, 0, 1])
        });

        add_primitive!(eg, "rust_x_pathsum" = |q: i64| -> PSum {
            constant_gate(2, 2, &[q], &[0, 1, 1, 0])
        });

        add_primitive!(eg, "rust_z_pathsum" = |q: i64| -> PSum {
            // In a Finite Field, -1 is represented as (PRIME - 1)
            constant_gate(2, 2, &[q], &[1, 0, 0, PRIME - 1])
        });

        add_primitive!(eg, "rust_cx_pathsum" = |qc: i64, qt: i64| -> PSum {
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
    fn test_native_math_logic() {
        // 1. Pure Rust execution to verify matrix logic without the E-Graph overhead
        let cx = constant_gate(4, 4, &[0, 1], &[
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 0, 1,
            0, 0, 1, 0,
        ]);

        // CX * CX = ID
        let cx_squared = combine_pathsum_logic(cx.clone(), cx.clone());
        let id4 = constant_gate(4, 4, &[0, 1], &[
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ]);

        assert_eq!(cx_squared.data, id4.data, "Matrix multiplication logic failed!");
    }

    #[test]
    fn test_egglog_engine_integration() {
        // 2. Execute the primitives natively inside the actual Egglog engine
        // We use the crate's bootstrapper to ensure all standard sorts (i64, etc.) are loaded
        let mut eg = crate::new_experimental_egraph();

        let program = r#"
            ;; Create three identical CNOT matrices using our registered Rust primitives
            (let cx1 (rust_cx_pathsum 0 1))
            (let cx2 (rust_cx_pathsum 0 1))
            (let cx3 (rust_cx_pathsum 0 1))

            ;; Multiply them together: cx3 * (cx2 * cx1)
            (let seq (rust_combine_pathsum cx3 (rust_combine_pathsum cx2 cx1)))

            ;; Assert that the e-graph successfully evaluated the math natively
            ;; and realized that CX * CX * CX == CX
            (check (= seq cx1))
        "#;

        let result = eg.parse_and_run_program(None, program);
        assert!(result.is_ok(), "Egglog failed to evaluate the native Rust program: {:?}", result);
    }
}