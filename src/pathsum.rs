use egglog::prelude::BaseSort;
use egglog::sort::{BaseValues, Boxed};
use crate::ast::Literal;
use super::*;

// A large prime for our Finite Field (2^61 - 1)
const PRIME: u64 = 2305843009213693951; 

// A complex number over our Finite Field
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComplexModP {
    pub real: u64,
    pub imag: u64,
}

impl ComplexModP {
    // Helper for exact modular multiplication of two complex numbers
    // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    #[inline(always)]
    fn mul_mod(self, other: Self) -> Self {
        let ac = (self.real as u128 * other.real as u128) % (PRIME as u128);
        let bd = (self.imag as u128 * other.imag as u128) % (PRIME as u128);
        let ad = (self.real as u128 * other.imag as u128) % (PRIME as u128);
        let bc = (self.imag as u128 * other.real as u128) % (PRIME as u128);
        
        // To safely do (ac - bd) mod P, we add P before subtracting to prevent underflow
        let real = (ac + (PRIME as u128) - bd) % (PRIME as u128);
        let imag = (ad + bc) % (PRIME as u128);
        
        ComplexModP { real: real as u64, imag: imag as u64 }
    }

    #[inline(always)]
    fn add_mod(self, other: Self) -> Self {
        ComplexModP {
            real: (self.real + other.real) % PRIME,
            imag: (self.imag + other.imag) % PRIME,
        }
    }
}

// 1. Define the exact integer data structure
// Deriving Eq and Hash is what enables the O(N) Hash-Join in egglog!
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EvaluatedPathSum {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<ComplexModP>, 
}

pub type PSum = Boxed<EvaluatedPathSum>;

#[derive(Debug)]
pub struct PathSumSort;

// Helper function for matrix multiplication over F_p
// Extracting this avoids `add_primitive!` macro parsing errors with semicolons
fn combine_pathsum_logic(a: PSum, b: PSum) -> PSum {
    let mut result_data = vec![ComplexModP { real: 0, imag: 0 }; a.rows * b.cols];
    
    for i in 0..a.rows {
        for j in 0..b.cols {
            let mut sum = ComplexModP { real: 0, imag: 0 };
            for k in 0..a.cols {
                let val_a = a.data[i * a.cols + k];
                let val_b = b.data[k * b.cols + j];
                sum = sum.add_mod(val_a.mul_mod(val_b));
            }
            result_data[i * b.cols + j] = sum;
        }
    }

    PSum::new(EvaluatedPathSum {
        rows: a.rows,
        cols: b.cols,
        data: result_data,
    })
}

// Helper function for quickly creating constant gate matrices
fn constant_gate(rows: usize, cols: usize, data: &[u64]) -> PSum {
    let complex_data: Vec<ComplexModP> = data.iter().map(|&val| ComplexModP {
        real: val,
        imag: 0,
    }).collect();

    PSum::new(EvaluatedPathSum {
        rows,
        cols,
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
        add_primitive!(eg, "combine-pathsum" = |a: PSum, b: PSum| -> PSum { 
            combine_pathsum_logic(a, b) 
        });
        
        // Constant Base Cases
        add_primitive!(eg, "id-pathsum" = | | -> PSum {
            constant_gate(2, 2, &[1, 0, 0, 1])
        });

        add_primitive!(eg, "x-pathsum" = | | -> PSum {
            constant_gate(2, 2, &[0, 1, 1, 0])
        });

        add_primitive!(eg, "z-pathsum" = | | -> PSum {
            // In a Finite Field, -1 is represented as (PRIME - 1)
            constant_gate(2, 2, &[1, 0, 0, PRIME - 1])
        });

        add_primitive!(eg, "cx-pathsum" = | | -> PSum {
            constant_gate(4, 4, &[
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
        let id = constant_gate(2, 2, &[1, 0, 0, 1]);
        let x = constant_gate(2, 2, &[0, 1, 1, 0]);
        let z = constant_gate(2, 2, &[1, 0, 0, PRIME - 1]);
        let cx = constant_gate(4, 4, &[
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 0, 1,
            0, 0, 1, 0,
        ]);
        let id4 = constant_gate(4, 4, &[
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
        let expected_xz = constant_gate(2, 2, &[0, PRIME - 1, 1, 0]);
        assert_eq!(x_z.data, expected_xz.data, "X * Z matrix is mathematically incorrect");

        // Z*X = [0, 1; -1, 0] (Using PRIME - 1 for -1)
        let expected_zx = constant_gate(2, 2, &[0, 1, PRIME - 1, 0]);
        assert_eq!(z_x.data, expected_zx.data, "Z * X matrix is mathematically incorrect");
        
        // Ensure they correctly evaluated to different matrices!
        assert_ne!(x_z.data, z_x.data, "X*Z should NOT equal Z*X");
    }
}