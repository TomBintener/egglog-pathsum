use egglog::prelude::BaseSort;
use egglog::sort::{BaseValues, Boxed};
use crate::ast::Literal;
use super::*;

// A large prime for our Finite Field (2^61 - 1)
const PRIME: u64 = 2305843009213693951; 

// 1. Define the exact integer data structure
// Deriving Eq and Hash is what enables the O(N) Hash-Join in egglog!
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EvaluatedPathSum {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<u64>, 
}

pub type PSum = Boxed<EvaluatedPathSum>;

#[derive(Debug)]
pub struct PathSumSort;

// Helper function for matrix multiplication over F_p
// Extracting this avoids `add_primitive!` macro parsing errors with semicolons
fn combine_pathsum_logic(a: PSum, b: PSum) -> PSum {
    let mut result_data = vec![0; a.rows * b.cols];
    
    for i in 0..a.rows {
        for j in 0..b.cols {
            let mut sum: u128 = 0;
            for k in 0..a.cols {
                let val_a = a.data[i * a.cols + k] as u128;
                let val_b = b.data[k * b.cols + j] as u128;
                // Multiply, add, and modulo immediately to prevent overflow
                sum = (sum + (val_a * val_b) % (PRIME as u128)) % (PRIME as u128);
            }
            result_data[i * b.cols + j] = sum as u64;
        }
    }

    PSum::new(EvaluatedPathSum {
        rows: a.rows,
        cols: b.cols,
        data: result_data,
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
        
        // TODO: You will add base cases here like "h-pathsum", "cx-pathsum", etc.
    }

    fn reconstruct_termdag(&self, _base_values: &BaseValues, _value: Value, termdag: &mut TermDag) -> TermId {
        termdag.lit(Literal::String("<EvaluatedPathSum>".into()))
    }
}