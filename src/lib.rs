//! # egglog-experimental
//!
//! This crate layers several experimental features on top of the core
//! [`egglog`](https://github.com/egraphs-good/egglog) language and runtime.
//! It can serve as a standard library when building equality
//! saturation workflows in Rust.
//!
//! ## Implemented extensions
//!
//! - [`for`-loops](https://egraphs-good.github.io/egglog-demo/?example=for)
//! - [`with-ruleset`](https://egraphs-good.github.io/egglog-demo/?example=with-ruleset)
//! - [Rationals support](https://egraphs-good.github.io/egglog-demo/?example=rational)
//!   (see [`rational`] for the exposed primitives)
//! - [Dynamic cost models with `set-cost`](https://egraphs-good.github.io/egglog-demo/?example=05-cost-model-and-extraction)
//! - [Custom schedulers via `run-with`](https://egraphs-good.github.io/egglog-demo/?example=math-backoff)
//! - [`(get-size!)` primitive](https://github.com/egraphs-good/egglog-experimental/blob/main/tests/web-demo/node-limit.egg)
//!   for inspecting total tuple counts or counts for specific tables
//! - [Multi-extraction](https://github.com/egraphs-good/egglog-experimental/blob/main/tests/web-demo/multi-extract.egg)
//!
//! Each bullet links to a runnable demo so you can explore the feature quickly.
//! The rest of this crate exposes the Rust APIs and helpers that back these extensions.
//!
use egglog::ast::Parser;
use egglog::prelude::{RustSpan, Span, add_base_sort};
pub use egglog::*;
use std::sync::Arc;

pub mod rational;
pub use rational::*;
mod scheduling;
pub use scheduling::*;
mod fresh_macro;

mod set_cost;
pub use set_cost::*;
mod multi_extract;
pub use multi_extract::*;
mod size;
pub use size::*;
mod pathsum;
pub use pathsum::*;

// Sugar modules using parse-time macros
mod sugar;
pub use sugar::*;

pub fn new_experimental_egraph() -> EGraph {
    let mut egraph = EGraph::default();

    // Set up the parser with experimental parse-time macros
    egraph.parser = experimental_parser();

    // Rational support
    add_base_sort(&mut egraph, RationalSort, span!()).unwrap();

    // Support for set cost
    add_set_cost(&mut egraph);
    egraph.add_read_primitive(GetSizePrimitive, None);

    // Register the PathSum custom Sort for our exact Hash-Joins
    add_base_sort(&mut egraph, PathSumSort, span!()).unwrap();

    // unstable-fresh! macro
    egraph
        .command_macros_mut()
        .register(Arc::new(fresh_macro::FreshMacro::new()));

    // scheduler support
    egraph
        .add_command("run-schedule".into(), Arc::new(RunExtendedSchedule))
        .unwrap();

    egraph
        .add_command(
            "multi-extract".into(),
            Arc::new(MultiExtract::new(DynamicCostModel)),
        )
        .unwrap();
    egraph
}

// Create a parser with experimental macros
pub fn experimental_parser() -> Parser {
    let mut parser = Parser::default();
    parser.add_command_macro(Arc::new(sugar::For));
    parser.add_command_macro(Arc::new(sugar::WithRuleset));
    parser
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pathsum_addon_recognition() {
        let mut eg = new_experimental_egraph();
        
        let program = r#"
            ;; 1. Check if the engine recognizes the custom Sort "PathSum"
            ;; If PathSumSort wasn't registered, this will panic.
            (function test-pathsum-table (i64) PathSum :merge old)
            
            ;; 2. Create dummy inputs to feed into the primitive
            (function dummy-ps (i64) PathSum :merge old)

            ;; 3. Force the engine to typecheck the primitive inside a rule.
            ;; The rule won't actually fire because there's no data, but it will 
            ;; successfully compile and verify the primitive's type signature!
            (rule ((= a (dummy-ps 1))
                   (= b (dummy-ps 2)))
                  ((let test-evaluation (combine-pathsum a b))))
        "#;
        
        if let Err(e) = eg.parse_and_run_program(None, program) {
            panic!("Egglog engine failed!\nDetailed Error:\n{:#?}", e);
        }
    }
}
