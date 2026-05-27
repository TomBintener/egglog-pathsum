use std::convert::TryFrom;

use egglog::{
    Primitive,
    constraint::{AllEqualTypeConstraint, TypeConstraint},
    prelude::BaseSort,
    prelude::{I64Sort, Span, StringSort},
};

#[derive(Clone)]
pub struct GetSizePrimitive;

impl Primitive for GetSizePrimitive {
    fn name(&self) -> &str {
        "get-size!"
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        AllEqualTypeConstraint::new(self.name(), span.clone())
            .with_output_sort(I64Sort.to_arcsort())
            .with_all_arguments_sort(StringSort.to_arcsort())
            .into_box()
    }

    fn apply(
        &self,
        egraph: &mut egglog::ExecutionState<'_>,
        values: &[egglog::Value],
    ) -> Option<egglog::Value> {
        let size: usize = match values {
            _ => 0,
        };
        let size = i64::try_from(size).ok()?;
        Some(egraph.base_values().get::<i64>(size))
    }
}
