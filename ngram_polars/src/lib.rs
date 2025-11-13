use pyo3_polars::PolarsAllocator;

mod expression;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();
