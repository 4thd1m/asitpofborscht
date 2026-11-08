use pyo3::prelude::*;
mod kernelscpu;
mod kernelsgpu;

// mod tensor;
// mod engine;

/// A Python module implemented in Rust.
#[pymodule]
mod _borscht {
    use pyo3::prelude::*;

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }
}
