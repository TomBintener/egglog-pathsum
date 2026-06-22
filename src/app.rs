use crate::bridge::{
    id_pathsum_logic_64, apply_h_logic_64, apply_cx_logic_64, apply_t_logic_64, apply_s_logic_64, apply_z_logic_64, apply_x_logic_64, apply_sdg_logic_64, apply_tdg_logic_64, apply_sx_logic_64, apply_rz_logic_64
};
use crate::bridge::{PSum64, PSum128};
use pyo3::prelude::*;

#[pyfunction]
fn id_pathsum(num_qubits: i64) -> PSum64 {
    id_pathsum_logic_64(num_qubits)
}

#[pyfunction]
fn apply_h(state: PSum64, q: i64) -> PSum64 {
    apply_gate_logic_64(state, q, |st, q_| st.apply_h(q_))
}

#[pyfunction]
fn apply_cx(state: PSum64, qc: i64, qt: i64) -> PSum64 {
    apply_cx_logic_64(state, qc, qt)
}

#[pyfunction]
fn apply_t(state: PSum64, q: i64) -> PSum64 {
    apply_gate_logic_64(state, q, |st, q_| st.apply_t(q_))
}

#[pyfunction]
fn apply_s(state: PSum64, q: i64) -> PSum64 {
    apply_gate_logic_64(state, q, |st, q_| st.apply_s(q_))
}

#[pyfunction]
fn apply_z(state: PSum64, q: i64) -> PSum64 {
    apply_gate_logic_64(state, q, |st, q_| st.apply_z(q_))
}

#[pyfunction]
fn apply_x(state: PSum64, q: i64) -> PSum64 {
    apply_gate_logic_64(state, q, |st, q_| st.apply_x(q_))
}

#[pyfunction]
fn apply_sdg(state: PSum64, q: i64) -> PSum64 {
    apply_gate_logic_64(state, q, |st, q_| st.apply_sdg(q_))
}

#[pyfunction]
fn apply_tdg(state: PSum64, q: i64) -> PSum64 {
    apply_gate_logic_64(state, q, |st, q_| st.apply_tdg(q_))
}

#[pyfunction]
fn apply_sx(state: PSum64, q: i64) -> PSum64 {
    apply_gate_logic_64(state, q, |st, q_| st.apply_sx(q_))
}

#[pyfunction]
fn apply_rz(state: PSum64, q: i64, theta_bits: i64) -> PSum64 {
    apply_rz_logic_64(state, q, theta_bits)
}

#[pymodule]
fn egglog_experimental(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(id_pathsum, m)?)?;
    m.add_function(wrap_pyfunction!(apply_h, m)?)?;
    m.add_function(wrap_pyfunction!(apply_cx, m)?)?;
    m.add_function(wrap_pyfunction!(apply_t, m)?)?;
    m.add_function(wrap_pyfunction!(apply_s, m)?)?;
    m.add_function(wrap_pyfunction!(apply_z, m)?)?;
    m.add_function(wrap_pyfunction!(apply_x, m)?)?;
    m.add_function(wrap_pyfunction!(apply_sdg, m)?)?;
    m.add_function(wrap_pyfunction!(apply_tdg, m)?)?;
    m.add_function(wrap_pyfunction!(apply_sx, m)?)?;
    m.add_function(wrap_pyfunction!(apply_rz, m)?)?;
    m.add_class::<PSum64>()?;
    m.add_class::<PSum128>()?;
    Ok(())
}
