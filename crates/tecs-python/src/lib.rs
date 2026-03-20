use pyo3::prelude::*;
use pyo3::types::PyDict;
use tecs_core::topology::{PersistenceComputer, PersistenceResult};
use tecs_core::pruner::graph_distance_matrix;
use tecs_core::hyperbolic::hyperbolicity;
use nalgebra::DMatrix;

#[pyclass(frozen)]
struct RustEngine {}

#[pymethods]
impl RustEngine {
    #[new]
    fn new() -> Self {
        Self {}
    }

    fn build_candidate_graph<'py>(
        &self,
        py: Python<'py>,
        seed_ids: Vec<String>,
        hops: usize,
        epsilon: f64,
        max_nodes: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("seed_entities", &seed_ids)?;
        out.set_item("hops", hops)?;
        out.set_item("epsilon", epsilon)?;
        out.set_item("max_nodes", max_nodes)?;
        out.set_item("candidate_cycle_nodes", vec!["Q1", "Q2", "Q3", "Q4"])?;
        out.set_item("bridge_candidates", vec!["Q_mediator_1", "Q_mediator_2"])?;
        Ok(out)
    }

    /// Backward-compatible stub: compute topology from a graph bundle dict.
    fn compute_topology<'py>(
        &self,
        py: Python<'py>,
        _graph_bundle: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("beta0", 3)?;
        out.set_item("beta1", 1)?;
        out.set_item("long_h1", vec![(0.3, 1.2)])?;
        Ok(out)
    }

    /// Compute topology from a distance matrix (list of list of floats).
    /// Uses tecs-core's VietorisRipsBuilder and PersistenceComputer.
    /// Releases the GIL during computation.
    fn compute_topology_from_matrix<'py>(
        &self,
        py: Python<'py>,
        distance_matrix: Vec<Vec<f64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let n = distance_matrix.len();
        if n == 0 {
            let out = PyDict::new(py);
            out.set_item("beta0", 0)?;
            out.set_item("beta1", 0)?;
            out.set_item("long_h1", Vec::<(f64, f64)>::new())?;
            return Ok(out);
        }

        // Flatten the distance matrix
        let flat: Vec<f64> = distance_matrix.iter().flat_map(|row| row.iter().copied()).collect();

        // Run computation with GIL released
        let result: PersistenceResult = py.detach(move || {
            let dm = DMatrix::from_row_slice(n, n, &flat);
            let max_radius = PersistenceComputer::auto_radius(&dm);
            let computer = PersistenceComputer::new(2, max_radius);
            computer.compute(&dm)
        });

        // Extract Betti numbers
        let beta0 = result.unpaired.iter().filter(|u| u.dimension == 0).count();

        // Count H₁ bars with persistence > threshold (not unpaired — always 0 for finite VR)
        let persistence_threshold = 0.01;
        let h1_bars: Vec<(f64, f64)> = result.pairs.iter()
            .filter(|p| p.dimension == 1 && p.persistence > persistence_threshold)
            .map(|p| (p.birth, p.death))
            .collect();
        let beta1 = h1_bars.len();
        let max_persistence_h1 = h1_bars.iter()
            .map(|(b, d)| d - b)
            .fold(0.0_f64, f64::max);
        let n_h1_bars = result.pairs.iter().filter(|p| p.dimension == 1).count();

        let out = PyDict::new(py);
        out.set_item("beta0", beta0)?;
        out.set_item("beta1", beta1)?;
        out.set_item("long_h1", &h1_bars)?;
        out.set_item("max_persistence_h1", max_persistence_h1)?;
        out.set_item("n_h1_bars", n_h1_bars)?;
        Ok(out)
    }
    /// Compute topology from graph edges (graph shortest-path distance → VR → PH)
    fn compute_topology_from_edges<'py>(
        &self,
        py: Python<'py>,
        edges: Vec<(usize, usize)>,
        n_nodes: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        // Run computation with GIL released
        let result: PersistenceResult = py.detach(move || {
            let dm = graph_distance_matrix(&edges, n_nodes);
            let max_radius = PersistenceComputer::auto_radius(&dm);
            let computer = PersistenceComputer::new(2, max_radius);
            computer.compute(&dm)
        });

        // Extract Betti numbers
        let beta0 = result.unpaired.iter().filter(|u| u.dimension == 0).count();

        // Count H₁ bars with persistence > threshold
        let persistence_threshold = 0.01;
        let h1_bars: Vec<(f64, f64)> = result.pairs.iter()
            .filter(|p| p.dimension == 1 && p.persistence > persistence_threshold)
            .map(|p| (p.birth, p.death))
            .collect();
        let beta1 = h1_bars.len();
        let max_persistence_h1 = h1_bars.iter()
            .map(|(b, d)| d - b)
            .fold(0.0_f64, f64::max);
        let n_h1_bars = result.pairs.iter().filter(|p| p.dimension == 1).count();

        let out = PyDict::new(py);
        out.set_item("beta0", beta0)?;
        out.set_item("beta1", beta1)?;
        out.set_item("long_h1", &h1_bars)?;
        out.set_item("max_persistence_h1", max_persistence_h1)?;
        out.set_item("n_h1_bars", n_h1_bars)?;
        Ok(out)
    }

    /// Compute Gromov δ-hyperbolicity from graph edges.
    /// Returns {"delta": f64, "hierarchy_score": f64}
    fn compute_hyperbolicity<'py>(
        &self,
        py: Python<'py>,
        edges: Vec<(usize, usize)>,
        n_nodes: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (delta, score) = py.detach(move || {
            let dm = graph_distance_matrix(&edges, n_nodes);
            let delta = hyperbolicity::delta_hyperbolicity(&dm);
            let score = hyperbolicity::hierarchy_score(delta);
            (delta, score)
        });
        let out = PyDict::new(py);
        out.set_item("delta", delta)?;
        out.set_item("hierarchy_score", score)?;
        Ok(out)
    }
}

#[pymodule]
fn tecs_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustEngine>()?;
    Ok(())
}
