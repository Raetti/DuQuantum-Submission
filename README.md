# DuQuantum 2026 — Google Quantum AI Dicke State Submission

Divide-and-Conquer Dicke state preparation ($|D^{(N)}_n\rangle$) and calibration-aware 2D Ladder / 1D LNN grid selection for the Google `Willow105` superconducting processor.

## Judge Submission Files

- **`DuQuantum_Google_Standard.py`**: Standalone, self-contained Python submission file containing `create_dicke_state`, `create_dicke_state_adaptive`, and all official DuQuantum validation and scoring functions (`test_circuit`, `create_dicke_statevector`, `check_z_bitstring_distribution`, `check_x_bitstring_distribution`, `overall_score`, `score_solution`, and synthetic calibration generators). Imports only `cirq`, `cirq_google`, their dependencies, and Python's standard library.
- **`DuQuantum_Google_Standard.ipynb`**: Self-contained Colab/Jupyter notebook version of the submission.

## Quick Start

```bash
pip install -r requirements.txt

# Run the standalone judge submission file
python DuQuantum_Google_Standard.py

# Run the unit test suite (Dicke state fidelity + grid selection)
pytest tests/ -v
```

## Module Layout

- `src/dicke_state.py`: Ladder (Section III-B) and LNN (Section III-C) Divide-and-Conquer Dicke state preparation (Aktar et al., arXiv:2112.12435) with KAK lowering to native `cirq.CZ` and `cirq.PhasedXZGate` operations.
- `src/willow_utils.py`: `Willow105` connectivity graph construction, log-polarization calibration weighting, state-dependent marginal readout weighting, and chain/subgraph selection.
- `src/scoring.py`: Official DuQuantum statevector verification and $Z$/$X$-basis bitstring scoring functions.
- `src/noise_models.py`: Synthetic `Willow105` calibration error rate and dropout generators.
