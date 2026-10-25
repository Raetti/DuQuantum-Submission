# DuQuantum 2026 — Google Quantum AI Dicke State Submission

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Raetti/DuQuantum-Submission/blob/main/DuQuantum_Google_Standard.ipynb)

Divide-and-Conquer Dicke state preparation ($|D^{(N)}_n\rangle$) and calibration-aware 2D Ladder / 1D LNN grid selection for the Google `Willow105` superconducting processor.

## Open in Google Colab (One-Click Verification)

Click the **Open in Colab** badge above, or open the link directly:
👉 **[Open DuQuantum_Google_Standard.ipynb in Colab](https://colab.research.google.com/github/Raetti/DuQuantum-Submission/blob/main/DuQuantum_Google_Standard.ipynb)**

The notebook is pre-configured to:
1. `!pip install cirq cirq-google` automatically in the first cell.
2. Run noiseless statevector fidelity tests confirming $|\\langle D^{(N)}_n | \\psi \\rangle|^2 = 1.00000000$ across all $(N, n)$.
3. Audit all operations to guarantee strictly native `cirq.CZ` and `cirq.PhasedXZGate` gates (0 CNOTs, 0 SWAPs).
4. Simulate 2,000 projective Z-basis measurements confirming exact $\\frac{1}{\\binom{N}{n}}$ distribution with zero leakage.
5. Simulate noisy density matrices to measure $\\langle D | \\rho | D \\rangle$ under depolarizing and thermal noise.

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
