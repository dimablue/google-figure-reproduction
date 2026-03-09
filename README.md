# Repetition-Code Scaling Simulation

Partial reproduction of Google's Fig. 3a (repetition-code scaling) using Qiskit Aer,
with an optional rare 28-qubit correlated burst channel. Done for Physics/Computer Science C191 (Quantum Computing) at UC Berkeley.

Simulates a distance-d repetition code over 1000 cycles, sweeps code distances
d = 3...29, and plots logical error per cycle (ε_d) vs. code distance to extract
the error suppression factor Λ.

## What it does

- Builds a repetition-code memory circuit with Pauli-X noise per cycle
- Optionally injects a rare correlated burst that flips a contiguous block of 28 qubits
- Decodes with a majority-vote decoder
- Fits an exponential to the ε_d vs. d curve and reports Λ
- Runs distance sweeps in parallel across 64 workers

## Requirements

```bash
pip install qiskit qiskit-aer numpy matplotlib
```

## Usage

```bash
python fig3a.py
```

Output: `fig3a_cluster.pdf`

**Note:** This takes a long time to run. I ran it on the [OCF's HPC cluster](https://www.ocf.berkeley.edu/docs/services/hpc/) at Berkeley.

## Reference

Google Quantum AI - *Below-threshold error correction with a superconducting quantum processor*, Nature (2023).
