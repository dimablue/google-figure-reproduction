#!/usr/bin/env python
# Reproduction of Google Fig‑3a (repetition‑code scaling)
# with an optional rare 28‑qubit burst channel.

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import pauli_error

# ---------------------------------------------------------------------------
# 1. Build a distance‑d repetition‑code memory circuit (1000 cycles)
# ---------------------------------------------------------------------------
def build_repetition_memory(d: int, n_cycles: int = 1000) -> QuantumCircuit:
    qc = QuantumCircuit(d, d)
    for _ in range(n_cycles):
        qc.barrier()
        qc.id(range(d))      
    qc.measure(range(d), range(d))
    return qc

# ---------------------------------------------------------------------------
# 2. Pauli‑X noise + rare correlated burst
# ---------------------------------------------------------------------------
def burst_channel(d: int, block: int = 28, p_burst: float = 0.0):
    """Return a Pauli channel that flips one contiguous 'block' of qubits."""
    if p_burst == 0.0:
        return None
    block = min(block, d)
    ops = []
    frac = p_burst / (d - block + 1)
    for start in range(d - block + 1):
        mask = ["I"] * d
        for q in range(start, start + block):
            mask[q] = "X"
        ops.append(("".join(mask), frac))
    ops.append(("I" * d, 1.0 - p_burst))   # probabilities sum to 1
    return pauli_error(ops)

def make_noise_model(p_flip: float, p_burst: float, d: int) -> NoiseModel:
    nm = NoiseModel()
    x_err = pauli_error([("X", p_flip), ("I", 1.0 - p_flip)])
    nm.add_all_qubit_quantum_error(x_err, ["id"])
    if p_burst > 0.0:
        nm.add_all_qubit_quantum_error(burst_channel(d, 28, p_burst), ["barrier"])
    return nm

# ---------------------------------------------------------------------------
# 3. Logical error per cycle with a majority‑vote decoder
# ---------------------------------------------------------------------------
def logical_eps(d: int, p_flip: float, p_burst: float,
                n_cycles: int, shots: int) -> float:
    qc  = build_repetition_memory(d, n_cycles)
    sim = AerSimulator(method="stabilizer",
                       noise_model=make_noise_model(p_flip, p_burst, d),
                       max_parallel_threads=0,
                       max_parallel_shots=0)
    tqc = transpile(qc, sim, optimization_level=0)
    counts = sim.run(tqc, shots=shots).result().get_counts()

    fails = sum(c for bits, c in counts.items() if bits.count("1") > d // 2)
    # Jeffreys prior: soft floor ≈ 1 / (4·shots²)
    p_L = (fails + 0.5) / (shots + 1)
    eps = 0.5 * (1.0 - (1.0 - 2.0 * p_L) ** (1.0 / n_cycles))
    return eps

# ---------------------------------------------------------------------------
# 4. Parallel sweep
# ---------------------------------------------------------------------------
def worker(d, p_flip, p_burst, n_cycles, shots):
    eps = logical_eps(d, p_flip, p_burst, n_cycles, shots)
    print(f"Running d = {d:<2d} …  ε = {eps:.2e}")
    return eps

def collect_eps(dists, p_flip, p_burst, *, n_cycles, shots, pool_size):
    args = [(d, p_flip, p_burst, n_cycles, shots) for d in dists]
    with multiprocessing.Pool(pool_size) as pool:
        return np.array(pool.starmap(worker, args))

# ---------------------------------------------------------------------------
# 5. Plot & fit
# ---------------------------------------------------------------------------
def plot_fig3a(dists, eps, label=None, *, outfile):
    fig, ax = plt.subplots(figsize=(3.2, 5.0))
    if label is not None:
        ax.semilogy(dists, eps, "ko", label=label)
        ax.legend()
    else:
        ax.semilogy(dists, eps, "ko")
    ax.set_xlabel("Code distance  $d$")
    ax.set_ylabel(r"Logical error per cycle  $\epsilon_d$")
    ax.set_ylim(0.000000000001, 0.01)   # 1×10^-12 … 1×10^-2
    ax.set_xticks(dists[::2])

    mask = eps > 0.0
    m, b = np.polyfit(np.array(dists)[mask], np.log(eps[mask]), 1)
    ax.plot(dists, np.exp(m * np.array(dists) + b), color="lightgrey", lw=1.0)

    ax.legend()
    plt.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"Slope m = {m:+.4f}  →  Λ ≈ {np.exp(-2.0 * m):.2f}")

# ---------------------------------------------------------------------------
# 6. Cluster‑scale run
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    DISTANCES = list(range(3, 30, 2))          # 3, 5, …, 29
    P_FLIP    = 0.002                       # physical single‑qubit X error
    P_BURST   = 0.0000000000000005                    # five in a billion burst per cycle
    N_CYCLES  = 1000
    SHOTS     = 1000000                      # million shots / distance
    POOL_SIZE = 64                            # to run on HPC

    eps = collect_eps(DISTANCES, P_FLIP, P_BURST,
                      n_cycles=N_CYCLES, shots=SHOTS, pool_size=POOL_SIZE)

    plot_fig3a(DISTANCES, eps,
               outfile="fig3a_cluster.pdf")
