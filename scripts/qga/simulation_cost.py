"""Cost of simulating a Pattern 2 circuit: state vector versus the equivalent classical program (15 Nov 2026).

The circuit is the EAQGA crossover (Kashfi Haghighi et al. 2025, Algorithm 1, chain reading), built by
certify_product_state.eaqga_cadeia with random angles. For each number of qubits n it measures:
  - the time of the verifier (classifica), which reads the circuit gate by gate;
  - the time to draw 1000 samples with the classical program of the theorem of Section 5, one sample
    at a time (each fresh qubit keeps a 2-amplitude vector and becomes a bit when it enters a gate on
    several qubits); this sampler never builds the 2^n distribution;
  - the time of Qiskit's state-vector simulation (Statevector), while it fits in memory.
Before timing, the per-shot sampler is checked against the state vector for small n (total variation
of the empirical distribution, 200000 shots).

Usage: python scripts/qga/simulation_cost.py
Output: results/qga/simulation_cost-<date>.txt (table n | verifier s | sampler s | state vector s)
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "qcalc"))
from certify_product_state import FLIP1, DIAG1, PERM, _aplica_perm, classifica, eaqga_cadeia  # noqa: E402

RAIZ = AQUI.parents[1]
SEMENTE = 20261117
SHOTS = 1000
REPS = 3


def amostra(plano, n, rng):
    """One sample of the classical program. Requires an empty exact prefix (all qubits start in |0>)."""
    gs, prefixo = plano
    assert prefixo == 0, "per-shot sampler implemented for circuits without an exact prefix"
    vet = {q: np.array([1.0 + 0j, 0j]) for q in range(n)}
    bits = {}

    def sorteia(q):
        if q in vet:
            v = vet.pop(q)
            p1 = abs(v[1]) ** 2 / (abs(v[0]) ** 2 + abs(v[1]) ** 2)
            bits[q] = int(rng.random() < p1)

    for nm, qs, op in gs:
        if nm == "measure":
            continue
        if len(qs) == 1:
            q = qs[0]
            if q in vet:
                vet[q] = op.to_matrix() @ vet[q]
            elif nm in FLIP1 | DIAG1:
                if nm in FLIP1:
                    _aplica_perm(bits, nm, qs)
            else:  # rule R2: the classical bit becomes a fresh qubit U|b>
                e = np.zeros(2, complex)
                e[bits.pop(q)] = 1
                vet[q] = op.to_matrix() @ e
            continue
        for q in qs:
            sorteia(q)
        if nm in PERM:
            _aplica_perm(bits, nm, qs)
    for q in range(n):
        sorteia(q)
    return sum(b << q for q, b in bits.items())


def main():
    rng = np.random.default_rng(SEMENTE)
    L = [f"Simulation cost of a Pattern 2 circuit (EAQGA chain reading) | seed {SEMENTE} | {date.today()}",
         f"sampler: {SHOTS} samples, one at a time; verifier and sampler: median of {REPS} timings; "
         "state vector: qiskit.quantum_info.Statevector, one timing", ""]

    # check of the per-shot sampler against the state vector
    L.append("check: n | TV(empirical 200000 shots, state vector) | expected TV scale sqrt(2^n/shots)")
    for n in (4, 6, 8):
        qc = eaqga_cadeia(n, rng.uniform(0, 2 * np.pi, n), rng)
        ok, det, plano = classifica(qc)
        assert ok, det
        sv = Statevector(qc.remove_final_measurements(inplace=False)).probabilities()
        cont = np.bincount([amostra(plano, n, rng) for _ in range(200000)], minlength=2 ** n) / 200000
        L.append(f"  {n} | {0.5 * np.abs(cont - sv).sum():.4f} | {np.sqrt(2 ** n / 200000):.4f}")
    L.append("")

    L.append("n | verifier (s) | sampler, 1000 samples (s) | state vector (s)")
    for n in (4, 8, 12, 16, 18, 20, 22, 24, 26, 28, 40, 60, 100, 200, 400):
        qc = eaqga_cadeia(n, rng.uniform(0, 2 * np.pi, n), rng)
        tvs, tas = [], []
        for _ in range(REPS):  # median of REPS timings for the cheap methods
            t0 = time.perf_counter()
            ok, det, plano = classifica(qc)
            tvs.append(time.perf_counter() - t0)
            assert ok, det
            t0 = time.perf_counter()
            for _ in range(SHOTS):
                amostra(plano, n, rng)
            tas.append(time.perf_counter() - t0)
        tv, ta = float(np.median(tvs)), float(np.median(tas))
        if n <= 28:
            t0 = time.perf_counter()
            Statevector(qc.remove_final_measurements(inplace=False)).probabilities()
            ts = f"{time.perf_counter() - t0:.4g}"
        else:
            ts = "-"
        L.append(f"{n} | {tv:.4g} | {ta:.4g} | {ts}")
        print(L[-1], flush=True)
    L.append("")
    L.append("state vector memory (complex128): 2^n x 16 bytes; n = 28 -> 4 GiB; n = 100 -> about 2e31 bytes")
    (RAIZ / "results" / "qga" / f"simulation_cost-{date.today()}.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
