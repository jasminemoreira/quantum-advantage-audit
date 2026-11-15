"""Acampora group (15 Nov 2026): HQGA (Acampora & Vitiello 2021/2022), QMO (Acampora, Schiattarella &
Vitiello 2022, as described in the follow-up papers of 2023-2025) and Rosales et al. 2024 (CyS).

Circuits reconstructed from the texts and figures (no code from the authors). For each reading: classifica()
from certify_product_state.py, n in (3, 5, 8) with randomly drawn angles, and TV(sampler, statevector)
when the circuit is classified as classical sampling.

Usage: python scripts/qga/cases_acampora.py
Output: results/qga/cases_acampora-<date>.txt
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from certify_product_state import amostrador, classifica  # noqa: E402

RAIZ = AQUI.parents[1]
SEMENTE = 20261115
rng = np.random.default_rng(SEMENTE)
L: list[str] = []


def out(s=""):
    print(s)
    L.append(s)


# ---------------- HQGA ----------------
def hqga(n, k=3, elitismo="angle", mut_pos="free", mu=0.5, inicial=False):
    """HQGA, one iteration (Info. Sci. 2021, Sec. 4.1-4.5; Granular Computing 2022, Figs. 7-9).

    k chromosomes of n qubits; qubit (c, i) -> index c*n + i. Chromosome 0 is the best one.
    - best: H + Ry(sum of angles [+-rho]) on each qubit (pure/reinforcement elitism) or X where b*_i = 1
      (deterministic elitism, Eq. 14);
    - each of the k-1 groups of the best chromosome (Eq. 11) controls CNOTs onto the corresponding qubits
      of another chromosome (targets in |0>, no gate before; Eq. 13 and Figs. 7b-d);
    - free qubits: H + Ry(reconstructed angle); mutation Ry(theta_m) with prob. mu, AFTER the CNOT
      layer (barrier in the figures), only on free qubits (Def. 3).
    inicial=True: initialization circuit (H + Ry(+-delta) on all qubits, Figs. 7a/8a/9a).
    mut_pos (hypothetical readings, not described in the paper): "control" = extra Ry on a qubit of the
    best chromosome after the CNOT; "target" = extra Ry on a target qubit after the CNOT.
    """
    N = k * n
    qc = QuantumCircuit(N, N)
    idx = lambda c, i: c * n + i
    if inicial:
        for q in range(N):
            qc.h(q)
            qc.ry(rng.choice([-1, 1]) * rng.uniform(0, np.pi / 8), q)
        qc.measure(range(N), range(N))
        return qc
    outros = list(rng.permutation(np.arange(1, k)))
    grupos = np.array_split(np.arange(n), k - 1)
    alvo_de = {}
    for g, c in zip(grupos, outros):
        for i in g:
            alvo_de[i] = c
    # best chromosome
    for i in range(n):
        if elitismo == "deterministic":
            if rng.random() < 0.5:
                qc.x(idx(0, i))
        else:
            qc.h(idx(0, i))
            qc.ry(rng.uniform(-np.pi, np.pi), idx(0, i))
    # free qubits
    livres = [idx(c, i) for c in range(1, k) for i in range(n) if alvo_de[i] != c]
    for q in livres:
        qc.h(q)
        qc.ry(rng.uniform(-np.pi, np.pi), q)
    qc.barrier()
    for i in range(n):
        qc.cx(idx(0, i), idx(alvo_de[i], i))
    qc.barrier()
    for q in livres:
        if rng.random() < mu:
            qc.ry(rng.uniform(-np.pi, np.pi), q)
    if mut_pos == "control":
        qc.ry(rng.uniform(-np.pi, np.pi), idx(0, 0))
    elif mut_pos == "target":
        qc.ry(rng.uniform(-np.pi, np.pi), idx(alvo_de[0], 0))
    qc.measure(range(N), range(N))
    return qc


# ---------------- QMO ----------------
def qmo(n, pm=0.5):
    """QMO (CEC 2022), as described in CEC 2023 [11, Sec. II, Fig. 2], CEC 2023 [12, Fig. 2],
    RTSI 2024 [13, Fig. 1], MCII 2025 [16, Sec. II-C]: Ry(pi f_i) on |0>, Ry(r_i) with prob. pm,
    r_i in [0, pi]; measure."""
    qc = QuantumCircuit(n, n)
    f = rng.integers(0, 5, n) / 4  # frequencies of '1' in a pool of 4 parents
    for i in range(n):
        qc.ry(np.pi * f[i], i)
    for i in range(n):
        if rng.random() < pm:
            qc.ry(rng.uniform(0, np.pi), i)
    qc.measure(range(n), range(n))
    return qc


# ---------------- Rosales 2024 ----------------
def rosales(n, acumula=False):
    """Rosales et al. 2024, CyS 28(3), Sec. 3.1 and Fig. 1: H on each qubit, Ry(Delta theta_i) on each qubit
    (Eq. 12, Delta theta from Table 1 in {0, +-0.005, -0.01, 0.025, -0.05}); measure.
    acumula=True: reading in which the angle accumulates across generations (arbitrary angle)."""
    tab = np.array([0, 0, 0, -0.05, -0.01, 0.025, 0.005, 0.025])
    qc = QuantumCircuit(n, n)
    for i in range(n):
        qc.h(i)
        ang = rng.uniform(-np.pi, np.pi) if acumula else 2 * rng.choice(tab)  # Eq. 12 uses theta, Ry uses theta/2
        qc.ry(ang, i)
    qc.measure(range(n), range(n))
    return qc


def roda(nome, f, ns, reps=20, tv=True):
    linhas, oks, dets, tvs = [], set(), set(), []
    for n in ns:
        for _ in range(reps):
            qc = f(n)
            ok, det, plano = classifica(qc)
            oks.add(ok)
            dets.add(det.split(";")[0] if ok else det)
            if ok and tv and qc.num_qubits <= 16:
                sv = Statevector(qc.remove_final_measurements(inplace=False)).probabilities()
                tvs.append(0.5 * np.abs(amostrador(qc, plano) - sv).sum())
    s = (f"{nome} | n={ns} | classical sampling = {sorted(oks)} | "
         + (f"TV max {max(tvs):.1e} over {len(tvs)} circuits" if tvs else "no TV"))
    out(s)
    out("   " + "; ".join(sorted(dets))[:400])
    return s


def main():
    out(f"Acampora group | seed {SEMENTE} | {date.today()}")
    # HQGA: n = qubits per chromosome; k = 3 (3n in total). TV up to 15 qubits.
    roda("HQGA initialization (Figs. 7a/8a/9a)", lambda n: hqga(n, inicial=True), (1, 3, 5))
    roda("HQGA iteration, elitism by angle (Figs. 7b-d)", lambda n: hqga(n), (1, 3, 5))
    roda("HQGA iteration, deterministic elitism (Eq. 14)", lambda n: hqga(n, elitismo="deterministic"), (1, 3, 5))
    roda("HQGA k=4 and k=5", lambda n: hqga(n, k=4), (2, 3))
    roda("HQGA n=8 per chromosome (24 qubits, classification only)", lambda n: hqga(n), (8,), reps=3, tv=False)
    roda("HQGA hypothetical: Ry on the control after CNOT", lambda n: hqga(n, mut_pos="control"), (1, 3, 5))
    roda("HQGA hypothetical: Ry on the target after CNOT", lambda n: hqga(n, mut_pos="target"), (1, 3, 5))
    t0 = time.time()
    ok, det, _ = classifica(hqga(100, k=3))
    out(f"HQGA 300 qubits (k=3, n=100): {ok} ({det}) | {time.time() - t0:.2f}s")
    # QMO and Rosales: n = number of qubits
    roda("QMO (Ry crossover + Ry mutation)", qmo, (3, 5, 8))
    roda("Rosales 2024 (H + Ry(Delta theta), Table 1)", rosales, (3, 5, 8))
    roda("Rosales 2024 (H + accumulated Ry)", lambda n: rosales(n, acumula=True), (3, 5, 8))
    # Rosales: per-qubit marginal with Delta theta from Table 1 (Eq. 12 with theta as half-angle)
    for d in (-0.05, -0.01, 0, 0.005, 0.025):
        p1 = np.sin(np.pi / 4 + d) ** 2
        out(f"   Rosales: Delta theta = {d:+.3f} -> P(1) = {p1:.4f}")
    (RAIZ / "results" / "qga" / f"cases_acampora-{date.today()}.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
