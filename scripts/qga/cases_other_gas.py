"""Other genetic algorithms group (15 Nov 2026): applies classifica() to the circuits reconstructed from
17 (Sarracino et al. 2026, AEQGA), 19 (Abdolahi et al. 2025, SWAP test), 26 (Wang & Wang 2025,
QCHGEA-TOF) and 29 (Soloviev et al. 2021, QIEDA, modified W state).
Does not edit the package; only imports classifica/amostrador from certify_product_state.py.
Besides the classification, checks each reconstruction against the paper: angles of Fig. 2 and the
categorical distribution (29), a dense-vector computation of size N=2^n (17) and the SWAP-test
probabilities (19).

Usage: python scripts/qga/cases_other_gas.py
Output: results/qga/cases_other_gas-<date>.txt
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate, PhaseGate, StatePreparation
from qiskit.quantum_info import Statevector

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from certify_product_state import classifica, amostrador  # noqa: E402

RAIZ = AQUI.parents[1]
SEMENTE = 20261115
rng = np.random.default_rng(SEMENTE)
L = []


def tv(p, q):
    return 0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum()


def roda(nome, f, ns, reps=20):
    vered, tvs = set(), []
    for n in ns:
        for _ in range(reps):
            qc = f(n)
            ok, det, plano = classifica(qc)
            vered.add((ok, det if not ok else det.split(";")[0]))
            if ok:
                sv = Statevector(qc.remove_final_measurements(inplace=False)).probabilities()
                tvs.append(tv(amostrador(qc, plano), sv))
    L.append(f"{nome}: ok={sorted({v[0] for v in vered})} | "
             + (f"TV max {max(tvs):.1e} over {len(tvs)} circuits" if tvs else "no sampler"))
    L.append("   " + "; ".join(sorted({v[1] for v in vered}))[:400])


# ---------- 26: Wang & Wang 2025 -- H + rotation on each qubit, no two-qubit gates ----------
def wang(n, rot="ry", camadas=1):
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
        for _ in range(camadas):  # initial perturbation + accumulated evolution rotations
            th = rng.uniform(-np.pi, np.pi)
            if rot == "ry":
                qc.ry(th, q)
            elif rot == "rx":
                qc.rx(th, q)
            else:
                qc.u(*rng.uniform(0, 2 * np.pi, 3), q)
    qc.measure_all()
    return qc


# ---------- 29: Soloviev et al. 2021 -- modified W state (Alg. 3 and Fig. 2) ----------
def angulos_w(g, leitura):
    n = len(g)
    rots = [2 * np.arccos(np.sqrt(g[0]))]
    for i in range(1, n - 1):  # CRy on q1..q_{n-2}; Fig. 2 (n=5) shows 3 CRy
        acum = g[:i].sum() if leitura == "figure" else g[:i + 1].sum()  # text: sum up to i
        amp = np.sqrt(max(1 - acum, 1e-300))
        rots.append(2 * np.arccos(np.clip(np.sqrt(g[i]) / amp, -1, 1)))
    return rots


def soloviev(n, leitura="figure", g=None):
    if g is None:
        g = rng.dirichlet(np.ones(n))
    rots = angulos_w(g, leitura)
    qc = QuantumCircuit(n)
    qc.ry(rots[0], 0)
    for i in range(1, n - 1):
        qc.cry(rots[i], i - 1, i)
    for i in range(n - 1, 0, -1):
        qc.cx(i - 1, i)
    qc.x(0)
    qc.measure_all()
    return qc, g


def dist_w(qc):
    sv = Statevector(qc.remove_final_measurements(inplace=False)).probabilities()
    n = qc.num_qubits
    onehot = np.array([sv[1 << j] for j in range(n)])
    return sv, onehot


# ---------- 17: Sarracino et al. 2026 -- amplitude encoding + Rx(pi/2) + double CRy(pi/2) ----------
def sarracino(n, prep="stateprep", x=None):
    if x is None:
        x = rng.uniform(0, 1, 2 ** n)  # positive values of the population
    x = x / np.linalg.norm(x)
    qc = QuantumCircuit(n)
    if prep == "stateprep":
        qc.append(StatePreparation(x), range(n))
    else:
        qc.initialize(x, range(n))
    m = rng.integers(n)
    qc.rx(np.pi / 2, m)  # mutation (Eq. 15) on a randomly drawn qubit
    a, b = rng.choice(n, 2, replace=False)
    qc.cry(np.pi / 2, a, b)  # crossover (Eqs. 12-14)
    qc.cry(np.pi / 2, b, a)
    qc.measure_all()
    return qc, x, m, a, b


def sarracino_numpy(x, n, m, a, b):
    """Same computation on a dense vector of 2^n = size of the subpopulation (classical cost O(N) per gate)."""
    psi = x.astype(complex).reshape([2] * n)  # axis k = qubit n-1-k
    ax = lambda q: n - 1 - q
    rx = np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2)
    ry = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
    psi = np.moveaxis(np.tensordot(rx, psi, axes=([1], [ax(m)])), 0, ax(m))
    for c, t in ((a, b), (b, a)):
        sl = [slice(None)] * n
        sl[ax(c)] = 1
        sub = psi[tuple(sl)]
        axt = ax(t) - (1 if ax(t) > ax(c) else 0)
        sub = np.moveaxis(np.tensordot(ry, sub, axes=([1], [axt])), 0, axt)
        psi = psi.copy()
        psi[tuple(sl)] = sub
    return np.abs(psi.reshape(-1)) ** 2


# ---------- 19: Abdolahi et al. 2025 -- SWAP test with controlled initialization ----------
def abdolahi(mq, k, X=None, Y=None):
    """mq ancilla qubits (P = 2^mq individuals), registers F and G of k qubits (dense angle encoding,
    Eq. 5: |q_j> = cos(pi x_2j)|0> + e^{2 pi i x_2j+1} sin(pi x_2j)|1>), 1 control qubit."""
    P = 2 ** mq
    if X is None:
        X = rng.uniform(0, 1, (P, 2 * k))
    if Y is None:
        Y = rng.uniform(0, 1, 2 * k)
    anc = list(range(mq))
    F = list(range(mq, mq + k))
    G = list(range(mq + k, mq + 2 * k))
    c = mq + 2 * k
    qc = QuantumCircuit(mq + 2 * k + 1)
    for q in anc:
        qc.h(q)
    for i in range(P):  # initialization of |psi_i> controlled by the value i of the ancillas (Eq. 10)
        for j in range(k):
            qc.append(RYGate(2 * np.pi * X[i, 2 * j]).control(mq, ctrl_state=i), anc + [F[j]])
            qc.append(PhaseGate(2 * np.pi * X[i, 2 * j + 1]).control(mq, ctrl_state=i), anc + [F[j]])
    for j in range(k):  # register G: product state
        qc.ry(2 * np.pi * Y[2 * j], G[j])
        qc.p(2 * np.pi * Y[2 * j + 1], G[j])
    qc.h(c)
    for j in range(k):
        qc.cswap(c, F[j], G[j])
    qc.h(c)
    return qc, X, Y, anc, c


def ov_qubit(u, v):
    a = np.array([np.cos(np.pi * u[0]), np.exp(2j * np.pi * u[1]) * np.sin(np.pi * u[0])])
    b = np.array([np.cos(np.pi * v[0]), np.exp(2j * np.pi * v[1]) * np.sin(np.pi * v[0])])
    return abs(np.vdot(a, b)) ** 2


def main():
    L.append(f"Other genetic algorithms group | seed {SEMENTE} | {date.today()} | classifica() from certify_product_state.py")
    L.append("")
    L.append("== 26 Wang & Wang 2025 (H + rotation on each qubit; text: no entangling gates)")
    roda("26 reading Ry, 1 layer", lambda n: wang(n, "ry", 1), (3, 5, 8))
    roda("26 reading Rx, 1 layer", lambda n: wang(n, "rx", 1), (3, 5, 8))
    roda("26 reading general U, 3 layers (accumulated rotations)", lambda n: wang(n, "u", 3), (3, 5, 8))
    import time
    t0 = time.time()
    ok, det, _ = classifica(wang(200, "ry", 2))
    L.append(f"26 with 200 qubits: ok={ok} ({det}) | {time.time() - t0:.2f}s")

    L.append("")
    L.append("== 29 Soloviev et al. 2021 (modified W state)")
    for leit in ("figure", "text"):
        roda(f"29 reading {leit}", lambda n, le=leit: soloviev(n, le)[0], (3, 5, 8))
    # Fig. 2: angles with probabilities 1/5
    qc, g = soloviev(5, "figure", np.full(5, 0.2))
    L.append("29 Fig. 2 (n=5, gamma=1/5): angles = "
             + ", ".join(f"{t:.3f}" for t in angulos_w(np.full(5, 0.2), "figure"))
             + " (figure: 2.21, 2pi/3=2.094, 1.91, pi/2=1.571)")
    tvs, fora = [], []
    for n in (3, 5, 8):
        for _ in range(20):
            qc, g = soloviev(n, "figure")
            sv, oh = dist_w(qc)
            tvs.append(tv(oh, g))
            fora.append(1 - oh.sum())
    L.append(f"29 reading figure: TV(measured distribution, categorical gamma) max {max(tvs):.1e}; "
             f"prob. outside the one-hot states max {max(fora):.1e} (60 circuits)")
    tvs = []
    for n in (3, 5, 8):
        for _ in range(20):
            qc, g = soloviev(n, "text")
            sv, oh = dist_w(qc)
            tvs.append(tv(oh, g))
    L.append(f"29 reading text (sum up to i): TV(measured, gamma) min {min(tvs):.2f} max {max(tvs):.2f} "
             "-- reproduces neither the categorical distribution nor the angles of Fig. 2")

    L.append("")
    L.append("== 17 Sarracino et al. 2026 (amplitude encoding + Rx(pi/2) + CRy(pi/2) in both directions)")
    for prep in ("stateprep", "initialize"):
        roda(f"17 preparation {prep}", lambda n, p=prep: sarracino(n, p)[0], (2, 3, 5), reps=10)
    tvs = []
    for n in (2, 3, 5, 8):
        for _ in range(10):
            qc, x, m, a, b = sarracino(n)
            sv = Statevector(qc.remove_final_measurements(inplace=False)).probabilities()
            tvs.append(tv(sv, sarracino_numpy(x, n, m, a, b)))
    L.append(f"17: TV(statevector, computation on a dense vector of size N=2^n) max {max(tvs):.1e} (40 circuits)")

    L.append("")
    L.append("== 19 Abdolahi et al. 2025 (H on the ancillas, controlled initialization, SWAP test)")
    for mq, k in ((1, 1), (1, 2), (2, 2), (2, 3), (3, 2)):
        vered, erros, unif = set(), [], []
        for _ in range(5):
            qc, X, Y, anc, c = abdolahi(mq, k)
            qm = qc.copy()
            qm.measure_all()
            ok, det, _ = classifica(qm)
            vered.add((ok, det))
            sv = Statevector(qc)
            pr = sv.probabilities(anc + [c])  # index: anc (low bits) + c (high bit)
            P = 2 ** mq
            for i in range(P):
                ov = np.prod([ov_qubit(X[i, 2 * j:2 * j + 2], Y[2 * j:2 * j + 2]) for j in range(k)])
                erros.append(abs(pr[i + P] - (0.5 - 0.5 * ov) / P))
                unif.append(abs(pr[i] + pr[i + P] - 1 / P))
        L.append(f"19 m={mq}, k={k} ({qc.num_qubits} qubits): ok={sorted({v[0] for v in vered})} "
                 f"| {'; '.join(sorted({v[1] for v in vered}))[:200]}")
        L.append(f"   |P(i,c=1) - (1-prod_j|<a_ij|b_j>|^2)/(2P)| max {max(erros):.1e}; "
                 f"|P(i) - 1/P| max {max(unif):.1e}")
    out = "\n".join(L)
    print(out)
    (RAIZ / "results" / "qga" / f"cases_other_gas-{date.today()}.txt").write_text(out + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
