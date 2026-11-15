"""Ant colony / Grover group (15 Nov 2026): works 05/07b (Wu/Qiu et al., QACO + K-means), 07a (Ghosh et al.,
MNDAS QACO), 25 (Martinez et al., GAQS, APMC 2025), 42 (Cadavid et al., QE-MTS, QMI 2026).

Circuits reconstructed from the figures and the pseudocode of the papers (none of them publishes code).
For each case: classifica() from certify_product_state.py and, when the circuit is classified as classical
sampling, TV(sampler, statevector); for 07a, the pheromone box under two readings and the amplitude
amplification (QAA) under four readings; for 25, the Grover probabilities against the closed form.

Usage: python scripts/qga/cases_ant_colony_grover.py
Output: results/qga/cases_ant_colony_grover-<date>.txt
"""
from __future__ import annotations

import sys
from datetime import date
from math import asin, sin, sqrt
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit.quantum_info import Statevector, partial_trace

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


def marginal(probs, qs, n):
    """Marginal distribution (dict bitstring->p) of the qubits qs (qs[0] = most significant bit)."""
    d = {}
    for x, p in enumerate(probs):
        if p < 1e-15:
            continue
        k = "".join(str((x >> q) & 1) for q in qs)
        d[k] = d.get(k, 0) + p
    return d


def certifica(nome, f, esperado=True, ns=(3, 5, 8), reps=20):
    tvs, det = [], set()
    oks = set()
    for n in ns:
        for _ in range(reps):
            qc = f(n)
            ok, d, plano = classifica(qc)
            oks.add(ok)
            det.add(d if not ok else d.split(";")[0])
            if ok:
                sv = Statevector(qc.remove_final_measurements(inplace=False)).probabilities()
                tvs.append(0.5 * np.abs(amostrador(qc, plano) - sv).sum())
    out(f"{nome}: classical sampling = {sorted(oks)} (expected {esperado}) | "
        + (f"TV max {max(tvs):.1e} over {len(tvs)} circuits" if tvs else "no sampler"))
    out("   " + "; ".join(sorted(det))[:400])
    return oks, tvs


# ---------------------------------------------------------------- 05 / 07b (Wu 2025; Qiu 2024)
def wu_fig1(n):
    """05, Fig. 1: Ry on q0 and on n path qubits; three CSWAPs controlled by q0
    (q1_0,q1_1), (q1_0,q1_n), (q1_1,q1_n); measure the path qubits."""
    qc = QuantumCircuit(n + 1, n)
    for q in range(n + 1):
        qc.ry(rng.uniform(0, np.pi), q)
    for a, b in ((1, 2), (1, n), (2, n)):
        qc.cswap(0, a, b)
    for q in range(1, n + 1):
        qc.measure(q, q - 1)
    return qc


def qiu_fig2(n):
    """07b, Fig. 2: Ry on all qubits; CSWAP(q0; i, j) on pairs of path qubits, in order;
    each path qubit is measured after its last use (mid-circuit measurement, never followed by a gate)."""
    qc = QuantumCircuit(n + 1, n)
    for q in range(n + 1):
        qc.ry(rng.uniform(0, np.pi), q)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            qc.cswap(0, i, j)
        qc.measure(i, i - 1)
    return qc


def qiu_fig2_aleatoria(n):
    """Variant: randomly drawn pairs (the exact order in the figure does not change the rule)."""
    qc = QuantumCircuit(n + 1, n)
    for q in range(n + 1):
        qc.ry(rng.uniform(0, 2 * np.pi), q)
    for _ in range(3 * n):
        a, b = rng.choice(np.arange(1, n + 1), 2, replace=False)
        qc.cswap(0, int(a), int(b))
    for q in range(1, n + 1):
        qc.measure(q, q - 1)
    return qc


# ---------------------------------------------------------------- 07a (Ghosh et al. 2021)
W8 = [21, 18, 16, 11, 5, 2, 11, 14]          # Table 1 / Fig. 8
D = 4                                         # 4-qubit pheromone box


def dep(ph):
    """pheromone_Deposition (Alg. 4) on a basis state; ph = [ph0..ph3]."""
    ph = list(ph)
    for m in range(D - 2, -1, -1):
        if ph[m]:
            ph[m + 1] ^= 1
    ph[0] ^= 1
    return ph


def eva(ph):
    """pheromone_Evaporation (Alg. 5): inverse sequence of the deposition."""
    ph = list(ph)
    ph[0] ^= 1
    for m in range(0, D - 1):
        if ph[m]:
            ph[m + 1] ^= 1
    return ph


def evolui(K, W, leitura):
    """Content of the pheromone box of each path after K iterations (classical function of p).
    reading 'fig10': the rule that reproduces Fig. 10 (an iteration with no selected path does nothing;
      deposit on the selected paths that are not full; evaporate on the unselected paths that are neither
      empty nor full);
    reading 'alg': Algorithms 2-5 literally (a2 = NOT(selected) always; Alg. 3 with the C^dNOT gates
      in the order written)."""
    n = len(W)
    ph = {p: [0] * D for p in range(n)}
    cheio = [1] * D
    for t in range(1, K + 1):
        sel = {p: t % W[p] == 0 for p in range(n)}
        if leitura == "fig10":
            if not any(sel.values()):
                continue
            for p in range(n):
                if sel[p] and ph[p] != cheio:
                    ph[p] = dep(ph[p])
                elif not sel[p] and ph[p] != cheio and any(ph[p]):
                    ph[p] = eva(ph[p])
        else:
            for p in range(n):
                a1, a2 = int(sel[p]), int(not sel[p])
                if ph[p] == cheio:
                    a1 ^= 1
                if a1:
                    ph[p] = dep(ph[p])
                if ph[p] == cheio:
                    a2 ^= 1
                if not any(ph[p]):
                    a2 ^= 1
                if a2:
                    ph[p] = eva(ph[p])
    return ph


def ghosh_estado(ph, x):
    """State before the QAA: sum_p |p>|ph(p)>|at(p)> / sqrt(2^x). Qubits: P0..P_{x-1} = 0..x-1
    (P0 most significant, as in Fig. 9), ph0..ph3 = x..x+3, a_target = x+4."""
    n = 2 ** x
    nq = x + D + 1
    v = np.zeros(2 ** nq, complex)
    for p in range(n):
        bits = [(p >> (x - 1 - i)) & 1 for i in range(x)] + ph[p] + [int(all(ph[p]))]
        v[sum(b << q for q, b in enumerate(bits))] = 1 / sqrt(n)
    return Statevector(v), nq


def difusor(qc, qs):
    qc.h(qs); qc.x(qs)
    qc.h(qs[-1]); qc.append(MCXGate(len(qs) - 1), qs); qc.h(qs[-1])
    qc.x(qs); qc.h(qs)


def ghosh_qaa(ph, x, leitura, k=1, fase="cz"):
    sv, nq = ghosh_estado(ph, x)
    P = list(range(x))
    at = x + D
    qc = QuantumCircuit(nq)
    for _ in range(k):
        if leitura == "fig4":            # cRz/CPHASE(a_target -> P_{x-1}); QAA on the path qubits
            if fase == "cz":
                qc.cz(at, P[-1])
            else:
                qc.crz(np.pi, at, P[-1])
            difusor(qc, P)
        elif leitura == "text":          # "inversion about the mean on the target qubit"
            qc.cz(at, P[-1])
            qc.h(at); qc.x(at); qc.z(at); qc.x(at); qc.h(at)
        elif leitura in ("repair", "phase_P"):
            # phase_P: phase oracle on P itself, with the pheromone box still entangled;
            # repair: same, after uncomputing the box (P returns to the coherent uniform superposition)
            marc = [p for p in range(2 ** x) if all(ph[p])]
            for m in marc:
                bits = [(m >> (x - 1 - i)) & 1 for i in range(x)]
                for i, b in enumerate(bits):
                    if not b:
                        qc.x(P[i])
                qc.h(P[-1]); qc.append(MCXGate(x - 1), P); qc.h(P[-1])
                for i, b in enumerate(bits):
                    if not b:
                        qc.x(P[i])
            difusor(qc, P)
    if leitura == "repair":
        v = np.zeros(2 ** nq, complex)
        v[: 2 ** x] = 1 / sqrt(2 ** x)
        sv = Statevector(v)
    pr = sv.evolve(qc).probabilities()
    return marginal(pr, P, nq)


def ghosh_circuito(K, W, x):
    """Literal circuit (Algs. 2-6, Fig. 4) for classifica, with fresh ancillas at each iteration in place
    of the RESET (reinitialization Lemma). Qubits: P (x), ph (4), a_target, then 2 ancillas per iteration."""
    nq = x + D + 1 + 2 * K
    qc = QuantumCircuit(nq, x)
    P = list(range(x)); PH = list(range(x, x + D)); AT = x + D
    qc.h(P)
    for t in range(1, K + 1):
        a1, a2 = x + D + 1 + 2 * (t - 1), x + D + 2 + 2 * (t - 1)
        for i in range(len(W)):
            if t % W[i] == 0:
                zeros = [P[j] for j in range(x) if not (i >> (x - 1 - j)) & 1]
                if zeros:
                    qc.x(zeros)
                qc.mcx(P, a1); qc.mcx(P, a2)
                if zeros:
                    qc.x(zeros)
        qc.x(a2)
        qc.mcx(PH, a1)
        for m in range(D - 2, -1, -1):
            qc.ccx(a1, PH[m], PH[m + 1])
        qc.cx(a1, PH[0])
        qc.mcx(PH, a2)
        qc.x(PH); qc.mcx(PH, a2); qc.x(PH)
        qc.cx(a2, PH[0])
        for m in range(0, D - 1):
            qc.ccx(a2, PH[m], PH[m + 1])
    qc.mcx(PH, AT)
    qc.cz(AT, P[-1])
    difusor(qc, P)
    qc.measure(P, list(range(x)))
    return qc


# ---------------------------------------------------------------- 25 (Martinez et al. 2025)
def martinez(n):
    """Fig. 1: H on the n data qubits; X, H on the ancilla; MCX(data -> ancilla) (marks |1..1>);
    diffuser on the data; measure the data."""
    qc = QuantumCircuit(n + 1, n)
    qc.h(range(n)); qc.x(n); qc.h(n)
    qc.mcx(list(range(n)), n)
    difusor(qc, list(range(n)))
    qc.measure(range(n), range(n))
    return qc


# ---------------------------------------------------------------- 42 (Cadavid et al. 2026)
def ryz(qc, th, i, j):
    """exp(-i th Y_i Z_j) via change of basis on i and RZZ (Fig. 3 uses 2 RZZ + 4 rotations)."""
    qc.rx(np.pi / 2, i); qc.rzz(2 * th, i, j); qc.rx(-np.pi / 2, i)


def dcqo(n):
    """One DCQO step in the impulsive regime, two-body part only: |+>^n, R_YZ R_ZY on the pairs (i, i+k)."""
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            th = rng.uniform(0, np.pi)
            ryz(qc, th, i, j); ryz(qc, th, j, i)
    qc.measure(range(n), range(n))
    return qc


def main():
    out(f"Ant colony / Grover group | seed {SEMENTE} | {date.today()}")
    out("\n== 05 (Wu et al. 2025, Fig. 1) and 07b (Qiu et al. arXiv 2403.00367, Fig. 2) ==")
    certifica("05 Fig. 1 (3 CSWAP)", wu_fig1)
    certifica("07b Fig. 2 (CSWAP on all pairs, measurement after last use)", qiu_fig2)
    certifica("07b variant with randomly drawn pairs", qiu_fig2_aleatoria)
    qc = qiu_fig2(8)
    ok, d, _ = classifica(qc)
    out(f"07b with 9 qubits (size in the paper): {ok} ({d})")

    out("\n== 07a (Ghosh et al. 2021, MNDAS) ==")
    for leitura in ("fig10", "alg"):
        for K in (16, 20, 21, 200):
            ph = evolui(K, W8, leitura)
            s = " ".join(f"{p}:{''.join(map(str, ph[p]))}" for p in range(8))
            out(f"pheromone box, reading {leitura}, K={K}: {s}")
    out("(Fig. 10 of the paper, K=21: path 0 = 1000, path 5 = 1111, all others 0000)")
    for leitura in ("fig10", "alg"):
        for K in (21, 200):
            ph = evolui(K, W8, leitura)
            classes = {}
            for p in range(8):
                classes.setdefault("".join(map(str, ph[p])), []).append(p)
            out(f"\n reading {leitura}, K={K}: classes of paths with the same pheromone: {sorted(classes.values())}")
            for lq in ("fig4", "text", "phase_P", "repair"):
                for k in (1, 2, 3):
                    for fase in (("cz", "crz") if lq == "fig4" and k == 1 else ("cz",)):
                        m = ghosh_qaa(ph, 3, lq, k, fase)
                        p5 = m.get("101", 0.0)
                        resto = max(v for kk, v in m.items() if kk != "101")
                        out(f"   QAA {lq:9s} {fase:3s} k={k}: P(path 5) = {p5:.4f}; largest other = {resto:.4f}")
                        if lq == "fig4" and k == 1 and fase == "cz":
                            out("      full marginal: " + " ".join(
                                f"{int(kk, 2)}:{m.get(kk, 0):.4f}" for kk in sorted(m)))
    # no oracle: the diffuser alone in the fig4 reading
    ph = evolui(200, W8, "fig10")
    sv, nq = ghosh_estado(ph, 3)
    qc = QuantumCircuit(nq); difusor(qc, [0, 1, 2])
    m = marginal(sv.evolve(qc).probabilities(), [0, 1, 2], nq)
    out(f" control: diffuser alone on P, without the phase (K=200, fig10): P(5) = {m.get('101', 0):.4f}")
    out(" value reported in the paper (Fig. 12, ibmq_16_melbourne, K=200): P(5) = 0.501")
    # Biham-Kenigsberg Lemma / Lemma 2 of the paper if register P were a mixture? Bound of the lemma with |M|=1, M_B=8
    out(" bound of the Lemma (if P were a mixture of basis states): P0 + (1-P0)/(8-1) with P0=1/8 -> "
        f"{1/8 + (7/8)/7:.4f}")
    # purity of register P before the QAA
    sv, nq = ghosh_estado(ph, 3)
    rho = partial_trace(sv, list(range(3, nq)))
    out(f" purity of the reduced state of P before the QAA (K=200, fig10): {np.real(np.trace(rho.data @ rho.data)):.4f}")
    ok, d, _ = classifica(ghosh_circuito(4, W8, 3))
    out(f" classifica on the literal circuit (K=4, fresh ancillas in place of the RESET): {ok} ({d})")

    out("\n== 25 (Martinez et al. 2025, Fig. 1) ==")
    certifica("25 Fig. 1 (Grover, 1 marked, 1 iteration)", martinez, esperado=False)
    for n in (3, 5, 8):
        qc = martinez(n)
        sv = Statevector(qc.remove_final_measurements(inplace=False))
        pr = sv.probabilities()
        m = marginal(pr, list(range(n)), n + 1)
        rho = partial_trace(sv, list(range(n)))     # ancilla
        rhod = partial_trace(sv, [n])               # data
        out(f" n={n}: P(1..1) = {m.get('1' * n, 0):.4f}; sin^2(3 asin(2^(-n/2))) = {sin(3 * asin(2 ** (-n / 2))) ** 2:.4f}; "
            f"purity of the data = {np.real(np.trace(rhod.data @ rhod.data)):.6f}; "
            f"ancilla in |->: {abs(rho.data[0, 1] + 0.5) < 1e-12}")

    out("\n== 42 (Cadavid et al. 2026, DCQO, one step, two-body terms only) ==")
    certifica("42 DCQO (R_YZ R_ZY after |+>)", dcqo, esperado=False, ns=(3, 5), reps=5)

    (RAIZ / "results" / "qga" / f"cases_ant_colony_grover-{date.today()}.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
