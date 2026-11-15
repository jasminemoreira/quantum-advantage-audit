"""Exact certification of the uniformity of the RQGA (13 Nov 2026), pilot of the equivalence check.

The same circuits as verifica_rqga.py (authors' code: Ardelean and Udrescu 2022,
externos/rqga a018d70; Viana and de Paula Neto 2025, externos/maxcut-qga 23533c4), now
run on the exact sparse engine over Z[zeta_16] (scripts/qcalc/esparso.py), with no floating
point. For each case and number of Grover iterations:
  (A) certifies that the marginal of the measured register (individuals, N qubits) is EXACTLY
      uniform: 2^N * sum_{x: ind(x)=y} u_x conj(u_x) == 2^k * (1,0,...,0) for every y;
  (B) contrast: the marginal of the fitness register is NOT uniform and changes with the iterations
      (exact values, converted to decimal for reporting only);
  (C) Parseval invariant: sum of the squares of the integers == 2^k.
Aer (verifica_rqga.py, 11 Nov 2026) gave |p - 2^-N| <= 6e-17, which does not distinguish exactly
uniform from nearly uniform; here the equality is decided.

Usage (WSL): externos/mqtbench/.venv/bin/python scripts/qga/certifica_rqga.py
Output: resultados/qga/certifica_rqga-<data>.txt
"""
from __future__ import annotations

import sys
import time
from datetime import date
from fractions import Fraction
from pathlib import Path

import networkx as nx
import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "qcalc"))
import verifica_rqga as V  # noqa: E402
from esparso import MotorEsparso  # noqa: E402
from converte import converte  # noqa: E402

RAIZ = AQUI.parents[1]


def prob_decimal(num, k):
    """num in real Z[zeta]; P = num/2^k. Real part = a0 + sum a_j cos(j pi/8)."""
    import math
    return (num[0] + sum(num[j] * math.cos(j * math.pi / 8) for j in range(1, 8))) / 2 ** k


def certifica(rotulo, qc, ind, fit, linhas):
    idx = {q: i for i, q in enumerate(qc.qubits)}
    qi = [idx[q] for q in ind]
    qf = [idx[q] for q in fit]
    portas = converte(qc, rota_exata=True)
    if portas is None:
        linhas.append(f"{rotulo} | outside the ring")
        return False
    t0 = time.time()
    m = MotorEsparso(qc.num_qubits)
    for p in portas:
        m.aplica(p)
    N = len(qi)
    marg = m.marginal(qi)
    alvo = (2 ** m.k,) + (0,) * 7
    uniforme = len(marg) == 2 ** N and all(tuple(2 ** N * a for a in v) == alvo for v in marg.values())
    parseval = m.soma_quadrados() == 2 ** m.k
    mf = m.marginal(qf)
    pf = sorted((prob_decimal(v, m.k) for v in mf.values()), reverse=True)
    lin = (f"{rotulo} | qubits={qc.num_qubits} | gates={len(portas)} | k={m.k} | max support={m.max_suporte} | "
           f"bits={m.max_bits()} | ind (N={N}) exactly uniform={uniforme} | Parseval={parseval} | "
           f"fitness: {len(mf)} values, largest P={pf[0]:.6f} | {time.time() - t0:.1f}s")
    linhas.append(lin)
    print(lin, flush=True)
    return uniforme and parseval, pf


def main():
    rng = np.random.default_rng(V.SEMENTE)  # same graphs as verifica_rqga.py
    linhas = [f"RQGA and MaxCut-QGA on the exact sparse engine | seed {V.SEMENTE} | {date.today()}"]
    casos = []
    for n in (3, 4):
        g = nx.erdos_renyi_graph(n, 0.6, seed=int(rng.integers(2**31)))
        adj = nx.to_numpy_array(g, dtype=int).tolist()
        ns = V.carrega_rqga(n, adj)
        casos.append((f"RQGA-coloring n={n} |E|={g.number_of_edges()}", lambda it, ns=ns: V.circuito_rqga(ns, it)))
    Frag, to_binary = V.carrega_maxcut()
    for n in (4, 6, 8):
        g = nx.erdos_renyi_graph(n, 0.5, seed=int(rng.integers(2**31)))
        q = Frag(g, shots=1000)
        casos.append((f"MaxCut-QGA n={n} |E|={g.number_of_edges()}", lambda it, q=q: V.circuito_maxcut(q, to_binary, it)))
    tudo = True
    for rotulo, construtor in casos:
        dist_fit = []
        for it in range(1, 5):
            qc, ind, fit = construtor(it)
            ok, pf = certifica(f"{rotulo} iter={it}", qc, ind, fit, linhas)
            tudo &= ok
            dist_fit.append(pf)
        muda = any(abs(a - b) > 1e-12 for a, b in zip(dist_fit[0], dist_fit[-1])) or len(dist_fit[0]) != len(dist_fit[-1])
        linhas.append(f"{rotulo} | fitness marginal changes from iter=1 to iter=4: {muda}")
        print(linhas[-1], flush=True)
    linhas.append(f"all certified: {tudo}")
    print(linhas[-1])
    (RAIZ / "resultados" / "qga" / f"certifica_rqga-{date.today()}.txt").write_text("\n".join(linhas) + "\n")
    return 0 if tudo else 1


if __name__ == "__main__":
    sys.exit(main())
