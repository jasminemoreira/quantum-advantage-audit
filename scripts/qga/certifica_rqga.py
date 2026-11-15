"""Certificacao exata da uniformidade do RQGA (13/11/2026), piloto do controle de equivalencia.

Os mesmos circuitos de verifica_rqga.py (codigo dos autores: Ardelean e Udrescu 2022,
externos/rqga a018d70; Viana e de Paula Neto 2025, externos/maxcut-qga 23533c4), agora
executados no motor exato esparso em Z[zeta_16] (scripts/qcalc/esparso.py), sem ponto
flutuante. Para cada caso e numero de iteracoes de Grover:
  (A) certifica que a marginal do registro medido (individuos, N qubits) e EXATAMENTE
      uniforme: 2^N * soma_{x: ind(x)=y} u_x conj(u_x) == 2^k * (1,0,...,0) para todo y;
  (B) contraste: a marginal do registro de aptidao NAO e uniforme e muda com as iteracoes
      (valores exatos, convertidos a decimal so para relato);
  (C) invariante de Parseval: soma dos quadrados dos inteiros == 2^k.
O Aer (verifica_rqga.py, 11/11/2026) dava |p - 2^-N| <= 6e-17, que nao distingue exatamente
uniforme de quase uniforme; aqui a igualdade e decidida.

Uso (WSL): externos/mqtbench/.venv/bin/python scripts/qga/certifica_rqga.py
Saida: resultados/qga/certifica_rqga-<data>.txt
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
    """num em Z[zeta] real; P = num/2^k. Parte real = a0 + soma a_j cos(j pi/8)."""
    import math
    return (num[0] + sum(num[j] * math.cos(j * math.pi / 8) for j in range(1, 8))) / 2 ** k


def certifica(rotulo, qc, ind, fit, linhas):
    idx = {q: i for i, q in enumerate(qc.qubits)}
    qi = [idx[q] for q in ind]
    qf = [idx[q] for q in fit]
    portas = converte(qc, rota_exata=True)
    if portas is None:
        linhas.append(f"{rotulo} | fora do anel")
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
    lin = (f"{rotulo} | qubits={qc.num_qubits} | portas={len(portas)} | k={m.k} | suporte max={m.max_suporte} | "
           f"bits={m.max_bits()} | ind (N={N}) exatamente uniforme={uniforme} | Parseval={parseval} | "
           f"aptidao: {len(mf)} valores, maior P={pf[0]:.6f} | {time.time() - t0:.1f}s")
    linhas.append(lin)
    print(lin, flush=True)
    return uniforme and parseval, pf


def main():
    rng = np.random.default_rng(V.SEMENTE)  # mesmos grafos de verifica_rqga.py
    linhas = [f"RQGA e MaxCut-QGA no motor exato esparso | semente {V.SEMENTE} | {date.today()}"]
    casos = []
    for n in (3, 4):
        g = nx.erdos_renyi_graph(n, 0.6, seed=int(rng.integers(2**31)))
        adj = nx.to_numpy_array(g, dtype=int).tolist()
        ns = V.carrega_rqga(n, adj)
        casos.append((f"RQGA-coloracao n={n} |E|={g.number_of_edges()}", lambda it, ns=ns: V.circuito_rqga(ns, it)))
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
        linhas.append(f"{rotulo} | marginal da aptidao muda de iter=1 para iter=4: {muda}")
        print(linhas[-1], flush=True)
    linhas.append(f"todos certificados: {tudo}")
    print(linhas[-1])
    (RAIZ / "resultados" / "qga" / f"certifica_rqga-{date.today()}.txt").write_text("\n".join(linhas) + "\n")
    return 0 if tudo else 1


if __name__ == "__main__":
    sys.exit(main())
