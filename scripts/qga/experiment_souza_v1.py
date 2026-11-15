"""Experiments with the v1 operators (souza_v1.py), 13 Nov 2026.

Part A (redoes center_bias.py): |P| = 50, 30 generations, 100 repetitions; quantum evaluator,
classical stochastic analog without interference (H -> coin, Y -> X, phases removed) and classical
evaluator (G_cl); centered domain [-5.12, 5.12] (optimum at the center) and shifted domain
[-2.12, 8.12] (center at x = 3, optimum at x = 0).

Part B (entanglement between individuals, section 4.3 of their article): |P| = 10 as in their figure,
30 generations, 200 repetitions; quantum without entanglement, with "max" and with "bell" (see
souza_v1.py), and the classical equivalent of "bell" (analog with the initial state drawn between
b and ~b at each shot).
Prediction: "max" makes the output of every individual uniform, so x* falls at the center of the
domain for any circuit; on the centered domain this is the optimum, on the shifted one it is f(3, 3) = 18.

Reports the best evaluated f per generation (with shot noise, as in the article) and, for the
quantum evaluator, the exact f(E[x]) of the best final individual.

Usage: python scripts/qga/experiment_souza_v1.py
Output: results/qga/experiment_souza_v1-<date>.txt
"""
from __future__ import annotations

import sys
import time
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import souza_v1 as S  # noqa: E402

RAIZ = AQUI.parents[1]
SEMENTE = 20261115
DOMINIOS = ((-5.12, 5.12), (-2.12, 8.12))
GER = 30
GENS = (0, 1, 2, 5, 10, 20, 30)


def tarefa(args):
    parte, modo, emar, dom, pop_n, sem = args
    curva, melhor, f = S.ga(modo, dom, sem, pop_n=pop_n, ger=GER, emar=emar)
    fx = S.aptidao_exata(melhor, dom, emar) if modo == "quantico" else float("nan")
    return parte, modo, emar, dom, curva, fx


def tabela(res, chaves, rotulo):
    linhas = [rotulo, "domain | evaluator | " + " | ".join(f"g{g}" for g in GENS) + " | f(E[x]) of the best final (median)"]
    for dom in DOMINIOS:
        for modo, emar in chaves:
            rs = [r for r in res if r[1] == modo and r[2] == emar and r[3] == dom]
            c = np.array([r[4] for r in rs])
            med, ep = c.mean(axis=0), c.std(axis=0, ddof=1) / np.sqrt(len(c))
            fx = np.array([r[5] for r in rs])
            fxs = f"{np.median(fx):.3f}" if modo == "quantico" else "-"
            nome = f"{modo}" + (f"+{emar}" if emar else "")
            linhas.append(f"{dom} | {nome:14s} | " + " | ".join(f"{med[g]:.3f} ({ep[g]:.3f})" for g in GENS) + f" | {fxs}")
    return linhas


def main():
    t0 = time.time()
    A = [("A", modo, None, dom, 50, SEMENTE + 10000 * k + 1000 * j + r)
         for k, dom in enumerate(DOMINIOS) for j, modo in enumerate(("quantico", "gemeo", "classico")) for r in range(100)]
    chB = [("quantico", None), ("quantico", "max"), ("quantico", "bell"), ("gemeo", "bell")]
    B = [("B", modo, emar, dom, 10, SEMENTE + 500000 + 10000 * k + 1000 * j + r)
         for k, dom in enumerate(DOMINIOS) for j, (modo, emar) in enumerate(chB) for r in range(200)]
    with Pool(16) as pool:
        res = pool.map(tarefa, A + B, chunksize=2)
    linhas = [f"Souza/Dardenne/Portugal, v1 operators (souza_v1.py) | seed {SEMENTE} | {date.today()}",
              f"m=2 n=8 shots=1024 d=5 G={GER}; best evaluated f per generation, mean (standard error)", ""]
    linhas += tabela([r for r in res if r[0] == "A"], [("quantico", None), ("gemeo", None), ("classico", None)],
                     "Part A: |P|=50, 100 repetitions")
    linhas.append("")
    linhas += tabela([r for r in res if r[0] == "B"], chB, "Part B: entanglement between individuals, |P|=10, 200 repetitions")
    # part B differences at generations 5 and 30 (with - without), with standard error
    linhas.append("")
    linhas.append("Part B, difference (with entanglement - without), mean (SE), at generations 5 and 30")
    for dom in DOMINIOS:
        base = np.array([r[4] for r in res if r[0] == "B" and r[1] == "quantico" and r[2] is None and r[3] == dom])
        for modo, emar in chB[1:]:
            c = np.array([r[4] for r in res if r[0] == "B" and r[1] == modo and r[2] == emar and r[3] == dom])
            partes = []
            for g in (5, 30):
                d = c[:, g].mean() - base[:, g].mean()
                ep = np.sqrt(c[:, g].var(ddof=1) / len(c) + base[:, g].var(ddof=1) / len(base))
                partes.append(f"g{g}: {d:+.3f} ({ep:.3f})")
            linhas.append(f"{dom} {modo}+{emar} | " + " | ".join(partes))
    linhas.append(f"\ntime {time.time() - t0:.0f}s")
    print("\n".join(linhas))
    (RAIZ / "results" / "qga" / f"experiment_souza_v1-{date.today()}.txt").write_text("\n".join(linhas) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
