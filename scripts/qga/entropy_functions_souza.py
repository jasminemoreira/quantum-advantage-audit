"""Entropy and other test functions in the reimplementation of Souza/Dardenne/Portugal (14 Nov 2026).

Part E (entropy figures of their article): they show that circuits with higher output Shannon
entropy give smaller f on the centered Rastrigin and read this as an effect of superposition.
Prediction of the center bias: more spread-out output -> mean closer to the domain center -> f closer
to f(center). On the centered domain f(center) = 0 (optimum), so entropy "helps"; on the shifted
domain f(center) = 18, so entropy helps individuals worse than 18 (the drawn ones) and hurts the
better ones (the final ones).
  E1: 5000 drawn initial individuals (no selection); for each one, entropy H(p) in bits,
      distance from E[x] to the domain center and exact f(E[x]) on both domains;
  E2: best final individual of 300 GAs per domain, |P| = 50, G = 50 (as in their figure).
Part F (the other functions their article cites as "qualitatively similar"): Rastrigin,
Sphere (esfera), Ackley, Griewank, Rosenbrock on [-2.048, 2.048] and Rosenbrock on [-30, 30], on the
usual domain (symmetric around the origin) and on a shifted domain in which the optimum lies at 20.7%
of the width from the left edge (the same relative position as x = 0 in [-2.12, 8.12]).
Their article does not report the domains of these functions; the usual ones are our choice.
|P| = 50, G = 30, 50 repetitions; quantum evaluator, classical analog and classical evaluator.

Usage: python scripts/qga/entropy_functions_souza.py
Output: results/qga/entropy_functions_souza-<date>.txt
"""
from __future__ import annotations

import sys
import time
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import souza_v1 as S  # noqa: E402
import center_bias as VC  # noqa: E402

RAIZ = AQUI.parents[1]
SEMENTE = 20261118
DOM_R = ((-5.12, 5.12), (-2.12, 8.12))
N_E1, REPS_E2, GER_E2 = 5000, 300, 50
REPS_F, GER_F = 50, 30
X_TODOS = None


def deslocado(otimo, a, b):
    w = b - a
    return (round(otimo - 0.207 * w, 4), round(otimo + 0.793 * w, 4))


FUNCS = [("rastrigin", "rastrigin", (-5.12, 5.12)), ("esfera", "esfera", (-5.12, 5.12)),
         ("ackley", "ackley", (-32.768, 32.768)), ("griewank", "griewank", (-600.0, 600.0)),
         ("rosenbrock", "rosenbrock", (-2.048, 2.048)), ("rosenbrock30", "rosenbrock", (-30.0, 30.0))]


def entropia(p):
    q = p[p > 1e-15]
    return float(-(q * np.log2(q)).sum())


def medias(p, dom):
    return (VC.decodifica(np.arange(S.DIM), dom) * p[:, None]).sum(axis=0)


def e1(sem):
    rng = np.random.default_rng(sem)
    ind = S.individuo(rng, S.G_QU)
    p = S.p_quantico(ind)
    out = [entropia(p)]
    for dom in DOM_R:
        ex = medias(p, dom)
        c = (dom[0] + dom[1]) / 2
        out += [float(np.linalg.norm(ex - c)), VC.rastrigin(ex)]
    return out


def e2(args):
    dom, sem = args
    S.OPC.clear()
    S.OPC.update(S.PADRAO)
    _, mel, f = S.ga("quantico", dom, sem, pop_n=50, ger=GER_E2)
    p = S.p_quantico(mel)
    ex = medias(p, dom)
    c = (dom[0] + dom[1]) / 2
    return dom, entropia(p), f, VC.rastrigin(ex), float(np.linalg.norm(ex - c))


def ftarefa(args):
    nome, func, dom, modo, sem = args
    S.OPC.clear()
    S.OPC.update(S.PADRAO, funcao=func)
    curva, _, _ = S.ga(modo, dom, sem, pop_n=50, ger=GER_F)
    return nome, dom, modo, curva


def por_faixa(H, f):
    linhas = []
    for h in range(0, 17, 2):
        m = (H >= h) & (H < h + 2)
        if m.sum() >= 5:
            linhas.append(f"[{h},{h + 2}): n={m.sum()}, median f {np.median(f[m]):.2f}")
    return "; ".join(linhas)


def main():
    t0 = time.time()
    tarefas_f = []
    for i, (nome, func, dom) in enumerate(FUNCS):
        doms = (dom, deslocado(S.OTIMOS[func], *dom))
        for k, d in enumerate(doms):
            for j, modo in enumerate(("quantico", "gemeo", "classico")):
                tarefas_f += [(nome, func, d, modo, SEMENTE + 10 ** 6 * (i + 1) + 10000 * k + 1000 * j + r) for r in range(REPS_F)]
    tarefas_e2 = [(dom, SEMENTE + 500000 + 10000 * k + r) for k, dom in enumerate(DOM_R) for r in range(REPS_E2)]
    with Pool(16) as pool:
        r1 = np.array(pool.map(e1, [SEMENTE + r for r in range(N_E1)], chunksize=20))
        r2 = pool.map(e2, tarefas_e2, chunksize=1)
        rf = pool.map(ftarefa, sorted(tarefas_f, key=lambda a: a[3] != "quantico"), chunksize=2)

    L = [f"Entropy and other functions, Souza/Dardenne/Portugal reimplementation (v1 operators) | seed {SEMENTE} | {date.today()}", ""]
    H = r1[:, 0]
    L.append(f"E1: {N_E1} drawn initial individuals (no selection), Rastrigin; H in bits (0 to 16)")
    for k, dom in enumerate(DOM_R):
        dist, f = r1[:, 1 + 2 * k], r1[:, 2 + 2 * k]
        fc = VC.rastrigin(np.full(S.M, (dom[0] + dom[1]) / 2))
        L.append(f"  {dom}: f(center) = {fc:.2f} | Spearman(H, f) = {spearmanr(H, f).statistic:+.2f} | "
                 f"Spearman(H, |f - f(center)|) = {spearmanr(H, np.abs(f - fc)).statistic:+.2f} | "
                 f"Spearman(H, distance from E[x] to the center) = {spearmanr(H, dist).statistic:+.2f}")
        L.append(f"    by H band: {por_faixa(H, f)}")
    L.append("")
    L.append(f"E2: best final individual, |P|=50, G={GER_E2}, {REPS_E2} GAs per domain")
    for dom in DOM_R:
        rs = [r for r in r2 if r[0] == dom]
        h2 = np.array([r[1] for r in rs])
        fa = np.array([r[2] for r in rs])
        fx = np.array([r[3] for r in rs])
        di = np.array([r[4] for r in rs])
        fc = VC.rastrigin(np.full(S.M, (dom[0] + dom[1]) / 2))
        L.append(f"  {dom}: f(center) = {fc:.2f} | Spearman(H, |f(E[x]) - f(center)|) = {spearmanr(h2, np.abs(fx - fc)).statistic:+.2f}")
        L.append(f"    Spearman(H, evaluated f) = {spearmanr(h2, fa).statistic:+.2f} | Spearman(H, f(E[x])) = "
                 f"{spearmanr(h2, fx).statistic:+.2f} | Spearman(H, distance to the center) = {spearmanr(h2, di).statistic:+.2f} | "
                 f"H median {np.median(h2):.2f}, fraction with H = 0: {(h2 < 1e-9).mean():.0%}")
        L.append(f"    by H band (evaluated f): {por_faixa(h2, fa)}")
    L.append("")
    L.append(f"F: |P|=50, G={GER_F}, {REPS_F} repetitions; best evaluated f, mean (standard error); f(center) = value of the function at the domain center")
    L.append("function | domain | f(center) | quantum g0 | classical g0 | analog g0 | quantum g30 | classical g30 | analog g30 | "
             "q - c g0 | q - c g30")
    for nome, func, dom in FUNCS:
        for d in (dom, deslocado(S.OTIMOS[func], *dom)):
            cur = {m: np.array([r[3] for r in rf if r[0] == nome and r[1] == d and r[2] == m]) for m in ("quantico", "gemeo", "classico")}
            ms = lambda m, g: f"{cur[m][:, g].mean():.3g} ({cur[m][:, g].std(ddof=1) / np.sqrt(REPS_F):.2g})"  # noqa: E731

            def dq(g):
                a, b = cur["quantico"][:, g], cur["classico"][:, g]
                return f"{a.mean() - b.mean():+.3g} ({np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)):.2g})"
            fc = S.FUNCOES[func](np.full(S.M, (d[0] + d[1]) / 2))
            L.append(f"{nome} | {d} | {fc:.3g} | {ms('quantico', 0)} | {ms('classico', 0)} | {ms('gemeo', 0)} | "
                     f"{ms('quantico', GER_F)} | {ms('classico', GER_F)} | {ms('gemeo', GER_F)} | {dq(0)} | {dq(GER_F)}")
    L.append(f"\ntime {time.time() - t0:.0f}s")
    print("\n".join(L))
    (RAIZ / "results" / "qga" / f"entropy_functions_souza-{date.today()}.txt").write_text("\n".join(L) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
