"""Sensitivity analysis of the reimplementation of Souza/Dardenne/Portugal (13 Nov 2026).

Each choice not fixed by their text (see souza_v1.py, OPC) is replaced by an alternative, one at a
time, plus one variant with all alternatives together. For each variant, the predictions that
support the argument of the article:
  P1 centered domain, g0: quantum better than classical (center bias helps);
  P2 shifted domain, g0: quantum worse than classical (center bias hurts);
  P3 shifted domain, g30: quantum does not beat the classical;
  P4 the classical analog is not worse than the quantum (g0 and g30, on both domains);
  P5 "bell" entanglement between individuals helps on the centered domain and hurts on the shifted one (g30);
  P6 the classical equivalent of "bell" reproduces quantum+bell (g30);
  P7 entanglement within the best final individual: more frequent on the centered than on the shifted domain.
Part A: |P| = 50, 30 generations, 50 repetitions. Part B: |P| = 10, 30 generations, 100 repetitions.
Schmidt rank between the registers computed in floating point (SVD, cutoff 1e-8); the exact
version was already done for the default variant in certify_souza_v1.py.

Usage: python scripts/qga/sensitivity_souza.py [variant ...]
Output: results/qga/sensitivity_souza-<date>.txt
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
import center_bias as VC  # noqa: E402

RAIZ = AQUI.parents[1]
SEMENTE = 20261117
DOMINIOS = ((-5.12, 5.12), (-2.12, 8.12))
GER = 30
REPS_A, REPS_B = 50, 100

VARIANTES = {
    "padrao": {},
    "prof3": dict(prof=3),
    "prof8": dict(prof=8),
    "roleta": dict(selecao="roleta"),
    "regiao_global": dict(regiao="global"),
    "descarta_fica": dict(colisao="descarta_fica"),
    "sorteio_aridade": dict(sorteio="por_aridade"),
    "h_congelada": dict(h_muta=False),
    "base_cruza": dict(base="cruza"),
    "todas_alt": dict(selecao="roleta", regiao="global", colisao="descarta_fica",
                      sorteio="por_aridade", h_muta=False, base="cruza"),
}


def posto_registros(ind):
    psi = VC.estado(S.sem_i(ind)).reshape(1 << S.NB, 1 << S.NB)  # rows: register 1; columns: register 0
    s = np.linalg.svd(psi, compute_uv=False)
    return int((s > 1e-8 * s[0]).sum())


def tarefa(args):
    var, parte, modo, emar, dom, pop_n, sem = args
    S.OPC.clear()
    S.OPC.update(S.PADRAO, **VARIANTES[var])
    curva, melhor, _ = S.ga(modo, dom, sem, pop_n=pop_n, ger=GER, emar=emar)
    posto = posto_registros(melhor) if (parte == "A" and modo == "quantico") else -1
    return var, parte, modo, emar, dom, curva, posto


def dif(res, a, b, g):
    """mean(a) - mean(b) at generation g, with standard error."""
    ca = np.array([r[5][g] for r in res if (r[1], r[2], r[3], r[4]) == a])
    cb = np.array([r[5][g] for r in res if (r[1], r[2], r[3], r[4]) == b])
    return ca.mean() - cb.mean(), np.sqrt(ca.var(ddof=1) / len(ca) + cb.var(ddof=1) / len(cb))


def fmt(d):
    return f"{d[0]:+.2f} ({d[1]:.2f})"


def relatorio(var, res):
    C, D = DOMINIOS
    q, gm, cl = ("A", "quantico", None), ("A", "gemeo", None), ("A", "classico", None)
    qb, qe, ge = ("B", "quantico", None), ("B", "quantico", "bell"), ("B", "gemeo", "bell")
    t = lambda x, dom: x + (dom,)  # noqa: E731
    p1 = dif(res, t(q, C), t(cl, C), 0)
    p2 = dif(res, t(q, D), t(cl, D), 0)
    p3 = dif(res, t(q, D), t(cl, D), GER)
    p4 = [dif(res, t(gm, dom), t(q, dom), g) for dom in DOMINIOS for g in (0, GER)]
    p5c, p5d = dif(res, t(qe, C), t(qb, C), GER), dif(res, t(qe, D), t(qb, D), GER)
    p6 = [dif(res, t(ge, dom), t(qe, dom), GER) for dom in DOMINIOS]
    postos = {dom: np.array([r[6] for r in res if r[1] == "A" and r[2] == "quantico" and r[4] == dom]) for dom in DOMINIOS}
    sep = {dom: (postos[dom] == 1).mean() for dom in DOMINIOS}
    ok = dict(
        P1=p1[0] + 2 * p1[1] < 0,
        P2=p2[0] - 2 * p2[1] > 0,
        P3=p3[0] + 2 * p3[1] > 0,  # quantum not significantly better
        P4=all(d[0] - 2 * d[1] < 0 for d in p4),  # analog not significantly worse
        P5=p5c[0] + 2 * p5c[1] < 0 and p5d[0] - 2 * p5d[1] > 0,
        P6=all(abs(d[0]) < 2.5 * d[1] or abs(d[0]) < 0.5 for d in p6),
        P7=sep[C] < sep[D],
    )
    linhas = [f"[{var}] {VARIANTES[var] or 'declared choices'}",
              f"  P1 centered g0, quantum - classical: {fmt(p1)}",
              f"  P2 shifted g0, quantum - classical: {fmt(p2)}",
              f"  P3 shifted g30, quantum - classical: {fmt(p3)}",
              "  P4 analog - quantum (centered g0, g30; shifted g0, g30): " + ", ".join(fmt(d) for d in p4),
              f"  P5 bell - without (g30): centered {fmt(p5c)}, shifted {fmt(p5d)}",
              "  P6 analog+bell - quantum+bell (g30): centered " + fmt(p6[0]) + ", shifted " + fmt(p6[1]),
              f"  P7 best final separable between registers: centered {sep[C]:.0%} (median rank {np.median(postos[C]):.0f}), "
              f"shifted {sep[D]:.0%} (median rank {np.median(postos[D]):.0f})",
              "  confirmed: " + " ".join(f"{k}={'yes' if v else 'NO'}" for k, v in ok.items())]
    return linhas, ok


def main():
    t0 = time.time()
    nomes = sys.argv[1:] or list(VARIANTES)
    tarefas = []
    for v, var in enumerate(nomes):
        for k, dom in enumerate(DOMINIOS):
            for j, modo in enumerate(("quantico", "gemeo", "classico")):
                tarefas += [(var, "A", modo, None, dom, 50, SEMENTE + 10 ** 6 * v + 10000 * k + 1000 * j + r) for r in range(REPS_A)]
            for j, (modo, emar) in enumerate((("quantico", None), ("quantico", "bell"), ("gemeo", "bell"))):
                tarefas += [(var, "B", modo, emar, dom, 10, SEMENTE + 10 ** 6 * v + 500000 + 10000 * k + 1000 * j + r) for r in range(REPS_B)]
    # expensive tasks first, to balance the queue
    tarefas.sort(key=lambda a: -(a[5] * (3 if a[2] == "quantico" else 1) * (2 if a[3] else 1) * (8 if a[0] == "prof8" else 1)))
    with Pool(16) as pool:
        res = pool.map(tarefa, tarefas, chunksize=1)
    linhas = [f"Sensitivity of the Souza/Dardenne/Portugal reimplementation | seed {SEMENTE} | {date.today()}",
              f"Part A |P|=50, {REPS_A} rep.; Part B |P|=10, {REPS_B} rep.; {GER} generations; differences: mean (standard error)",
              "criterion: P1, P2, P5 with |diff| > 2 SE in the predicted direction; P3, P4 with no difference > 2 SE against the argument; "
              "P6 with |diff| < 2.5 SE or < 0.5; P7 separable fraction smaller on the centered domain", ""]
    resumo = []
    for var in nomes:
        ls, ok = relatorio(var, [r for r in res if r[0] == var])
        linhas += ls + [""]
        resumo.append(f"{var:16s} " + " ".join("yes" if ok[k] else "NO" for k in ok))
    linhas += ["summary (P1 to P7):"] + resumo + [f"\ntime {time.time() - t0:.0f}s"]
    print("\n".join(linhas))
    (RAIZ / "results" / "qga" / f"sensitivity_souza-{date.today()}.txt").write_text("\n".join(linhas) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
