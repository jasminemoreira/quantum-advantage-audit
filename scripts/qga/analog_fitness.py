"""Are the best final individuals that differ from the analog better? (14 Nov 2026)

For the best final individuals (|P| = 50, G = 30, 100 GAs per domain; seeds of analog_exact.py,
which include the 30 already analysed), computes:
  - igual: analog exactly equal to the quantum circuit (decided in the ring, analog_exact_ring.decide);
  - fq: exact f(E[x]) (no shot noise) from the quantum distribution;
  - fg: f(E[x]) from the exact distribution of the analog of the same individual.
Compares (a) fq between the equal / different groups (Mann-Whitney) and (b) within the different
ones, fq against fg (what interference does to that individual; paired Wilcoxon).

Usage: python scripts/qga/analog_fitness.py
Output: results/qga/analog_fitness-<date>.txt
"""
from __future__ import annotations

import sys
import time
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "qcalc"))
import analog_exact as GE  # noqa: E402
import analog_exact_ring as A  # noqa: E402
import souza_v1 as S  # noqa: E402
import center_bias as VC  # noqa: E402

RAIZ = AQUI.parents[1]
REPS = 100
XS = {d: VC.decodifica(np.arange(S.DIM), d) for d in GE.DOMINIOS}


def tarefa(args):
    dom, sem = args
    _, melhor, f_aval = S.ga("quantico", dom, sem, pop_n=50, ger=30)
    classe, igual, tv, _ = A.decide(melhor)
    pq = S.p_quantico(melhor)
    pg = GE.p_gemeo(melhor)
    fq = VC.rastrigin((XS[dom] * pq[:, None]).sum(axis=0))
    fg = VC.rastrigin((XS[dom] * pg[:, None]).sum(axis=0))
    return dom, igual, classe, float(fq), float(fg), float(f_aval), tv


def main():
    t0 = time.time()
    tarefas = [(d, GE.SEMENTE + 1000 * k + r) for k, d in enumerate(GE.DOMINIOS) for r in range(REPS)]
    with Pool(16) as pool:
        rs = pool.map(tarefa, tarefas, chunksize=1)
    L = [f"Fitness of the best final individuals: equal to the analog x different | seed {GE.SEMENTE} | {date.today()}",
         f"|P|=50, G=30, {REPS} GAs per domain; fq = exact quantum f(E[x]); fg = f(E[x]) of the analog of the same individual",
         "equality decided in the ring Z[zeta_16]; tests: Mann-Whitney (groups), paired Wilcoxon (fq x fg)", ""]
    for d in GE.DOMINIOS:
        sub = [r for r in rs if r[0] == d]
        ig = np.array([r[3] for r in sub if r[1]])
        di = [r for r in sub if not r[1]]
        fq_d = np.array([r[3] for r in di])
        fg_d = np.array([r[4] for r in di])
        L.append(f"domain {d}: equal {len(ig)}, different {len(di)}")
        L.append(f"  fq median: equal {np.median(ig):.4f} | different {np.median(fq_d):.4f}" if len(di) else
                 f"  fq median: equal {np.median(ig):.4f}")
        if len(di) >= 3 and len(ig) >= 3:
            u = mannwhitneyu(fq_d, ig, alternative="two-sided")
            L.append(f"  Mann-Whitney different x equal: p = {u.pvalue:.3f} | fraction of pairs (diff, equal) with diff better: "
                     f"{np.mean([a < b for a in fq_d for b in ig]):.2f}")
        if len(di):
            melhora = (fq_d < fg_d - 1e-12).sum()
            piora = (fq_d > fg_d + 1e-12).sum()
            L.append(f"  within the different ones: quantum better than its own analog in {melhora}, worse in {piora}, "
                     f"tie in {len(di) - melhora - piora} | median fq - fg = {np.median(fq_d - fg_d):+.4f}")
            if len(di) >= 6 and np.any(fq_d != fg_d):
                w = wilcoxon(fq_d, fg_d)
                L.append(f"  paired Wilcoxon fq x fg: p = {w.pvalue:.3f}")
            L.append("  different (fq, fg, class): " + "; ".join(f"{r[3]:.3f}/{r[4]:.3f}/{r[2]}" for r in sorted(di, key=lambda r: r[3])))
        L.append("")
    L.append(f"time {time.time() - t0:.0f}s")
    print("\n".join(L))
    (RAIZ / "results" / "qga" / f"analog_fitness-{date.today()}.txt").write_text("\n".join(L) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
