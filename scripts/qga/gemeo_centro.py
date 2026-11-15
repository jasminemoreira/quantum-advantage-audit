"""Hipotese: os finais que diferem do gemeo sao melhores no centrado por estarem mais perto do centro.

Mesmos AGs de gemeo_aptidao.py (mesmas sementes, 100 por dominio). Para cada melhor final:
igual (decidido no anel), fq = f(E[x]) quantica exata, distancia de E[x] ao centro do dominio,
distancia de E[x] ao otimo (origem), entropia H da saida em bits.
Previsao (vies de centro): no centrado, os diferentes tem E[x] mais perto do centro (= otimo) e
entropia maior; no deslocado, se ficarem mais perto do centro, isso nao os aproxima do otimo.

Uso (WSL): externos/QFIE/.venv/bin/python scripts/qga/gemeo_centro.py
Saida: resultados/qga/gemeo_centro-<data>.txt
"""
from __future__ import annotations

import sys
import time
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "qcalc"))
import gemeo_exato as GE  # noqa: E402
import gemeo_exato_anel as A  # noqa: E402
import souza_v1 as S  # noqa: E402
import vies_centro as VC  # noqa: E402

RAIZ = AQUI.parents[1]
REPS = 100
XS = {d: VC.decodifica(np.arange(S.DIM), d) for d in GE.DOMINIOS}


def entropia(p):
    q = p[p > 1e-15]
    return float(-(q * np.log2(q)).sum())


def tarefa(args):
    dom, sem = args
    _, melhor, _ = S.ga("quantico", dom, sem, pop_n=50, ger=30)
    _, igual, _, _ = A.decide(melhor)
    pq = S.p_quantico(melhor)
    ex = (XS[dom] * pq[:, None]).sum(axis=0)
    c = (dom[0] + dom[1]) / 2
    return dom, igual, float(VC.rastrigin(ex)), float(np.linalg.norm(ex - c)), float(np.linalg.norm(ex)), entropia(pq)


def compara(nome, a, b):
    u = mannwhitneyu(a, b, alternative="two-sided")
    return f"  {nome}: diferentes {np.median(a):.3f} | iguais {np.median(b):.3f} | Mann-Whitney p = {u.pvalue:.3f}"


def main():
    t0 = time.time()
    tarefas = [(d, GE.SEMENTE + 1000 * k + r) for k, d in enumerate(GE.DOMINIOS) for r in range(REPS)]
    with Pool(16) as pool:
        rs = pool.map(tarefa, tarefas, chunksize=1)
    L = [f"Distancia ao centro: finais iguais ao gemeo x diferentes | semente {GE.SEMENTE} | {date.today()}",
         f"|P|=50, G=30, {REPS} AGs por dominio (mesmos de gemeo_aptidao.py); medianas", ""]
    for d in GE.DOMINIOS:
        sub = [r for r in rs if r[0] == d]
        dif = [r for r in sub if not r[1]]
        ig = [r for r in sub if r[1]]
        col = lambda rr, i: np.array([r[i] for r in rr])  # noqa: E731
        L.append(f"dominio {d}: iguais {len(ig)}, diferentes {len(dif)}")
        L.append(compara("f(E[x])", col(dif, 2), col(ig, 2)))
        L.append(compara("distancia de E[x] ao centro", col(dif, 3), col(ig, 3)))
        L.append(compara("distancia de E[x] ao otimo (origem)", col(dif, 4), col(ig, 4)))
        L.append(compara("entropia H (bits)", col(dif, 5), col(ig, 5)))
        todos = sub
        L.append(f"  Spearman(distancia ao centro, f) em todos os {len(todos)}: "
                 f"{spearmanr(col(todos, 3), col(todos, 2)).statistic:+.2f}")
        L.append("")
    L.append(f"tempo {time.time() - t0:.0f}s")
    print("\n".join(L))
    (RAIZ / "resultados" / "qga" / f"gemeo_centro-{date.today()}.txt").write_text("\n".join(L) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
