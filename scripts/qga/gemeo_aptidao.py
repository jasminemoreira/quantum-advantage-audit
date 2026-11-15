"""Os melhores finais que diferem do gemeo sao melhores? (14/11/2026)

Para os melhores individuos finais (|P| = 50, G = 30, 100 AGs por dominio; sementes de
gemeo_exato.py, que incluem as 30 ja analisadas), calcula:
  - igual: gemeo exatamente igual ao quantico (decidido no anel, gemeo_exato_anel.decide);
  - fq: f(E[x]) exata (sem ruido de shots) pela distribuicao quantica;
  - fg: f(E[x]) pela distribuicao exata do gemeo do mesmo individuo.
Compara (a) fq entre os grupos igual / diferente (Mann-Whitney) e (b) dentro dos diferentes,
fq contra fg (o que a interferencia faz com aquele individuo; Wilcoxon pareado).

Uso (WSL): externos/QFIE/.venv/bin/python scripts/qga/gemeo_aptidao.py
Saida: resultados/qga/gemeo_aptidao-<data>.txt
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
import gemeo_exato as GE  # noqa: E402
import gemeo_exato_anel as A  # noqa: E402
import souza_v1 as S  # noqa: E402
import vies_centro as VC  # noqa: E402

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
    L = [f"Aptidao dos melhores finais: iguais ao gemeo x diferentes | semente {GE.SEMENTE} | {date.today()}",
         f"|P|=50, G=30, {REPS} AGs por dominio; fq = f(E[x]) quantica exata; fg = f(E[x]) do gemeo do mesmo individuo",
         "igualdade decidida no anel Z[zeta_16]; testes: Mann-Whitney (grupos), Wilcoxon pareado (fq x fg)", ""]
    for d in GE.DOMINIOS:
        sub = [r for r in rs if r[0] == d]
        ig = np.array([r[3] for r in sub if r[1]])
        di = [r for r in sub if not r[1]]
        fq_d = np.array([r[3] for r in di])
        fg_d = np.array([r[4] for r in di])
        L.append(f"dominio {d}: iguais {len(ig)}, diferentes {len(di)}")
        L.append(f"  fq mediana: iguais {np.median(ig):.4f} | diferentes {np.median(fq_d):.4f}" if len(di) else
                 f"  fq mediana: iguais {np.median(ig):.4f}")
        if len(di) >= 3 and len(ig) >= 3:
            u = mannwhitneyu(fq_d, ig, alternative="two-sided")
            L.append(f"  Mann-Whitney diferentes x iguais: p = {u.pvalue:.3f} | fracao de pares (dif, igual) com dif melhor: "
                     f"{np.mean([a < b for a in fq_d for b in ig]):.2f}")
        if len(di):
            melhora = (fq_d < fg_d - 1e-12).sum()
            piora = (fq_d > fg_d + 1e-12).sum()
            L.append(f"  dentro dos diferentes: quantico melhor que o proprio gemeo em {melhora}, pior em {piora}, "
                     f"empate em {len(di) - melhora - piora} | mediana fq - fg = {np.median(fq_d - fg_d):+.4f}")
            if len(di) >= 6 and np.any(fq_d != fg_d):
                w = wilcoxon(fq_d, fg_d)
                L.append(f"  Wilcoxon pareado fq x fg: p = {w.pvalue:.3f}")
            L.append("  diferentes (fq, fg, classe): " + "; ".join(f"{r[3]:.3f}/{r[4]:.3f}/{r[2]}" for r in sorted(di, key=lambda r: r[3])))
        L.append("")
    L.append(f"tempo {time.time() - t0:.0f}s")
    print("\n".join(L))
    (RAIZ / "resultados" / "qga" / f"gemeo_aptidao-{date.today()}.txt").write_text("\n".join(L) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
