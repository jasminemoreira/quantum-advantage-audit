"""Decisao EXATA de 'gemeo classico = circuito quantico', individuo a individuo (14/11/2026).

Mesmos individuos de gemeo_exato.py (mesma semente: 300 iniciais sorteados e os melhores finais de
30 AGs por dominio). Para cada um:
  - distribuicao quantica exata no motor esparso em Z[zeta_16]: P(x) = num(x) / 2^k;
  - distribuicao exata do gemeo em racionais diadicos: P(x) = w(x) / 2^h, h = numero de H;
  - igualdade decidida sem ponto flutuante: 2^h * num(x) == w(x) * 2^k * (1, 0, ..., 0) para todo x.
Relata, por classe do corolario (corolario, so_T1, fora), quantos sao exatamente iguais e confere
com a TV em ponto flutuante de gemeo_exato.py.

Uso (WSL): externos/QFIE/.venv/bin/python scripts/qga/gemeo_exato_anel.py
Saida: resultados/qga/gemeo_exato_anel-<data>.txt
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
sys.path.insert(0, str(AQUI.parent / "qcalc"))
import certifica_souza as CS  # noqa: E402
import gemeo_exato as GE  # noqa: E402
import souza_v1 as S  # noqa: E402

RAIZ = AQUI.parents[1]
IDX = np.arange(S.DIM)


def gemeo_inteiro(ind):
    """Pesos inteiros w(x) e expoente h com P_gemeo(x) = w(x) / 2^h."""
    w = np.zeros(S.DIM, dtype=object)
    w[ind["base"]] = 1
    h = 0
    for cam in S.sem_i(ind)["cams"]:
        for nome, qs in cam:
            if nome in GE.DIAG:
                continue
            if nome == "h":
                w = w + w[IDX ^ (1 << qs[0])]
                h += 1
            elif nome in GE.FLIP:
                w = w[IDX ^ (1 << qs[0])]
            else:
                b = lambda q: (IDX >> q) & 1  # noqa: E731
                if nome == "cx":
                    pi = IDX ^ (b(qs[0]) << qs[1])
                elif nome == "ccx":
                    pi = IDX ^ ((b(qs[0]) & b(qs[1])) << qs[2])
                else:
                    a, c = qs[-2:]
                    troca = (b(a) != b(c)) if nome == "swap" else ((b(a) != b(c)) & (b(qs[0]) == 1))
                    pi = np.where(troca, IDX ^ (1 << a) ^ (1 << c), IDX)
                w = w[pi]
    return w, h


def decide(ind):
    t0 = time.time()
    m = CS.estado_exato(S.sem_i(ind))
    marg = m.marginal(list(range(S.NQ)))
    w, h = gemeo_inteiro(ind)
    suporte_g = set(int(x) for x in np.nonzero(w)[0])
    igual = set(marg) == suporte_g
    if igual:
        for x, num in marg.items():
            alvo = (int(w[x]) * 2 ** m.k,) + (0,) * 7
            if tuple(2 ** h * a for a in num) != alvo:
                igual = False
                break
    classe, tv = GE.analisa(ind)
    return classe, igual, tv, time.time() - t0


def final(args):
    dom, sem = args
    _, melhor, _ = S.ga("quantico", dom, sem, pop_n=50, ger=30)
    return dom, decide(melhor)


def resumo(rot, rs):
    L = [f"{rot}: {len(rs)} individuos | exatamente iguais ao gemeo: {sum(r[1] for r in rs)} "
         f"({sum(r[1] for r in rs) / len(rs):.0%})"]
    for c in ("corolario", "so_T1", "fora"):
        sub = [r for r in rs if r[0] == c]
        if sub:
            ig = sum(r[1] for r in sub)
            disc = sum((r[1] != (r[2] < 1e-12)) for r in sub)
            L.append(f"  {c:9s}: {len(sub):4d} | iguais (exato) {ig:4d} ({ig / len(sub):.0%}) | "
                     f"discordancias com TV<1e-12 do float: {disc}")
    return L


def main():
    t0 = time.time()
    rng = np.random.default_rng(GE.SEMENTE)
    iniciais = [S.individuo(rng, S.G_QU) for _ in range(GE.N_INI)]
    with Pool(16) as pool:
        r_ini = pool.map(decide, iniciais, chunksize=4)
        r_fin = pool.map(final, [(d, GE.SEMENTE + 1000 * k + r) for k, d in enumerate(GE.DOMINIOS)
                                 for r in range(GE.REPS)], chunksize=1)
    L = [f"Gemeo = quantico decidido no anel Z[zeta_16] | semente {GE.SEMENTE} (a de gemeo_exato.py) | {date.today()}",
         "igualdade exata das distribuicoes medidas completas (16 qubits); classes do Corolario 1", ""]
    L += resumo("iniciais sorteados (G_qu, v1)", r_ini)
    fins = []
    for d in GE.DOMINIOS:
        rs = [r for dd, r in r_fin if dd == d]
        fins += rs
        L += resumo(f"melhores finais, |P|=50, G=30, dominio {d}", rs)
    ok_cor = all(r[1] for r in r_ini + fins if r[0] == "corolario")
    disc = sum((r[1] != (r[2] < 1e-12)) for r in r_ini + fins)
    L.append(f"\nCorolario 1 confirmado no exato (todo 'corolario' igual): {ok_cor}")
    L.append(f"discordancias entre o exato e o criterio float TV < 1e-12: {disc}")
    L.append(f"tempo {time.time() - t0:.0f}s; maior tempo por individuo {max(r[3] for r in r_ini + fins):.1f}s")
    print("\n".join(L))
    (RAIZ / "resultados" / "qga" / f"gemeo_exato_anel-{date.today()}.txt").write_text("\n".join(L) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
