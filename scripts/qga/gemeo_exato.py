"""Quando o gemeo classico e EXATAMENTE igual ao circuito quantico (14/11/2026).

Corolario do Teorema 1 (provas-2026-11-14.md): um individuo de Souza (estado inicial de base,
camadas de G_qu) cujo circuito obedece as regras R0, R1, R2 e no qual toda porta H age num qubit que,
naquele instante, esta num estado de base (fresco sem porta nao monomial desde o inicio, ou bit
classico re-sorteado pela regra R2) tem a mesma distribuicao medida que o gemeo classico
(H -> XOR com moeda justa, Y -> X, fases descartadas).

Classes de cada individuo:
  corolario : obedece ao corolario -> previsao TV(gemeo, quantico) = 0;
  so_T1     : obedece ao Teorema 1 (amostragem classica), mas alguma H age num qubit fresco ja em
              superposicao -> o gemeo pode diferir;
  fora      : nao obedece ao Teorema 1 (pode haver interferencia).
Conferencia: distribuicao exata do gemeo (propagacao de probabilidades) contra p_quantico (statevector).

Uso (WSL): externos/QFIE/.venv/bin/python scripts/qga/gemeo_exato.py
Saida: resultados/qga/gemeo_exato-<data>.txt
"""
from __future__ import annotations

import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import souza_v1 as S  # noqa: E402

RAIZ = AQUI.parents[1]
SEMENTE = 20261121
DOMINIOS = ((-5.12, 5.12), (-2.12, 8.12))
N_INI, REPS = 300, 30
IDX = np.arange(S.DIM)
DIAG = {"z", "s", "sdg", "t", "tdg", "i"}
FLIP = {"x", "y"}
PERM = {"cx", "ccx", "swap", "cswap"}


def classe_individuo(ind):
    """Aplica as regras do Teorema 1 (com R2 e trava da testemunha) mais a condicao do corolario."""
    n = S.NQ
    estado = {q: "base" for q in range(n)}  # base | super (frescos) | classico
    grupo = {q: {q} for q in range(n)}
    travado = set()
    corolario = True

    def sai(q):
        grupo[q].discard(q)
        grupo[q] = {q}

    for cam in S.sem_i(ind)["cams"]:
        for nome, qs in cam:
            if any(q in travado for q in qs):
                return "fora"
            if nome in DIAG or nome in FLIP:
                continue
            if nome == "h":
                q = qs[0]
                if estado[q] == "base":
                    estado[q] = "super"
                elif estado[q] == "super":
                    corolario = False
                else:  # classico: precisa de testemunha (R2)
                    outros = grupo[q] - {q}
                    if not outros:
                        return "fora"
                    travado.update(outros)
                    sai(q)
                    estado[q] = "super"
                continue
            assert nome in PERM, nome
            if nome == "cx" and estado[qs[1]] == "base":
                c, t = qs
                sai(t)
                grupo[c].add(t)
                grupo[t] = grupo[c]
            else:
                for t in (qs[-2:] if "swap" in nome else qs[-1:]):
                    sai(t)
            for q in qs:
                estado[q] = "classico"
    return "corolario" if corolario else "so_T1"


def p_gemeo(ind):
    """Distribuicao exata do gemeo classico."""
    p = np.zeros(S.DIM)
    p[ind["base"]] = 1.0
    for cam in S.sem_i(ind)["cams"]:
        for nome, qs in cam:
            if nome in DIAG:
                continue
            if nome == "h":
                p = 0.5 * (p + p[IDX ^ (1 << qs[0])])
            elif nome in FLIP:
                p = p[IDX ^ (1 << qs[0])]
            else:  # permutacoes involutivas: p_novo = p o pi
                b = lambda q: (IDX >> q) & 1  # noqa: E731
                if nome == "cx":
                    pi = IDX ^ (b(qs[0]) << qs[1])
                elif nome == "ccx":
                    pi = IDX ^ ((b(qs[0]) & b(qs[1])) << qs[2])
                else:
                    a, c = qs[-2:]
                    troca = (b(a) != b(c)) if nome == "swap" else ((b(a) != b(c)) & (b(qs[0]) == 1))
                    pi = np.where(troca, IDX ^ (1 << a) ^ (1 << c), IDX)
                p = p[pi]
    return p


def analisa(ind):
    tv = 0.5 * np.abs(p_gemeo(ind) - S.p_quantico(ind)).sum()
    return classe_individuo(ind), float(tv)


def final(args):
    dom, sem = args
    _, melhor, _ = S.ga("quantico", dom, sem, pop_n=50, ger=30)
    return dom, analisa(melhor)


def resumo(rot, rs):
    linhas = [f"{rot}: {len(rs)} individuos"]
    for c in ("corolario", "so_T1", "fora"):
        tvs = np.array([tv for k, tv in rs if k == c])
        if len(tvs):
            linhas.append(f"  {c:9s}: {len(tvs):4d} ({len(tvs) / len(rs):.0%}) | TV max {tvs.max():.1e} | "
                          f"TV mediana {np.median(tvs):.1e} | TV > 1e-12 em {(tvs > 1e-12).mean():.0%}")
    return linhas


def main():
    rng = np.random.default_rng(SEMENTE)
    iniciais = [S.individuo(rng, S.G_QU) for _ in range(N_INI)]
    with Pool(16) as pool:
        r_ini = pool.map(analisa, iniciais, chunksize=10)
        r_fin = pool.map(final, [(d, SEMENTE + 1000 * k + r) for k, d in enumerate(DOMINIOS) for r in range(REPS)], chunksize=1)
    L = [f"Gemeo classico exato (corolario do Teorema 1) | semente {SEMENTE} | {date.today()}",
         "previsao: TV(gemeo, quantico) = 0 (ate arredondamento) em todo individuo da classe 'corolario'", ""]
    L += resumo("iniciais sorteados (G_qu, v1)", r_ini)
    for d in DOMINIOS:
        L += resumo(f"melhores finais, |P|=50, G=30, dominio {d}", [r for dd, r in r_fin if dd == d])
    ok = all(tv < 1e-12 for k, tv in r_ini + [r for _, r in r_fin] if k == "corolario")
    L.append(f"\nprevisao confirmada (TV < 1e-12 em todo 'corolario'): {ok}")
    print("\n".join(L))
    (RAIZ / "resultados" / "qga" / f"gemeo_exato-{date.today()}.txt").write_text("\n".join(L) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
