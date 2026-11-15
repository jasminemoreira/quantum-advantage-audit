"""When the classical stochastic analog is EXACTLY equal to the quantum circuit (14 Nov 2026).

Corollary of Theorem 1 (provas-2026-11-14.md): a Souza individual (computational-basis initial
state, layers of G_qu) whose circuit obeys rules R0, R1, R2 and in which every H gate acts on a qubit
that, at that moment, is in a basis state (fresh with no non-monomial gate since the start, or a
classical bit re-drawn by rule R2) has the same measured distribution as the classical analog
(H -> XOR with a fair coin, Y -> X, phases discarded).

Classes of each individual:
  corolario : obeys the corollary -> prediction TV(analog, quantum) = 0;
  so_T1     : obeys Theorem 1 (classical sampling), but some H acts on a fresh qubit already in
              superposition -> the analog may differ;
  fora      : does not obey Theorem 1 (interference may occur).
Check: exact distribution of the analog (probability propagation) against p_quantico (statevector).

Usage (WSL): externos/QFIE/.venv/bin/python scripts/qga/gemeo_exato.py
Output: resultados/qga/gemeo_exato-<date>.txt
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
    """Applies the rules of Theorem 1 (with R2 and the witness lock) plus the corollary condition."""
    n = S.NQ
    estado = {q: "base" for q in range(n)}  # base | super (fresh) | classico (classical)
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
                else:  # classical: needs a witness (R2)
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
    """Exact distribution of the classical analog."""
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
            else:  # involutive permutations: p_new = p o pi
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
    linhas = [f"{rot}: {len(rs)} individuals"]
    for c in ("corolario", "so_T1", "fora"):
        tvs = np.array([tv for k, tv in rs if k == c])
        if len(tvs):
            linhas.append(f"  {c:9s}: {len(tvs):4d} ({len(tvs) / len(rs):.0%}) | TV max {tvs.max():.1e} | "
                          f"TV median {np.median(tvs):.1e} | TV > 1e-12 in {(tvs > 1e-12).mean():.0%}")
    return linhas


def main():
    rng = np.random.default_rng(SEMENTE)
    iniciais = [S.individuo(rng, S.G_QU) for _ in range(N_INI)]
    with Pool(16) as pool:
        r_ini = pool.map(analisa, iniciais, chunksize=10)
        r_fin = pool.map(final, [(d, SEMENTE + 1000 * k + r) for k, d in enumerate(DOMINIOS) for r in range(REPS)], chunksize=1)
    L = [f"Exact classical analog (corollary of Theorem 1) | seed {SEMENTE} | {date.today()}",
         "prediction: TV(analog, quantum) = 0 (up to rounding) in every individual of class 'corolario'", ""]
    L += resumo("drawn initial individuals (G_qu, v1)", r_ini)
    for d in DOMINIOS:
        L += resumo(f"best final individuals, |P|=50, G=30, domain {d}", [r for dd, r in r_fin if dd == d])
    ok = all(tv < 1e-12 for k, tv in r_ini + [r for _, r in r_fin] if k == "corolario")
    L.append(f"\nprediction confirmed (TV < 1e-12 in every 'corolario'): {ok}")
    print("\n".join(L))
    (RAIZ / "resultados" / "qga" / f"gemeo_exato-{date.today()}.txt").write_text("\n".join(L) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
