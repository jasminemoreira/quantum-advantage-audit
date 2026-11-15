"""EXACT decision of 'classical analog = quantum circuit', individual by individual (14 Nov 2026).

Same individuals as analog_exact.py (same seed: 300 drawn initial individuals and the best final
individuals of 30 GAs per domain). For each one:
  - exact quantum distribution in the sparse engine over Z[zeta_16]: P(x) = num(x) / 2^k;
  - exact distribution of the analog in dyadic rationals: P(x) = w(x) / 2^h, h = number of H gates;
  - equality decided without floating point: 2^h * num(x) == w(x) * 2^k * (1, 0, ..., 0) for every x.
Reports, per corollary class (corolario, so_T1, fora), how many are exactly equal and checks against
the floating-point TV of analog_exact.py.

Usage: python scripts/qga/analog_exact_ring.py
Output: results/qga/analog_exact_ring-<date>.txt
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
import certify_souza as CS  # noqa: E402
import analog_exact as GE  # noqa: E402
import souza_v1 as S  # noqa: E402

RAIZ = AQUI.parents[1]
IDX = np.arange(S.DIM)


def gemeo_inteiro(ind):
    """Integer weights w(x) and exponent h with P_analog(x) = w(x) / 2^h."""
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
    L = [f"{rot}: {len(rs)} individuals | exactly equal to the analog: {sum(r[1] for r in rs)} "
         f"({sum(r[1] for r in rs) / len(rs):.0%})"]
    for c in ("corolario", "so_T1", "fora"):
        sub = [r for r in rs if r[0] == c]
        if sub:
            ig = sum(r[1] for r in sub)
            disc = sum((r[1] != (r[2] < 1e-12)) for r in sub)
            L.append(f"  {c:9s}: {len(sub):4d} | equal (exact) {ig:4d} ({ig / len(sub):.0%}) | "
                     f"disagreements with float TV<1e-12: {disc}")
    return L


def main():
    t0 = time.time()
    rng = np.random.default_rng(GE.SEMENTE)
    iniciais = [S.individuo(rng, S.G_QU) for _ in range(GE.N_INI)]
    with Pool(16) as pool:
        r_ini = pool.map(decide, iniciais, chunksize=4)
        r_fin = pool.map(final, [(d, GE.SEMENTE + 1000 * k + r) for k, d in enumerate(GE.DOMINIOS)
                                 for r in range(GE.REPS)], chunksize=1)
    L = [f"Analog = quantum decided in the ring Z[zeta_16] | seed {GE.SEMENTE} (that of analog_exact.py) | {date.today()}",
         "exact equality of the full measured distributions (16 qubits); classes of Corollary 1", ""]
    L += resumo("drawn initial individuals (G_qu, v1)", r_ini)
    fins = []
    for d in GE.DOMINIOS:
        rs = [r for dd, r in r_fin if dd == d]
        fins += rs
        L += resumo(f"best final individuals, |P|=50, G=30, domain {d}", rs)
    ok_cor = all(r[1] for r in r_ini + fins if r[0] == "corolario")
    disc = sum((r[1] != (r[2] < 1e-12)) for r in r_ini + fins)
    L.append(f"\nCorollary 1 confirmed in exact arithmetic (every 'corolario' equal): {ok_cor}")
    L.append(f"disagreements between the exact check and the float criterion TV < 1e-12: {disc}")
    L.append(f"time {time.time() - t0:.0f}s; longest time per individual {max(r[3] for r in r_ini + fins):.1f}s")
    print("\n".join(L))
    (RAIZ / "results" / "qga" / f"analog_exact_ring-{date.today()}.txt").write_text("\n".join(L) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
