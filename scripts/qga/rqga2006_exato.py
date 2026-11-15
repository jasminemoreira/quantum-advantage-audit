"""Exact certificate (ring Z[zeta_16]) of the worked example of Udrescu, Prodan and Vladutiu 2006 (14 Nov 2026).

Same example as rqga2006_exemplo.py, now as a gate circuit on the exact sparse engine:
  - 4 individual qubits (q0..q3, u = sum bit_i 2^i) and 10 fitness qubits (q4..q13, value y = sum bit_j 2^j);
  - preparation: H on the individuals; for each u, MCX gates controlled by the bit pattern of u write f(u) (Eq. 16);
  - oracle: phase -1 (zeta^8) on the two marked fitness values (negative sign in Eq. 16);
  - diffuser on the fitness register only: H^10 X^10 CZ_10 X^10 H^10 (= -D, global sign irrelevant).
For k = 0..12 iterations: decides EXACTLY that the marginal of the individuals is 1/16 for each u and
computes EXACTLY P(marked fitness) (numerator in Z[zeta_16] / 2^k), comparing it with the float value and
with the bound of Lemma 2.

Usage (WSL): externos/mqtbench/.venv/bin/python scripts/qga/rqga2006_exato.py
Output: resultados/qga/rqga2006_exato-<data>.txt
"""
from __future__ import annotations

import math
import sys
from datetime import date
from fractions import Fraction
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "qcalc"))
from esparso import MotorEsparso  # noqa: E402

RAIZ = AQUI.parents[1]
F = ["1000000000", "1000011110", "1000110010", "1001010000", "1001100100", "1010000010", "1010010110",
     "1010110100", "1000101000", "1001000110", "1001011010", "0110011011", "0110101111", "0111001101",
     "0111100001", "0111111111"]
FV = [int(b, 2) for b in F]
# example 2 (Eqs. 15-17): individuals 6 and 7 marked; example 1 (Eqs. 12-14): 4, 5, 6, 7 and 10
# (negative signs in Eq. 13; the fitness values of Eqs. 13 and 16 are the same).
EXEMPLO = int(sys.argv[sys.argv.index("--exemplo") + 1]) if "--exemplo" in sys.argv else 2
MARC = [FV[u] for u in ((4, 5, 6, 7, 10) if EXEMPLO == 1 else (6, 7))]
IND = list(range(4))
FIT = list(range(4, 14))


def padrao(m, qubits, valor, acao):
    """Applies X to the qubits whose bit in `valor` is 0, runs acao(), undoes the X gates."""
    zeros = [q for i, q in enumerate(qubits) if not (valor >> i) & 1]
    for q in zeros:
        m.aplica(("mcx", [], q))
    acao()
    for q in zeros:
        m.aplica(("mcx", [], q))


def oraculo(m):
    for y in MARC:
        padrao(m, FIT, y, lambda: m.aplica(("fase", 8, FIT)))


def difusor(m):
    for q in FIT:
        m.aplica(("h", q))
    padrao(m, FIT, 0, lambda: m.aplica(("fase", 8, FIT)))
    for q in FIT:
        m.aplica(("h", q))


def decimal(num, k):
    return (num[0] + sum(num[j] * math.cos(j * math.pi / 8) for j in range(1, 8))) / 2 ** k


def main():
    m = MotorEsparso(14)
    for q in IND:
        m.aplica(("h", q))
    for u, fu in enumerate(FV):
        for j in range(10):
            if (fu >> j) & 1:
                padrao(m, IND, u, lambda j=j: m.aplica(("mcx", IND, FIT[j])))
    oraculo(m)  # state of Eq. 16
    N, nm = 1024, len(MARC)
    L = [f"Example of Udrescu et al. 2006 on the exact engine Z[zeta_16] | example {EXEMPLO} (marked: {len(MARC)}) | {date.today()}",
         "k | marginal of the individuals exactly 1/16? | exact P(marked fitness) (decimal) | bound of Lemma 2 | exact P as numerator / 2^k"]
    p0 = None
    tudo = True
    for k in range(0, 13):
        if k:
            difusor(m)
            oraculo(m)  # next iteration: diffuser, then oracle (G = D O applied starting from O|psi0>)
        marg = m.marginal(IND)
        alvo = (2 ** m.k,) + (0,) * 7
        unif = len(marg) == 16 and all(tuple(16 * a for a in v) == alvo for v in marg.values())
        mf = m.marginal(FIT)
        num = [0] * 8
        for y in MARC:
            for i, a in enumerate(mf.get(y, (0,) * 8)):
                num[i] += a
        pm = decimal(num, m.k)
        if p0 is None:
            p0 = pm
            pexato0 = Fraction(num[0], 2 ** m.k) if not any(num[1:]) else None
        cota = p0 + (1 - p0) / (N - nm)
        tudo &= unif and pm <= cota + 1e-12
        L.append(f"{k:2d} | {unif} | {pm:.6f} | {cota:.6f} | numerator {tuple(num)} / 2^{m.k}")
    L.append(f"\nexact P_0(M) = {pexato0}; marginal of the individuals uniform and bound respected for every k: {tudo}")
    L.append(f"maximum support {m.max_suporte}")
    print("\n".join(L))
    (RAIZ / "resultados" / "qga" / (f"rqga2006_exato-{date.today()}.txt" if EXEMPLO == 2 else f"rqga2006_exato_ex1-{date.today()}.txt")).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
