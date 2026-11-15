"""Certificado exato (anel Z[zeta_16]) do exemplo resolvido de Udrescu, Prodan e Vladutiu 2006 (14/11/2026).

Mesmo exemplo de rqga2006_exemplo.py, agora como circuito de portas no motor exato esparso:
  - 4 qubits de individuo (q0..q3, u = sum bit_i 2^i) e 10 de aptidao (q4..q13, valor y = sum bit_j 2^j);
  - preparo: H nos individuos; para cada u, MCX controlados pelo padrao de u escrevem f(u) (Eq. 16);
  - oraculo: fase -1 (zeta^8) nos dois valores de aptidao marcados (sinal negativo na Eq. 16);
  - difusor so no registro de aptidao: H^10 X^10 CZ_10 X^10 H^10 (= -D, sinal global irrelevante).
Para k = 0..12 iteracoes: decide EXATAMENTE que a marginal dos individuos e 1/16 para cada u e
calcula EXATAMENTE P(aptidao marcada) (numerador em Z[zeta_16] / 2^k), comparando com o float e com
a cota do Lema 2.

Uso (WSL): externos/mqtbench/.venv/bin/python scripts/qga/rqga2006_exato.py
Saida: resultados/qga/rqga2006_exato-<data>.txt
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
# exemplo 2 (Eqs. 15-17): marcados os individuos 6 e 7; exemplo 1 (Eqs. 12-14): 4, 5, 6, 7 e 10
# (sinais negativos na Eq. 13; os valores de aptidao das Eqs. 13 e 16 sao os mesmos).
EXEMPLO = int(sys.argv[sys.argv.index("--exemplo") + 1]) if "--exemplo" in sys.argv else 2
MARC = [FV[u] for u in ((4, 5, 6, 7, 10) if EXEMPLO == 1 else (6, 7))]
IND = list(range(4))
FIT = list(range(4, 14))


def padrao(m, qubits, valor, acao):
    """Aplica X nos qubits cujo bit em `valor` e 0, executa acao(), desfaz os X."""
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
    oraculo(m)  # estado da Eq. 16
    N, nm = 1024, len(MARC)
    L = [f"Exemplo de Udrescu et al. 2006 no motor exato Z[zeta_16] | exemplo {EXEMPLO} (marcados: {len(MARC)}) | {date.today()}",
         "k | marginal dos individuos exatamente 1/16? | P(aptidao marcada) exata (decimal) | cota do Lema 2 | P exata como numerador / 2^k"]
    p0 = None
    tudo = True
    for k in range(0, 13):
        if k:
            difusor(m)
            oraculo(m)  # iteracao seguinte: difusor, depois oraculo (G = D O aplicado a partir de O|psi0>)
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
        L.append(f"{k:2d} | {unif} | {pm:.6f} | {cota:.6f} | numerador {tuple(num)} / 2^{m.k}")
    L.append(f"\nP_0(M) exato = {pexato0}; marginal dos individuos uniforme e cota respeitada em todo k: {tudo}")
    L.append(f"suporte maximo {m.max_suporte}")
    print("\n".join(L))
    (RAIZ / "resultados" / "qga" / (f"rqga2006_exato-{date.today()}.txt" if EXEMPLO == 2 else f"rqga2006_exato_ex1-{date.today()}.txt")).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
