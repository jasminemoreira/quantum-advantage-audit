"""Reproduction of the worked example of Udrescu, Prodan and Vladutiu 2006 (CF'06, doi 10.1145/1128022.1128034).

The paper (Eqs. 15 to 17) starts from 16 individuals of 4 qubits, each paired with its own fitness
in a 10-qubit register, |psi> = (1/4) sum_u |u>|f(u)>, applies the oracle (negative sign on
individuals 6 and 7, Eq. 16) and Grover "over the fitness register (rightmost 10 qubits)". It states that
the result (Eq. 17) has a6, a7 with |a6|^2 + |a7|^2 ~ 1 and the others ~ 0.

Here: oracle = phase -1 on the marked fitness values (those of u = 6 and 7 in Eq. 16); diffuser
2|s><s| - I on the fitness register (10 qubits). Reports, per iteration, the marginal of the individuals and the
probability of measuring a marked fitness. Prediction (Theorem 2 of provas-2026-11-14.md): marginal
of the individuals uniform at every iteration.

Usage: python scripts/qga/rqga2006_exemplo.py
Output: resultados/qga/rqga2006_exemplo-<data>.txt
"""
from datetime import date
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
F = ["1000000000", "1000011110", "1000110010", "1001010000", "1001100100", "1010000010", "1010010110",
     "1010110100", "1000101000", "1001000110", "1001011010", "0110011011", "0110101111", "0111001101",
     "0111100001", "0111111111"]  # fitness register in Eq. 16, individuals 0000 to 1111
FV = [int(b, 2) for b in F]
NF = 1024
MARC = {FV[6], FV[7]}

psi = np.zeros((16, NF))
for u in range(16):
    psi[u, FV[u]] = 0.25
O = np.array([-1.0 if y in MARC else 1.0 for y in range(NF)])
s = np.ones(NF) / np.sqrt(NF)
L = [f"Example of Udrescu et al. 2006 (Eqs. 15-17), Grover on the fitness register only | {date.today()}",
     "iter | P(individual) min, max | P(individual in {6,7}) | P(marked fitness)"]
for k in range(0, 13):
    if k:
        psi = psi * O
        psi = 2 * np.outer(psi @ s, s) - psi
    pind = (psi ** 2).sum(1)
    pfit = (psi ** 2).sum(0)
    L.append(f"{k:2d} | {pind.min():.6f}, {pind.max():.6f} | {pind[6] + pind[7]:.4f} | {sum(pfit[y] for y in MARC):.4f}")
L.append("paper (Eq. 17): |a6|^2 + |a7|^2 ~ 1 after the Grover iterations")
print("\n".join(L))
(RAIZ / "resultados" / "qga" / f"rqga2006_exemplo-{date.today()}.txt").write_text("\n".join(L) + "\n")
