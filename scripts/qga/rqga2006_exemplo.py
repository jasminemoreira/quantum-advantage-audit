"""Reproducao do exemplo resolvido de Udrescu, Prodan e Vladutiu 2006 (CF'06, doi 10.1145/1128022.1128034).

O artigo (Eqs. 15 a 17) parte de 16 individuos de 4 qubits, cada um emparelhado com a propria aptidao
num registro de 10 qubits, |psi> = (1/4) sum_u |u>|f(u)>, aplica o oraculo (sinal negativo nos
individuos 6 e 7, Eq. 16) e o Grover "over the fitness register (rightmost 10 qubits)". Afirma que
o resultado (Eq. 17) tem a6, a7 com |a6|^2 + |a7|^2 ~ 1 e os demais ~ 0.

Aqui: oraculo = fase -1 nos valores de aptidao marcados (os de u = 6 e 7 na Eq. 16); difusor
2|s><s| - I no registro de aptidao (10 qubits). Relata, por iteracao, a marginal dos individuos e a
probabilidade de medir uma aptidao marcada. Previsao (Teorema 2 de provas-2026-11-14.md): marginal
dos individuos uniforme em toda iteracao.

Uso: python scripts/qga/rqga2006_exemplo.py
Saida: resultados/qga/rqga2006_exemplo-<data>.txt
"""
from datetime import date
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
F = ["1000000000", "1000011110", "1000110010", "1001010000", "1001100100", "1010000010", "1010010110",
     "1010110100", "1000101000", "1001000110", "1001011010", "0110011011", "0110101111", "0111001101",
     "0111100001", "0111111111"]  # registro de aptidao na Eq. 16, individuos 0000 a 1111
FV = [int(b, 2) for b in F]
NF = 1024
MARC = {FV[6], FV[7]}

psi = np.zeros((16, NF))
for u in range(16):
    psi[u, FV[u]] = 0.25
O = np.array([-1.0 if y in MARC else 1.0 for y in range(NF)])
s = np.ones(NF) / np.sqrt(NF)
L = [f"Exemplo de Udrescu et al. 2006 (Eqs. 15-17), Grover so no registro de aptidao | {date.today()}",
     "iter | P(individuo) min, max | P(individuo em {6,7}) | P(aptidao marcada)"]
for k in range(0, 13):
    if k:
        psi = psi * O
        psi = 2 * np.outer(psi @ s, s) - psi
    pind = (psi ** 2).sum(1)
    pfit = (psi ** 2).sum(0)
    L.append(f"{k:2d} | {pind.min():.6f}, {pind.max():.6f} | {pind[6] + pind[7]:.4f} | {sum(pfit[y] for y in MARC):.4f}")
L.append("artigo (Eq. 17): |a6|^2 + |a7|^2 ~ 1 depois das iteracoes de Grover")
print("\n".join(L))
(RAIZ / "resultados" / "qga" / f"rqga2006_exemplo-{date.today()}.txt").write_text("\n".join(L) + "\n")
