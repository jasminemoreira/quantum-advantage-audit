"""Certificacao exata da reducao do emaranhamento entre individuos (13/11/2026).

Afirmacao (souza_v1.py): se dois individuos A e B comecam num estado emaranhado |Phi> e cada um
evolui pelo proprio circuito (U_A x U_B), a distribuicao medida em A e a de U_A rho_A0 U_A^dagger,
com rho_A0 = Tr_B |Phi><Phi|, e nao depende de U_B. Entao o individuo emaranhado equivale a um
individuo com estado inicial classico aleatorio.

Teste, no motor exato esparso em Z[zeta_16], com individuos de um registro (m = 1, n = 8; o par
tem 16 qubits, como no artigo para funcoes de uma variavel), circuitos de G_qu sorteados e
mutados pelos operadores da v1:
  "max" : |Phi> = H^8 em A, CX(A_j -> B_j); previsao: marginal de A exatamente uniforme;
  "bell": |Phi> = (|b_A b_B> + |~b_A ~b_B>)/sqrt(2); previsao: marginal de A exatamente igual a
          (P_{U_A |b_A>} + P_{U_A |~b_A>})/2, calculada em duas execucoes separadas de 8 qubits.
Igualdade decidida no anel (numerador em Z[zeta], expoente k), sem ponto flutuante.

Uso (WSL): externos/QFIE/.venv/bin/python scripts/qga/certifica_pares.py
Saida: resultados/qga/certifica_pares-<data>.txt
"""
from __future__ import annotations

import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "qcalc"))
import certifica_souza as CS  # noqa: E402
import souza_v1 as S  # noqa: E402
from esparso import MotorEsparso  # noqa: E402

RAIZ = AQUI.parents[1]
SEMENTE = 20261114
NB = 8
CASOS = 30


def desloca(portas, d):
    out = []
    for p in portas:
        if p[0] == "h":
            out.append(("h", p[1] + d))
        elif p[0] == "fase":
            out.append(("fase", p[1], [q + d for q in p[2]]))
        else:
            out.append(("mcx", [c + d for c in p[1]], p[2] + d))
    return out


def portas_ind(ind, d=0, com_base=True):
    s = S.sem_i(ind)
    if not com_base:
        s = {"base": 0, "cams": s["cams"]}
    return desloca(CS.portas_exatas(s), d)


def roda(n, portas):
    m = MotorEsparso(n)
    for p in portas:
        m.aplica(p)
    return m


def iguais(n1, k1, n2, k2):
    """n1/2^k1 == n2/2^k2 no anel."""
    return tuple(a * 2 ** k2 for a in n1) == tuple(b * 2 ** k1 for b in n2)


def caso(args):
    modo, semente = args
    rng = np.random.default_rng(semente)
    A = S.muta(rng, S.individuo(rng, S.G_QU, nq=NB), S.G_QU)
    B = S.muta(rng, S.individuo(rng, S.G_QU, nq=NB), S.G_QU)
    # os individuos de um registro precisam de mutacao com nq=8: as camadas ja sao de 8 qubits
    if modo == "max":
        prep = [("h", q) for q in range(NB)] + [("mcx", [q], q + NB) for q in range(NB)]
        par = prep + portas_ind(A, 0, False) + portas_ind(B, NB, False)
        m = roda(2 * NB, par)
        marg = m.marginal(list(range(NB)))
        alvo = (2 ** m.k,) + (0,) * 7
        ok = len(marg) == 2 ** NB and all(tuple(2 ** NB * a for a in v) == alvo for v in marg.values())
        ing = roda(NB, portas_ind(A))
        gi = ing.marginal(list(range(NB)))
        difere = not all(iguais(marg.get(y, (0,) * 8), m.k, gi.get(y, (0,) * 8), ing.k) for y in set(marg) | set(gi))
        return modo, ok, m.k, m.max_suporte, difere
    bA, bB = A["base"], B["base"]
    prep = [("h", 0)] + [("mcx", [0], q) for q in range(1, 2 * NB)]
    prep += [("mcx", [], q) for q in range(NB) if (bA >> q) & 1]
    prep += [("mcx", [], q + NB) for q in range(NB) if (bB >> q) & 1]
    par = prep + portas_ind(A, 0, False) + portas_ind(B, NB, False)
    m = roda(2 * NB, par)
    marg = m.marginal(list(range(NB)))
    todos = (1 << NB) - 1
    m1 = roda(NB, portas_ind({"base": bA, "cams": A["cams"]}))
    m2 = roda(NB, portas_ind({"base": bA ^ todos, "cams": A["cams"]}))
    g1, g2 = m1.marginal(list(range(NB))), m2.marginal(list(range(NB)))
    ok = True
    zero = (0,) * 8
    for y in set(marg) | set(g1) | set(g2):
        # previsao (n1/2^k1 + n2/2^k2)/2 = (n1 2^k2 + n2 2^k1) / 2^(k1+k2+1)
        n1, n2 = g1.get(y, zero), g2.get(y, zero)
        prev = tuple(a * 2 ** m2.k + b * 2 ** m1.k for a, b in zip(n1, n2))
        ok &= iguais(marg.get(y, zero), m.k, prev, m1.k + m2.k + 1)
    # controle: a previsao ingenua (A sozinho, sem emaranhamento) deveria falhar em geral
    difere = not all(iguais(marg.get(y, zero), m.k, g1.get(y, zero), m1.k) for y in set(marg) | set(g1))
    return modo, ok, m.k, m.max_suporte, difere


def main():
    tarefas = [(modo, SEMENTE + 1000 * i + j) for j, modo in enumerate(("max", "bell")) for i in range(CASOS)]
    with Pool(16) as pool:
        res = pool.map(caso, tarefas, chunksize=1)
    linhas = [f"Reducao do emaranhamento entre individuos, conferida no anel | semente {SEMENTE} | {date.today()}",
              f"individuos de 1 registro (n={NB}), par de {2 * NB} qubits, G_qu com operadores da v1, {CASOS} pares por modo"]
    for modo in ("max", "bell"):
        rs = [r for r in res if r[0] == modo]
        linhas.append(f"{modo}: marginal de A == previsao exata em {sum(r[1] for r in rs)}/{len(rs)} | "
                      f"k max {max(r[2] for r in rs)} | suporte max {max(r[3] for r in rs)} | "
                      f"controle (previsao sem emaranhamento falha) em {sum(r[4] for r in rs)}/{len(rs)}")
    print("\n".join(linhas))
    (RAIZ / "resultados" / "qga" / f"certifica_pares-{date.today()}.txt").write_text("\n".join(linhas) + "\n")
    return 0 if all(r[1] for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
