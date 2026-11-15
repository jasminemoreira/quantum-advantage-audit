"""Experimentos com os operadores da v1 (souza_v1.py), 13/11/2026.

Parte A (refaz vies_centro.py): |P| = 50, 30 geracoes, 100 repeticoes; avaliadores quantico,
gemeo classico (H -> moeda, Y -> X, fases removidas) e classico (G_cl); dominios centrado
[-5.12, 5.12] (otimo no centro) e deslocado [-2.12, 8.12] (centro em x = 3, otimo em x = 0).

Parte B (emaranhamento entre individuos, secao 4.3 do artigo): |P| = 10 como na figura deles,
30 geracoes, 200 repeticoes; quantico sem emaranhamento, com "max" e com "bell" (ver souza_v1.py),
e o equivalente classico do "bell" (gemeo com estado inicial sorteado entre b e ~b a cada shot).
Previsao: "max" deixa a saida de todo individuo uniforme, entao x* cai no centro do dominio
para qualquer circuito; no centrado isso e otimo, no deslocado e f(3, 3) = 18.

Relata o melhor f avaliado por geracao (com ruido de shots, como o artigo) e, para o quantico,
f(E[x]) exato do melhor individuo final.

Uso (WSL): externos/QFIE/.venv/bin/python scripts/qga/exp_souza_v1.py
Saida: resultados/qga/exp_souza_v1-<data>.txt
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
import souza_v1 as S  # noqa: E402

RAIZ = AQUI.parents[1]
SEMENTE = 20261115
DOMINIOS = ((-5.12, 5.12), (-2.12, 8.12))
GER = 30
GENS = (0, 1, 2, 5, 10, 20, 30)


def tarefa(args):
    parte, modo, emar, dom, pop_n, sem = args
    curva, melhor, f = S.ga(modo, dom, sem, pop_n=pop_n, ger=GER, emar=emar)
    fx = S.aptidao_exata(melhor, dom, emar) if modo == "quantico" else float("nan")
    return parte, modo, emar, dom, curva, fx


def tabela(res, chaves, rotulo):
    linhas = [rotulo, "dominio | avaliador | " + " | ".join(f"g{g}" for g in GENS) + " | f(E[x]) do melhor final (mediana)"]
    for dom in DOMINIOS:
        for modo, emar in chaves:
            rs = [r for r in res if r[1] == modo and r[2] == emar and r[3] == dom]
            c = np.array([r[4] for r in rs])
            med, ep = c.mean(axis=0), c.std(axis=0, ddof=1) / np.sqrt(len(c))
            fx = np.array([r[5] for r in rs])
            fxs = f"{np.median(fx):.3f}" if modo == "quantico" else "-"
            nome = f"{modo}" + (f"+{emar}" if emar else "")
            linhas.append(f"{dom} | {nome:14s} | " + " | ".join(f"{med[g]:.3f} ({ep[g]:.3f})" for g in GENS) + f" | {fxs}")
    return linhas


def main():
    t0 = time.time()
    A = [("A", modo, None, dom, 50, SEMENTE + 10000 * k + 1000 * j + r)
         for k, dom in enumerate(DOMINIOS) for j, modo in enumerate(("quantico", "gemeo", "classico")) for r in range(100)]
    chB = [("quantico", None), ("quantico", "max"), ("quantico", "bell"), ("gemeo", "bell")]
    B = [("B", modo, emar, dom, 10, SEMENTE + 500000 + 10000 * k + 1000 * j + r)
         for k, dom in enumerate(DOMINIOS) for j, (modo, emar) in enumerate(chB) for r in range(200)]
    with Pool(16) as pool:
        res = pool.map(tarefa, A + B, chunksize=2)
    linhas = [f"Souza/Dardenne/Portugal, operadores da v1 (souza_v1.py) | semente {SEMENTE} | {date.today()}",
              f"m=2 n=8 shots=1024 d=5 G={GER}; melhor f avaliado por geracao, media (erro-padrao)", ""]
    linhas += tabela([r for r in res if r[0] == "A"], [("quantico", None), ("gemeo", None), ("classico", None)],
                     "Parte A: |P|=50, 100 repeticoes")
    linhas.append("")
    linhas += tabela([r for r in res if r[0] == "B"], chB, "Parte B: emaranhamento entre individuos, |P|=10, 200 repeticoes")
    # diferencas da parte B na geracao 5 e 30 (com - sem), com erro-padrao
    linhas.append("")
    linhas.append("Parte B, diferenca (com emaranhamento - sem), media (EP), nas geracoes 5 e 30")
    for dom in DOMINIOS:
        base = np.array([r[4] for r in res if r[0] == "B" and r[1] == "quantico" and r[2] is None and r[3] == dom])
        for modo, emar in chB[1:]:
            c = np.array([r[4] for r in res if r[0] == "B" and r[1] == modo and r[2] == emar and r[3] == dom])
            partes = []
            for g in (5, 30):
                d = c[:, g].mean() - base[:, g].mean()
                ep = np.sqrt(c[:, g].var(ddof=1) / len(c) + base[:, g].var(ddof=1) / len(base))
                partes.append(f"g{g}: {d:+.3f} ({ep:.3f})")
            linhas.append(f"{dom} {modo}+{emar} | " + " | ".join(partes))
    linhas.append(f"\ntempo {time.time() - t0:.0f}s")
    print("\n".join(linhas))
    (RAIZ / "resultados" / "qga" / f"exp_souza_v1-{date.today()}.txt").write_text("\n".join(linhas) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
