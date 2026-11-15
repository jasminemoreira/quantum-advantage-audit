"""Analise de sensibilidade da reimplementacao de Souza/Dardenne/Portugal (13/11/2026).

Cada escolha nao fixada pelo texto (ver souza_v1.py, OPC) e trocada por uma alternativa, uma de
cada vez, mais uma variante com todas as alternativas juntas. Para cada variante, as previsoes
que sustentam o argumento do artigo:
  P1 dominio centrado, g0: quantico melhor que classico (vies de centro ajuda);
  P2 dominio deslocado, g0: quantico pior que classico (vies de centro atrapalha);
  P3 dominio deslocado, g30: quantico nao supera o classico;
  P4 gemeo classico nao e pior que o quantico (g0 e g30, nos dois dominios);
  P5 emaranhamento "bell" entre individuos ajuda no centrado e atrapalha no deslocado (g30);
  P6 o equivalente classico do "bell" reproduz o quantico+bell (g30);
  P7 emaranhamento dentro do melhor individuo final: mais frequente no centrado que no deslocado.
Parte A: |P| = 50, 30 geracoes, 50 repeticoes. Parte B: |P| = 10, 30 geracoes, 100 repeticoes.
Posto de Schmidt entre os registros calculado em ponto flutuante (SVD, corte 1e-8); a versao
exata ja foi feita para o padrao em certifica_souza_v1.py.

Uso (WSL): externos/QFIE/.venv/bin/python scripts/qga/sensibilidade_souza.py [variante ...]
Saida: resultados/qga/sensibilidade_souza-<data>.txt
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
import vies_centro as VC  # noqa: E402

RAIZ = AQUI.parents[1]
SEMENTE = 20261117
DOMINIOS = ((-5.12, 5.12), (-2.12, 8.12))
GER = 30
REPS_A, REPS_B = 50, 100

VARIANTES = {
    "padrao": {},
    "prof3": dict(prof=3),
    "prof8": dict(prof=8),
    "roleta": dict(selecao="roleta"),
    "regiao_global": dict(regiao="global"),
    "descarta_fica": dict(colisao="descarta_fica"),
    "sorteio_aridade": dict(sorteio="por_aridade"),
    "h_congelada": dict(h_muta=False),
    "base_cruza": dict(base="cruza"),
    "todas_alt": dict(selecao="roleta", regiao="global", colisao="descarta_fica",
                      sorteio="por_aridade", h_muta=False, base="cruza"),
}


def posto_registros(ind):
    psi = VC.estado(S.sem_i(ind)).reshape(1 << S.NB, 1 << S.NB)  # linhas: registro 1; colunas: registro 0
    s = np.linalg.svd(psi, compute_uv=False)
    return int((s > 1e-8 * s[0]).sum())


def tarefa(args):
    var, parte, modo, emar, dom, pop_n, sem = args
    S.OPC.clear()
    S.OPC.update(S.PADRAO, **VARIANTES[var])
    curva, melhor, _ = S.ga(modo, dom, sem, pop_n=pop_n, ger=GER, emar=emar)
    posto = posto_registros(melhor) if (parte == "A" and modo == "quantico") else -1
    return var, parte, modo, emar, dom, curva, posto


def dif(res, a, b, g):
    """media(a) - media(b) na geracao g, com erro-padrao."""
    ca = np.array([r[5][g] for r in res if (r[1], r[2], r[3], r[4]) == a])
    cb = np.array([r[5][g] for r in res if (r[1], r[2], r[3], r[4]) == b])
    return ca.mean() - cb.mean(), np.sqrt(ca.var(ddof=1) / len(ca) + cb.var(ddof=1) / len(cb))


def fmt(d):
    return f"{d[0]:+.2f} ({d[1]:.2f})"


def relatorio(var, res):
    C, D = DOMINIOS
    q, gm, cl = ("A", "quantico", None), ("A", "gemeo", None), ("A", "classico", None)
    qb, qe, ge = ("B", "quantico", None), ("B", "quantico", "bell"), ("B", "gemeo", "bell")
    t = lambda x, dom: x + (dom,)  # noqa: E731
    p1 = dif(res, t(q, C), t(cl, C), 0)
    p2 = dif(res, t(q, D), t(cl, D), 0)
    p3 = dif(res, t(q, D), t(cl, D), GER)
    p4 = [dif(res, t(gm, dom), t(q, dom), g) for dom in DOMINIOS for g in (0, GER)]
    p5c, p5d = dif(res, t(qe, C), t(qb, C), GER), dif(res, t(qe, D), t(qb, D), GER)
    p6 = [dif(res, t(ge, dom), t(qe, dom), GER) for dom in DOMINIOS]
    postos = {dom: np.array([r[6] for r in res if r[1] == "A" and r[2] == "quantico" and r[4] == dom]) for dom in DOMINIOS}
    sep = {dom: (postos[dom] == 1).mean() for dom in DOMINIOS}
    ok = dict(
        P1=p1[0] + 2 * p1[1] < 0,
        P2=p2[0] - 2 * p2[1] > 0,
        P3=p3[0] + 2 * p3[1] > 0,  # quantico nao significativamente melhor
        P4=all(d[0] - 2 * d[1] < 0 for d in p4),  # gemeo nao significativamente pior
        P5=p5c[0] + 2 * p5c[1] < 0 and p5d[0] - 2 * p5d[1] > 0,
        P6=all(abs(d[0]) < 2.5 * d[1] or abs(d[0]) < 0.5 for d in p6),
        P7=sep[C] < sep[D],
    )
    linhas = [f"[{var}] {VARIANTES[var] or 'escolhas declaradas'}",
              f"  P1 centrado g0, quantico - classico: {fmt(p1)}",
              f"  P2 deslocado g0, quantico - classico: {fmt(p2)}",
              f"  P3 deslocado g30, quantico - classico: {fmt(p3)}",
              "  P4 gemeo - quantico (centrado g0, g30; deslocado g0, g30): " + ", ".join(fmt(d) for d in p4),
              f"  P5 bell - sem (g30): centrado {fmt(p5c)}, deslocado {fmt(p5d)}",
              "  P6 gemeo+bell - quantico+bell (g30): centrado " + fmt(p6[0]) + ", deslocado " + fmt(p6[1]),
              f"  P7 melhor final separavel entre registros: centrado {sep[C]:.0%} (posto mediano {np.median(postos[C]):.0f}), "
              f"deslocado {sep[D]:.0%} (posto mediano {np.median(postos[D]):.0f})",
              "  confirmadas: " + " ".join(f"{k}={'sim' if v else 'NAO'}" for k, v in ok.items())]
    return linhas, ok


def main():
    t0 = time.time()
    nomes = sys.argv[1:] or list(VARIANTES)
    tarefas = []
    for v, var in enumerate(nomes):
        for k, dom in enumerate(DOMINIOS):
            for j, modo in enumerate(("quantico", "gemeo", "classico")):
                tarefas += [(var, "A", modo, None, dom, 50, SEMENTE + 10 ** 6 * v + 10000 * k + 1000 * j + r) for r in range(REPS_A)]
            for j, (modo, emar) in enumerate((("quantico", None), ("quantico", "bell"), ("gemeo", "bell"))):
                tarefas += [(var, "B", modo, emar, dom, 10, SEMENTE + 10 ** 6 * v + 500000 + 10000 * k + 1000 * j + r) for r in range(REPS_B)]
    # tarefas caras primeiro, para equilibrar a fila
    tarefas.sort(key=lambda a: -(a[5] * (3 if a[2] == "quantico" else 1) * (2 if a[3] else 1) * (8 if a[0] == "prof8" else 1)))
    with Pool(16) as pool:
        res = pool.map(tarefa, tarefas, chunksize=1)
    linhas = [f"Sensibilidade da reimplementacao Souza/Dardenne/Portugal | semente {SEMENTE} | {date.today()}",
              f"Parte A |P|=50, {REPS_A} rep.; Parte B |P|=10, {REPS_B} rep.; {GER} geracoes; diferencas: media (erro-padrao)",
              "criterio: P1, P2, P5 com |dif| > 2 EP no sentido previsto; P3, P4 sem diferenca > 2 EP contra o argumento; "
              "P6 com |dif| < 2,5 EP ou < 0,5; P7 fracao separavel menor no centrado", ""]
    resumo = []
    for var in nomes:
        ls, ok = relatorio(var, [r for r in res if r[0] == var])
        linhas += ls + [""]
        resumo.append(f"{var:16s} " + " ".join("sim" if ok[k] else "NAO" for k in ok))
    linhas += ["resumo (P1 a P7):"] + resumo + [f"\ntempo {time.time() - t0:.0f}s"]
    print("\n".join(linhas))
    (RAIZ / "resultados" / "qga" / f"sensibilidade_souza-{date.today()}.txt").write_text("\n".join(linhas) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
