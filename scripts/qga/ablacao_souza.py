"""Ablacao do caso Souza, Dardenne e Portugal pelo protocolo fixado em
resultados/qga/ablacao_protocolo-2026-11-14.md (14/11/2026).

Uso (WSL): externos/QFIE/.venv/bin/python scripts/qga/ablacao_souza.py [--reps N]
Saidas: resultados/qga/ablacao_souza-<data>.txt (relatorio) e .npz (curvas).
"""
from __future__ import annotations

import sys
import time
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import stats

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import souza_v1 as S  # noqa: E402

RAIZ = AQUI.parents[1]
SEMENTE = 20261123
GER = 30
FUNCS = [("rastrigin", 10.24), ("esfera", 10.24), ("ackley", 65.536), ("griewank", 1200.0), ("rosenbrock", 4.096)]
ALFAS = [0.50, 0.355, 0.26, 0.16]
MODOS = ("quantico", "gemeo", "classico")
MARGEM = np.log(1.25)
EPS = 1e-9


def dominio(func, W, alfa):
    o = S.OTIMOS[func]
    return (round(o - alfa * W, 6), round(o + (1 - alfa) * W, 6))


def tarefa(args):
    func, W, alfa, modo, sem = args
    S.OPC.clear()
    S.OPC.update(S.PADRAO, funcao=func)
    curva, _, _ = S.ga(modo, dominio(func, W, alfa), sem, pop_n=50, ger=GER)
    return func, alfa, modo, np.asarray(curva, dtype=float)


def welch(a, b):
    """diferenca de medias (a - b), erro-padrao e graus de liberdade de Welch."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    gl = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se > 0 else len(a) + len(b) - 2
    return a.mean() - b.mean(), se, gl


def tost(a, b):
    d, se, gl = welch(a, b)
    if se == 0:
        return d, se, 0.0 if abs(d) < MARGEM else 1.0
    p1 = 1 - stats.t.cdf((d + MARGEM) / se, gl)   # H0: d <= -margem
    p2 = stats.t.cdf((d - MARGEM) / se, gl)       # H0: d >= +margem
    return d, se, max(p1, p2)


def superior(a, b):
    """p unilateral de 'a melhor que b' (log f menor)."""
    d, se, gl = welch(a, b)
    if se == 0:
        return 0.0 if d < 0 else 1.0
    return stats.t.cdf(d / se, gl)


def holm(ps):
    ps = np.asarray(ps)
    ordem = np.argsort(ps)
    m = len(ps)
    rej = np.zeros(m, bool)
    for i, j in enumerate(ordem):
        if ps[j] <= 0.05 / (m - i):
            rej[j] = True
        else:
            break
    return rej


def main():
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 50
    t0 = time.time()
    tarefas = []
    for i, (func, W) in enumerate(FUNCS):
        for k, alfa in enumerate(ALFAS):
            for j, modo in enumerate(MODOS):
                tarefas += [(func, W, alfa, modo, SEMENTE + 10 ** 6 * i + 10 ** 4 * k + 1000 * j + r) for r in range(reps)]
    tarefas.sort(key=lambda a: a[3] != "quantico")
    with Pool(16) as pool:
        res = pool.map(tarefa, tarefas, chunksize=2)
    cur = {}
    for func, alfa, modo, c in res:
        cur.setdefault((func, alfa, modo), []).append(c)
    cur = {k: np.log(np.array(v) + EPS) for k, v in cur.items()}  # (reps, GER+1)
    np.savez_compressed(RAIZ / "resultados" / "qga" / f"ablacao_souza-{date.today()}.npz",
                        **{f"{f}|{a}|{m}": v for (f, a, m), v in cur.items()})

    L = [f"Ablacao Souza (protocolo ablacao_protocolo-2026-11-14.md) | semente {SEMENTE} | reps {reps} | {date.today()}",
         "medida: log(f + 1e-9) do melhor f avaliado; diferencas = media(log) A - B; negativo = A melhor",
         f"margem de equivalencia +-{MARGEM:.4f} (razao de medias geometricas 0,80 a 1,25)", ""]
    # H_A1
    L.append("H_A1 (g0): quantico - classico; previsto < 0 no centrado e >= 0 nos deslocados")
    a1 = []
    for func, W in FUNCS:
        partes = []
        for alfa in ALFAS:
            d, se, _ = welch(cur[(func, alfa, "quantico")][:, 0], cur[(func, alfa, "classico")][:, 0])
            ok = (d + 1.96 * se < 0) if alfa == 0.5 else (d + 1.96 * se >= 0)
            a1.append(ok)
            partes.append(f"a={alfa}: {d:+.3f} [{d - 1.96 * se:+.3f}, {d + 1.96 * se:+.3f}] {'ok' if ok else 'NAO'}")
        L.append(f"  {func:10s} " + " | ".join(partes))
    L.append(f"  H_A1 vale em {sum(a1)} de {len(a1)} celulas")
    L.append("")
    # H_A2 e H_A3
    for rival in ("classico", "gemeo"):
        ps, celulas = [], []
        for func, W in FUNCS:
            for alfa in ALFAS[1:]:
                a, b = cur[(func, alfa, "quantico")][:, GER], cur[(func, alfa, rival)][:, GER]
                ps.append(superior(a, b))
                celulas.append((func, alfa, a, b))
        rej = holm(ps)
        L.append(f"H_A2 (g30, deslocados): quantico superior a {rival}? (Holm, familia de {len(ps)})")
        L.append(f"  pares com superioridade do quantico: {int(rej.sum())} de {len(ps)}"
                 + ("" if not rej.any() else " -> " + ", ".join(f"{c[0]} a={c[1]}" for c, r in zip(celulas, rej) if r)))
        L.append(f"H_A3 (g30, deslocados): equivalencia quantico x {rival} (TOST, margem 0,80-1,25)")
        neq = 0
        for (func, alfa, a, b), p in zip(celulas, ps):
            d, se, pt = tost(a, b)
            eq = pt < 0.05
            neq += eq
            L.append(f"  {func:10s} a={alfa}: d={d:+.3f} IC90 [{d - 1.645 * se:+.3f}, {d + 1.645 * se:+.3f}] "
                     f"TOST p={pt:.3f} {'EQUIVALENTE' if eq else 'inconclusivo'} | p(sup)={p:.3f}")
        L.append(f"  equivalencia demonstrada em {neq} de {len(celulas)} pares")
        L.append("")
    # centrado e curvas (sem previsao)
    L.append("Sem previsao: g30 no centrado e onde o analogo fica atras do quantico (qualquer geracao, > 2 EP)")
    for func, W in FUNCS:
        a, b = cur[(func, 0.5, "quantico")][:, GER], cur[(func, 0.5, "classico")][:, GER]
        d, se, _ = welch(a, b)
        L.append(f"  {func:10s} centrado g30 quantico - classico: {d:+.3f} (EP {se:.3f})")
    atras = []
    for func, W in FUNCS:
        for alfa in ALFAS:
            q, g = cur[(func, alfa, "quantico")], cur[(func, alfa, "gemeo")]
            for gen in range(GER + 1):
                d, se, _ = welch(g[:, gen], q[:, gen])
                if d > 2 * se:
                    atras.append(f"{func} a={alfa} g{gen} ({d:+.2f}, EP {se:.2f})")
    L.append(f"  analogo atras do quantico (> 2 EP) em {len(atras)} celulas (funcao, dominio, geracao) de {len(FUNCS) * len(ALFAS) * (GER + 1)}")
    L += ["    " + x for x in atras[:60]]
    L.append(f"\ntempo {time.time() - t0:.0f}s")
    print("\n".join(L))
    (RAIZ / "resultados" / "qga" / f"ablacao_souza-{date.today()}.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
