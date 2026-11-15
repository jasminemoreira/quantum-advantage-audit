"""Recertificacao do emaranhamento dentro de cada individuo, com os operadores da v1 (13/11/2026).

Mesmos testes exatos de certifica_souza.py (separabilidade entre os registros, posto de Schmidt
certificado, qubits emaranhados, fatoracao de p(x), marginais), agora com os melhores individuos
finais do AG de souza_v1.py (|P| = 50, 30 geracoes, 20 repeticoes por dominio) e 100 individuos
iniciais sorteados pela inicializacao da v1. Inclui a correlacao de Spearman entre f e o posto.

Uso (WSL): externos/QFIE/.venv/bin/python scripts/qga/certifica_souza_v1.py
Saida: resultados/qga/certifica_souza_v1-<data>.txt
"""
from __future__ import annotations

import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import certifica_souza as CS  # noqa: E402
import souza_v1 as S  # noqa: E402

RAIZ = AQUI.parents[1]
SEMENTE = 20261116
REPS = 20
DOMINIOS = ((-5.12, 5.12), (-2.12, 8.12))


def ga_ind(args):
    dom, sem = args
    _, melhor, f = S.ga("quantico", dom, sem, pop_n=50, ger=30)
    return dom, S.sem_i(melhor), f


def main():
    rng = np.random.default_rng(SEMENTE)
    tarefas = [(dom, SEMENTE + 1000 * r + k) for k, dom in enumerate(DOMINIOS) for r in range(REPS)]
    with Pool(16) as pool:
        finais = pool.map(ga_ind, tarefas, chunksize=1)
        iniciais = [("inicial", S.sem_i(S.individuo(rng, S.G_QU)), float("nan")) for _ in range(100)]
        args = [(f"final {d}", ind, f) for d, ind, f in finais] + iniciais
        certs = pool.map(CS.certifica, args, chunksize=1)
    linhas = [f"Certificacao exata do emaranhamento intra-individuo, operadores da v1 | semente {SEMENTE} | {date.today()}",
              f"primos {CS.PRIMOS}"]
    for rot in [f"final {d}" for d in DOMINIOS] + ["inicial"]:
        rs = [c for c in certs if c["rotulo"] == rot]
        linhas.append(CS.resumo(rot, rs))
        if rot != "inicial":
            rho = spearmanr([c["f"] for c in rs], [c["pinf"] for c in rs]).statistic
            linhas.append(f"  Spearman(f, posto) = {rho:+.2f} | melhor f {min(c['f'] for c in rs):.4f}, "
                          f"separavel? {min(rs, key=lambda c: c['f'])['sep']}")
    linhas.append("")
    linhas.append("melhores finais: dominio | f | k | suporte | separavel | posto >= | posto SVD | qubits emaranhados | p fatora | marginais em {0,1/2,1}")
    for c in certs:
        if c["rotulo"].startswith("final"):
            linhas.append(f"{c['rotulo']} | {c['f']:.4f} | {c['k']} | {c['suporte']} | {c['sep']} | {c['pinf']} | "
                          f"{c['psvd']} | {c['emar']} | {c['fatora']} | {c['meio']}")
    print("\n".join(linhas[:9]))
    (RAIZ / "resultados" / "qga" / f"certifica_souza_v1-{date.today()}.txt").write_text("\n".join(linhas) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
