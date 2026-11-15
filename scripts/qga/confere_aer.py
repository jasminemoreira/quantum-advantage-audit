"""Confere a marginal exata da aptidao (motor esparso) contra o Aer em dois casos (13/11/2026)."""
import sys, math
from pathlib import Path
import networkx as nx, numpy as np
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parent / "qcalc"))
import verifica_rqga as V
from esparso import MotorEsparso
from converte import converte
from certifica_rqga import prob_decimal
rng = np.random.default_rng(V.SEMENTE)
g = nx.erdos_renyi_graph(3, 0.6, seed=int(rng.integers(2**31)))
ns = V.carrega_rqga(3, nx.to_numpy_array(g, dtype=int).tolist())
for it in (1, 3):
    qc, ind, fit = V.circuito_rqga(ns, it)
    idx = {q: i for i, q in enumerate(qc.qubits)}
    m = MotorEsparso(qc.num_qubits)
    for p in converte(qc): m.aplica(p)
    mf = m.marginal([idx[q] for q in fit])
    ex = np.zeros(2 ** len(fit))
    for y, v in mf.items(): ex[y] = prob_decimal(v, m.k)
    _, aer = V.distribuicoes(qc, ind, fit)
    print(f"iter={it} max|exato - Aer| na aptidao = {np.abs(ex - aer).max():.2e}")
