"""Confere numericamente a cota de Pinsker do vies de centro em individuos do Souza (v1).

Cota global:   |E[x_i] - c_i| <= (b - a) * sqrt(ln2 * (N - H(p)) / 2)
Cota marginal: |E[x_i] - c_i| <= (b - a) * sqrt(ln2 * (n - H(p_i)) / 2), p_i = marginal do registro i
"""
import sys
import numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import souza_v1 as S  # noqa: E402
import vies_centro as VC  # noqa: E402


def H(p):
    q = p[p > 1e-15]
    return float(-(q * np.log2(q)).sum())


rng = np.random.default_rng(20261120)
dom = (-2.12, 8.12)
a, b = dom
c = (a + b) / 2
pior_g = pior_m = 0.0
viol = 0
idx = np.arange(S.DIM)
xs = VC.decodifica(idx, dom)  # (DIM, M)
for k in range(400):
    ind = S.individuo(rng, S.G_QU)
    p = S.p_quantico(ind)
    Hp = H(p)
    ex = (xs * p[:, None]).sum(axis=0)
    for i in range(S.M):
        # marginal do registro i: agrupa por valor decodificado de x_i (bijecao com z_i)
        vals, inv = np.unique(xs[:, i], return_inverse=True)
        pi = np.bincount(inv, weights=p, minlength=len(vals))
        cg = (b - a) * np.sqrt(np.log(2) * max(S.NQ - Hp, 0) / 2)
        cm = (b - a) * np.sqrt(np.log(2) * max(S.NB - H(pi), 0) / 2)
        d = abs(ex[i] - c)
        assert len(vals) == 2 ** S.NB
        if d > cm + 1e-12 or cm > cg + 1e-12:
            viol += 1
        if cg > 0:
            pior_g = max(pior_g, d / cg)
        if cm > 0:
            pior_m = max(pior_m, d / cm)
print(f"N={S.NQ}, n={S.NB}, m={S.M}; 400 individuos; violacoes (d > cota marginal ou marginal > global): {viol}")
print(f"maior razao d/cota global {pior_g:.3f}; d/cota marginal {pior_m:.3f}")
