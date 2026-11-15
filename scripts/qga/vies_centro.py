"""Article 3 (11 Nov 2026): test of the "center bias" in Souza, Dardenne and Portugal (arXiv 2511.05254 v1).

Hypothesis: the article's fitness is f(mean of the decoded values). A uniform output on a
register of n bits has its decoded mean exactly at the center of the domain, and for Rastrigin on
[-5.12, 5.12] the center is the global optimum. The gain attributed to superposition and
entanglement could come from this, and not from a quantum effect.

The article has no code; this is a REIMPLEMENTATION from the text. What the text fixes:
m=2 variables, n=8 qubits per variable, 1024 shots, |P|=50, 20% elite, 70% crossover,
30% mutation, G_cl = {I, X, CNOT, SWAP, Toffoli, Fredkin}, G_qu = G_cl + {H, Y, Z, T, T+, S, S+},
random basis initial state, first layer of G_qu with H on k qubits (k uniform in
1..mn), at most one gate per qubit per layer, fitness f(x*) with x* the decoded mean,
decoding x = a + z/(2^n - 1)(b - a). What the text does NOT fix and was chosen here
(to be declared in the article): fixed depth d=5; binary tournament selection; one-point
crossover between layer lists; mutation = replace one layer with another drawn at random; elites
keep the fitness already computed; 30 generations; 100 repetitions.

Three evaluators over the same genetic operators:
  quantico  : G_qu simulated as an exact state vector (numpy) and sampled with 1024 shots;
  classico  : G_cl (deterministic; "zero entropy" in the article);
  gemeo     : stochastic analog without interference: G_qu with H replaced by a fair classical
              coin on the bit, Y by X and the phases (Z, S, T) removed. It is a randomized
              classical algorithm with the same structure.
Two domains: centered [-5.12, 5.12] (optimum at the center) and shifted [-2.12, 8.12] (optimum
x=0 far from the center, which lies at x=3).

Usage (WSL): externos/QFIE/.venv/bin/python scripts/qga/vies_centro.py
Output: resultados/qga/vies_centro-<date>.txt
"""
from __future__ import annotations

import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
SEMENTE = 20261111
M, NB = 2, 8
NQ = M * NB
DIM = 2 ** NQ
SHOTS, POP, GER, REPS, PROF = 1024, 50, 30, 100, 5
P_ELITE, P_CROSS, P_MUT = 0.2, 0.7, 0.3
IDX = np.arange(DIM, dtype=np.int64)
BIT = [((IDX >> q) & 1).astype(bool) for q in range(NQ)]  # qubit q = bit q of the index
G_CL = [("i", 1), ("x", 1), ("cx", 2), ("swap", 2), ("ccx", 3), ("cswap", 3)]
G_QU = G_CL + [("h", 1), ("y", 1), ("z", 1), ("t", 1), ("tdg", 1), ("s", 1), ("sdg", 1)]
FASE = {"z": -1, "s": 1j, "sdg": -1j, "t": np.exp(1j * np.pi / 4), "tdg": np.exp(-1j * np.pi / 4)}


def rastrigin(x):
    x = np.asarray(x)
    return 10 * len(x) + float(np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


# ---------------- circuits ----------------
def camada(rng, conjunto, so_h=None):
    if so_h is not None:
        return [("h", (q,)) for q in so_h]
    livres = list(rng.permutation(NQ))
    gates = []
    while livres:
        cabe = [g for g in conjunto if g[1] <= len(livres)]
        nome, ar = cabe[rng.integers(len(cabe))]
        qs = tuple(int(livres.pop()) for _ in range(ar))
        if nome != "i":
            gates.append((nome, qs))
    return gates


def individuo(rng, conjunto):
    base = int(rng.integers(DIM))
    if conjunto is G_QU:
        k = int(rng.integers(1, NQ + 1))
        cams = [camada(rng, None, so_h=list(rng.choice(NQ, size=k, replace=False)))]
        cams += [camada(rng, conjunto) for _ in range(PROF - 1)]
    else:
        cams = [camada(rng, conjunto) for _ in range(PROF)]
    return {"base": base, "cams": cams}


def perm_x(qs):
    """Source index for permutation gates (all involutions)."""
    nome, q = qs
    if nome == "x":
        return IDX ^ (1 << q[0])
    if nome == "cx":
        c, t = q
        return np.where(BIT[c], IDX ^ (1 << t), IDX)
    if nome == "ccx":
        c1, c2, t = q
        return np.where(BIT[c1] & BIT[c2], IDX ^ (1 << t), IDX)
    if nome == "swap":
        a, b = q
        dif = BIT[a] != BIT[b]
        return np.where(dif, IDX ^ ((1 << a) | (1 << b)), IDX)
    if nome == "cswap":
        c, a, b = q
        dif = BIT[c] & (BIT[a] != BIT[b])
        return np.where(dif, IDX ^ ((1 << a) | (1 << b)), IDX)
    raise ValueError(nome)


def estado(ind):
    psi = np.zeros(DIM, dtype=complex)
    psi[ind["base"]] = 1.0
    for cam in ind["cams"]:
        for nome, q in cam:
            if nome in ("x", "cx", "ccx", "swap", "cswap"):
                psi = psi[perm_x((nome, q))]
            elif nome == "h":
                b = BIT[q[0]]
                par = psi[IDX ^ (1 << q[0])]
                psi = np.where(b, par - psi, psi + par) / np.sqrt(2)
            elif nome == "y":  # Y|0> = i|1>, Y|1> = -i|0>
                psi = psi[IDX ^ (1 << q[0])] * np.where(BIT[q[0]], 1j, -1j)
            else:
                psi = np.where(BIT[q[0]], psi * FASE[nome], psi)
    return psi


def decodifica(idx, dominio):
    a, b = dominio
    xs = []
    for r in range(M):
        z = np.zeros(len(idx), dtype=np.int64)
        for j in range(NB):  # b_1 (most significant) = qubit r*NB + 0
            z = (z << 1) | ((idx >> (r * NB + j)) & 1)
        xs.append(a + z / (2 ** NB - 1) * (b - a))
    return np.stack(xs, axis=1)


def amostra_quantico(ind, rng):
    p = np.abs(estado(ind)) ** 2
    return rng.choice(DIM, size=SHOTS, p=p / p.sum())


def amostra_bits(ind, rng, gemeo):
    """Classical (G_cl) or the stochastic analog of G_qu, shot by shot over bits."""
    bits = np.zeros((SHOTS, NQ), dtype=bool)
    for q in range(NQ):
        bits[:, q] = (ind["base"] >> q) & 1
    for cam in ind["cams"]:
        for nome, q in cam:
            if nome in ("x", "y"):
                bits[:, q[0]] ^= True
            elif nome == "h":
                if not gemeo:
                    raise ValueError("H in the classical evaluator")
                bits[:, q[0]] ^= rng.random(SHOTS) < 0.5
            elif nome == "cx":
                bits[:, q[1]] ^= bits[:, q[0]]
            elif nome == "ccx":
                bits[:, q[2]] ^= bits[:, q[0]] & bits[:, q[1]]
            elif nome == "swap":
                bits[:, [q[0], q[1]]] = bits[:, [q[1], q[0]]]
            elif nome == "cswap":
                c = bits[:, q[0]]
                a, b = bits[:, q[1]].copy(), bits[:, q[2]].copy()
                bits[:, q[1]] = np.where(c, b, a)
                bits[:, q[2]] = np.where(c, a, b)
            # z, s, sdg, t, tdg: phase only, no effect in the computational basis
    return (bits.astype(np.int64) << np.arange(NQ)).sum(axis=1)


def aptidao(ind, modo, dominio, rng):
    if modo == "quantico":
        idx = amostra_quantico(ind, rng)
    else:
        idx = amostra_bits(ind, rng, gemeo=(modo == "gemeo"))
    xstar = decodifica(idx, dominio).mean(axis=0)
    return rastrigin(xstar)


# ---------------- genetic algorithm ----------------
def roda_ga(args):
    modo, dominio, semente = args
    rng = np.random.default_rng(semente)
    conjunto = G_CL if modo == "classico" else G_QU
    pop = [individuo(rng, conjunto) for _ in range(POP)]
    fit = [aptidao(i, modo, dominio, rng) for i in pop]
    melhor = [min(fit)]
    n_elite = int(P_ELITE * POP)
    for _ in range(GER):
        ordem = np.argsort(fit)
        nova = [pop[i] for i in ordem[:n_elite]]
        nfit = [fit[i] for i in ordem[:n_elite]]

        def torneio():
            a, b = rng.integers(POP, size=2)
            return pop[a] if fit[a] <= fit[b] else pop[b]

        while len(nova) < POP:
            p1, p2 = torneio(), torneio()
            cams = list(p1["cams"])
            if rng.random() < P_CROSS:
                corte = int(rng.integers(1, PROF))
                cams = list(p1["cams"][:corte]) + list(p2["cams"][corte:])
            if rng.random() < P_MUT:
                cams[int(rng.integers(PROF))] = camada(rng, conjunto)
            filho = {"base": p1["base"], "cams": cams}
            nova.append(filho)
            nfit.append(aptidao(filho, modo, dominio, rng))
        pop, fit = nova, nfit
        melhor.append(min(fit))
    return modo, dominio, melhor


# ---------------- checks ----------------
def checagens():
    rng = np.random.default_rng(SEMENTE)
    linhas = []
    # 1) simulator: permutation-only circuits match the classical evaluator
    erros = 0
    for _ in range(50):
        ind = individuo(rng, G_CL)
        psi = estado(ind)
        k = int(np.argmax(np.abs(psi)))
        erros += int(abs(abs(psi[k]) - 1) > 1e-12 or k != int(amostra_bits(ind, rng, False)[0]))
    linhas.append(f"check 1: 50 G_cl circuits, state vector = classical bits? errors = {erros}")
    # 1b) own simulator against Qiskit's Statevector on randomly drawn G_qu circuits
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    dif = 0.0
    for _ in range(20):
        ind = individuo(rng, G_QU)
        qc = QuantumCircuit(NQ)
        for q in range(NQ):
            if (ind["base"] >> q) & 1:
                qc.x(q)
        for cam in ind["cams"]:
            for nome, q in cam:
                getattr(qc, nome)(*q)
        dif = max(dif, float(np.abs(Statevector(qc).data - estado(ind)).max()))
    linhas.append(f"check 1b: 20 G_qu circuits, max|psi_own - psi_qiskit| = {dif:.2e}")
    # 2) H on all qubits: exact decoded mean = center of the domain
    ind = {"base": 0, "cams": [[("h", (q,)) for q in range(NQ)]]}
    p = np.abs(estado(ind)) ** 2
    for dom in ((-5.12, 5.12), (-2.12, 8.12)):
        media = (decodifica(IDX, dom) * p[:, None]).sum(axis=0)
        linhas.append(f"check 2: H on all, domain {dom}: exact mean = {media.round(12)}, "
                      f"f(mean) = {rastrigin(media):.4f}")
    # 3) maximally entangled pair between two 8-qubit registers: uniform marginal
    ind = {"base": 0, "cams": [[("h", (q,)) for q in range(NB)], [("cx", (q, q + NB)) for q in range(NB)]]}
    p = (np.abs(estado(ind)) ** 2).reshape(2 ** NB, 2 ** NB)  # [register 2, register 1]
    marg = p.sum(axis=0)
    linhas.append(f"check 3: entangled pair, marginal of one register: max|p - 1/256| = "
                  f"{np.abs(marg - 1 / 256).max():.2e}")
    # 4) generation 0: f(x*) of randomly drawn initial individuals, per evaluator and domain
    for dom in ((-5.12, 5.12), (-2.12, 8.12)):
        for modo in ("quantico", "gemeo", "classico"):
            conj = G_CL if modo == "classico" else G_QU
            fs = [aptidao(individuo(rng, conj), modo, dom, rng) for _ in range(300)]
            linhas.append(f"check 4: generation 0, domain {dom}, {modo:8s}: median f {np.median(fs):7.3f} "
                          f"| fraction with f < 1: {np.mean(np.array(fs) < 1):.3f}")
    return linhas


def main():
    linhas = [f"Center bias in Souza/Dardenne/Portugal (reimplementation) | seed {SEMENTE} | "
              f"{date.today()}", f"m={M} n={NB} shots={SHOTS} |P|={POP} G={GER} reps={REPS} d={PROF}", ""]
    linhas += checagens()
    print("\n".join(linhas), flush=True)
    dominios = ((-5.12, 5.12), (-2.12, 8.12))
    tarefas = [(modo, dom, SEMENTE + 1000 * r + k)
               for k, dom in enumerate(dominios)
               for modo in ("quantico", "gemeo", "classico")
               for r in range(REPS)]
    with Pool(16) as pool:
        res = pool.map(roda_ga, tarefas, chunksize=1)
    linhas.append("")
    linhas.append("best f per generation (mean over repetitions; in parentheses, standard error)")
    gens = (0, 1, 2, 5, 10, 20, 30)
    linhas.append("domain evaluator | " + " | ".join(f"g{g}" for g in gens))
    for dom in dominios:
        for modo in ("quantico", "gemeo", "classico"):
            curvas = np.array([m for (mo, d, m) in res if mo == modo and d == dom])
            med, ep = curvas.mean(axis=0), curvas.std(axis=0, ddof=1) / np.sqrt(len(curvas))
            linhas.append(f"{dom} {modo:8s} | " + " | ".join(f"{med[g]:.3f} ({ep[g]:.3f})" for g in gens))
    texto = "\n".join(linhas)
    print("\n".join(linhas[-8:]))
    destino = RAIZ / "resultados" / "qga"
    destino.mkdir(parents=True, exist_ok=True)
    (destino / f"vies_centro-{date.today()}.txt").write_text(texto + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
