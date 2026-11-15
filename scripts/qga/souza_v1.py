"""Reimplementation of Souza, Dardenne and Portugal (arXiv 2511.05254 v1) with the operators of the
current text (13 Nov 2026). Replaces the operators of vies_centro.py, which followed the method
marked "OLD METHOD" (commented out) in the TeX source.

Fixed by the text (Gate-Based-QGA.tex, sections 3.1 to 3.8 and 4):
  - m registers of n = 8 qubits; decoding x = a + z/(2^n - 1)(b - a), b_1 most significant;
  - random basis initial state; G_cl = {I, X, CNOT, SWAP, Toffoli, Fredkin},
    G_qu = G_cl + {H, T, T+, S, S+, Y, Z}; at most one gate per qubit per layer;
  - G_qu: first layer with H on k qubits (k uniform in 1..mn), remaining layers drawn at random;
  - fitness f(x*), x* = sample mean of the decoded values over 1024 shots;
  - 20% elitism (copied without mutation); the rest by crossover; the whole population, elites
    included, is re-evaluated every generation;
  - crossover with probability 0.7 (otherwise both parents pass unchanged): one layer delta
    drawn in the second half of the circuit; k uniform in 1..n; the gates acting on the k least
    significant qubits are swapped, together with the multiqubit gates entirely contained in
    that region; two children;
  - per-gene mutation (gate at one position), probability 0.3 each, with the seven
    arity-change cases of the text.

NOT fixed by the text (declared choices):
  - fixed depth d = 5; parent selection by binary tournament;
  - "k least significant qubits": the k least significant of EACH register;
  - in crossover, a gate from the other parent that collides with a gate that stayed (partly in
    the region) is discarded, and qubits without a gate receive I;
  - mutation: the new gate is drawn uniformly from G; its arity defines the case; in case
    2 -> 3 the order is (kept qubit, the other two in random order); in case 3 -> 2 the
    leftover qubit receives I; the genes of the H layer can also mutate;
  - each child inherits the basis initial state of the parent it comes from.

Entanglement between individuals (section 4.3): the text says that randomly drawn pairs "share
maximally entangled states" at the start, without saying which. By no-signaling, the measured
distribution of an individual A depends only on its reduced state: rho_A = U_A rho_A0 U_A^dagger,
with rho_A0 = Tr_B |Phi><Phi|, whatever the circuit of B and the pairing. Two readings:
  - "max": |Phi> = sum_z |z>_A |z>_B / 2^(mn/2) (maximal entanglement between the two systems);
    rho_A0 = I / 2^(mn), so the output of A is uniform for ANY circuit;
  - "bell": |Phi> = (|b_A b_B> + |~b_A ~b_B>)/sqrt(2) (one ebit, with ~ the bitwise complement);
    rho_A0 = (|b><b| + |~b><~b|)/2.
In both, the entangled individual is exactly equivalent to an individual with a random classical
initial state. The reduction is checked exactly in certifica_pares.py.
"""
from __future__ import annotations

import numpy as np

import vies_centro as VC

M, NB, NQ, DIM = VC.M, VC.NB, VC.NQ, VC.DIM
SHOTS, PROF = VC.SHOTS, VC.PROF
P_ELITE, P_CROSS, P_MUT = VC.P_ELITE, VC.P_CROSS, VC.P_MUT
G_CL, G_QU = VC.G_CL, VC.G_QU
UM = 1  # arities
TODOS = (1 << NQ) - 1

# Choices not fixed by the text, as options (default = the declared choices above); used in the
# sensitivity analysis (sensibilidade_souza.py). Alternatives:
#   prof: integer; selecao: "torneio" (tournament) | "roleta" (roulette, prob. proportional to 1/(1+f));
#   regiao: "por_registro" (per register) | "global" (the k least significant qubits of the whole
#     string, which all lie in the last register); colisao: "descarta_vem" | "descarta_fica" (the
#     gate coming from the other parent prevails, the one that stayed leaves and its qubits receive I);
#   sorteio: "uniforme" (gate uniform in G) | "por_aridade" (uniform arity, then gate);
#   h_muta: True | False (H layer frozen); base: "herda" (inherit) | "cruza" (in crossover the
#     initial-state bits in the swapped region are also swapped).
PADRAO = dict(prof=PROF, selecao="torneio", regiao="por_registro", colisao="descarta_vem",
              sorteio="uniforme", h_muta=True, base="herda", funcao="rastrigin")
OPC = dict(PADRAO)


# test functions (global minimum f = 0), usual formulas (Jamil and Yang 2013)
def esfera(x):
    x = np.asarray(x)
    return float(np.sum(x ** 2))


def ackley(x):
    x = np.asarray(x)
    d = len(x)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / d)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / d) + 20 + np.e)


def griewank(x):
    x = np.asarray(x)
    i = np.arange(1, len(x) + 1)
    return float(1 + np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(i))))


def rosenbrock(x):
    x = np.asarray(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


FUNCOES = dict(rastrigin=VC.rastrigin, esfera=esfera, ackley=ackley, griewank=griewank, rosenbrock=rosenbrock)
OTIMOS = dict(rastrigin=0.0, esfera=0.0, ackley=0.0, griewank=0.0, rosenbrock=1.0)  # coordinate of the minimum


def por_aridade(conjunto, a):
    return [g for g in conjunto if g[1] == a]


# ---------------- circuits (full layers: every qubit has a gate, I included) ----------------
def camada_cheia(rng, conjunto, nq=NQ):
    livres = list(rng.permutation(nq))
    gates = []
    while livres:
        cabe = [g for g in conjunto if g[1] <= len(livres)]
        nome, ar = cabe[rng.integers(len(cabe))]
        gates.append((nome, tuple(int(livres.pop()) for _ in range(ar))))
    return gates


def individuo(rng, conjunto, nq=NQ, prof=PROF):
    base = int(rng.integers(2 ** nq))
    if conjunto is G_QU:
        k = int(rng.integers(1, nq + 1))
        hs = set(int(q) for q in rng.choice(nq, size=k, replace=False))
        cams = [[("h", (q,)) if q in hs else ("i", (q,)) for q in range(nq)]]
        cams += [camada_cheia(rng, conjunto, nq) for _ in range(prof - 1)]
    else:
        cams = [camada_cheia(rng, conjunto, nq) for _ in range(prof)]
    return {"base": base, "cams": cams}


def sem_i(ind):
    """vies_centro format (no I gates), used by the simulators."""
    return {"base": ind["base"], "cams": [[g for g in cam if g[0] != "i"] for cam in ind["cams"]]}


def confere_camada(cam, nq=NQ):
    qs = [q for _, t in cam for q in t]
    return sorted(qs) == list(range(nq))


# ---------------- crossover ----------------
def regiao_baixa(k, m=M, nb=NB):
    """k least significant qubits of each register (qubit r*nb + 0 is the most significant)."""
    return {r * nb + nb - 1 - j for r in range(m) for j in range(k)}


def regiao_global(k, m=M, nb=NB):
    """k least significant qubits of the whole string (register m-1 is the last one)."""
    return {m * nb - 1 - j for j in range(k)}


def troca_camada(c1, c2, R, nq=NQ, colisao=None):
    colisao = colisao or OPC["colisao"]
    dentro = lambda g: set(g[1]) <= R  # noqa: E731
    fica1 = [g for g in c1 if not dentro(g)]
    fica2 = [g for g in c2 if not dentro(g)]
    vem2 = [g for g in c2 if dentro(g)]
    vem1 = [g for g in c1 if dentro(g)]

    def monta(fica, vem):
        if colisao == "descarta_fica":
            ocup_vem = {q for g in vem for q in g[1]}
            fica = [g for g in fica if not (set(g[1]) & ocup_vem)]
            nova = fica + list(vem)
            ocup = {q for g in nova for q in g[1]}
        else:
            ocup = {q for g in fica for q in g[1]}
            nova = list(fica)
            for g in vem:
                if not (set(g[1]) & ocup):
                    nova.append(g)
                    ocup |= set(g[1])
        nova += [("i", (q,)) for q in range(nq) if q not in ocup]
        return nova

    return monta(fica1, vem2), monta(fica2, vem1)


def cruza(rng, p1, p2, m=M, nb=NB):
    if rng.random() >= P_CROSS:
        return ({"base": p1["base"], "cams": list(p1["cams"])},
                {"base": p2["base"], "cams": list(p2["cams"])})
    d = len(p1["cams"])
    delta = int(rng.integers(d // 2, d))  # second half: d=5 -> layers 2, 3, 4 (0-based)
    k = int(rng.integers(1, nb + 1))
    R = regiao_baixa(k, m, nb) if OPC["regiao"] == "por_registro" else regiao_global(k, m, nb)
    a, b = troca_camada(p1["cams"][delta], p2["cams"][delta], R, m * nb)
    c1, c2 = list(p1["cams"]), list(p2["cams"])
    c1[delta], c2[delta] = a, b
    b1, b2 = p1["base"], p2["base"]
    if OPC["base"] == "cruza":
        mask = sum(1 << q for q in R)
        b1, b2 = (b1 & ~mask) | (b2 & mask), (b2 & ~mask) | (b1 & mask)
    return {"base": b1, "cams": c1}, {"base": b2, "cams": c2}


# ---------------- mutation (seven cases) ----------------
def muta_camada(rng, cam, conjunto):
    cam = list(cam)
    um = por_aridade(conjunto, 1)
    sorteia = lambda lst: lst[rng.integers(len(lst))]  # noqa: E731
    aridades = sorted({g[1] for g in conjunto})
    for g in list(cam):
        if g not in cam or rng.random() >= P_MUT:
            continue
        if OPC["sorteio"] == "por_aridade":
            nome_novo, a_novo = sorteia(por_aridade(conjunto, aridades[rng.integers(len(aridades))]))
        else:
            nome_novo, a_novo = sorteia(conjunto)
        qs = g[1]
        a = len(qs)
        livres1 = [h for h in cam if len(h[1]) == 1 and h is not g and h != g]
        i = cam.index(g)
        if a_novo == a:  # case 1
            cam[i] = (nome_novo, qs)
        elif a == 1 and a_novo in (2, 3):  # cases 2 and 3
            precisa = a_novo - 1
            if len(livres1) < precisa:
                cam[i] = (sorteia(um)[0], qs)  # falls back to case 1
                continue
            alvos = [livres1[j] for j in rng.choice(len(livres1), size=precisa, replace=False)]
            for h in alvos:
                cam.remove(h)
            cam[cam.index(g)] = (nome_novo, qs + tuple(h[1][0] for h in alvos))
        elif a_novo == 1:  # cases 4 and 5
            cam.pop(i)
            cam += [(sorteia(um)[0], (q,)) for q in qs]
        elif a == 2 and a_novo == 3:  # case 6
            if not livres1:
                cam[i] = (sorteia(por_aridade(conjunto, 2))[0], qs)
                continue
            h = livres1[rng.integers(len(livres1))]
            cam.remove(h)
            j = int(rng.integers(2))
            resto = [qs[1 - j], h[1][0]]
            resto = [resto[t] for t in rng.permutation(2)]
            cam[cam.index(g)] = (nome_novo, (qs[j],) + tuple(resto))
        elif a == 3 and a_novo == 2:  # case 7
            ordem = rng.permutation(3)
            cam[i] = (nome_novo, (qs[ordem[0]], qs[ordem[1]]))
            cam.append(("i", (qs[ordem[2]],)))
        else:
            raise AssertionError((g, nome_novo))
    return cam


def muta(rng, ind, conjunto):
    congela = conjunto is G_QU and not OPC["h_muta"]
    return {"base": ind["base"], "cams": [c if (congela and j == 0) else muta_camada(rng, c, conjunto)
                                          for j, c in enumerate(ind["cams"])]}


# ---------------- evaluation ----------------
def p_quantico(ind, emar=None):
    """Exact distribution (floating point) of the measured output of an individual."""
    if emar == "max":
        return np.full(DIM, 1.0 / DIM)
    s = sem_i(ind)
    p = np.abs(VC.estado(s)) ** 2
    if emar == "bell":
        s2 = {"base": s["base"] ^ TODOS, "cams": s["cams"]}
        p = 0.5 * (p + np.abs(VC.estado(s2)) ** 2)
    return p / p.sum()


def amostra_bits(ind, rng, gemeo, emar=None):
    """Classical (G_cl) or the stochastic analog of G_qu, shot by shot, with a random classical
    initial state when emar != None (the classical equivalent of the entangled individual)."""
    if emar == "max":
        ini = rng.integers(DIM, size=SHOTS)
    elif emar == "bell":
        ini = np.where(rng.random(SHOTS) < 0.5, ind["base"], ind["base"] ^ TODOS)
    else:
        ini = np.full(SHOTS, ind["base"])
    bits = ((ini[:, None] >> np.arange(NQ)) & 1).astype(bool)
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
    return (bits.astype(np.int64) << np.arange(NQ)).sum(axis=1)


def aptidao(ind, modo, dominio, rng, emar=None):
    if modo == "quantico":
        idx = rng.choice(DIM, size=SHOTS, p=p_quantico(ind, emar))
    else:
        idx = amostra_bits(ind, rng, gemeo=(modo == "gemeo"), emar=emar)
    return FUNCOES[OPC["funcao"]](VC.decodifica(idx, dominio).mean(axis=0))


def aptidao_exata(ind, dominio, emar=None):
    """f(E[x]) without shot noise (quantum mode only)."""
    p = p_quantico(ind, emar)
    return FUNCOES[OPC["funcao"]]((VC.decodifica(np.arange(DIM), dominio) * p[:, None]).sum(axis=0))


# ---------------- genetic algorithm ----------------
def ga(modo, dominio, semente, pop_n=50, ger=30, emar=None):
    """Returns (curve of the best evaluated f per generation, best final individual, its f)."""
    rng = np.random.default_rng(semente)
    conjunto = G_CL if modo == "classico" else G_QU
    pop = [individuo(rng, conjunto, prof=OPC["prof"]) for _ in range(pop_n)]
    fit = [aptidao(i, modo, dominio, rng, emar) for i in pop]
    curva = [min(fit)]
    n_elite = int(P_ELITE * pop_n)
    for _ in range(ger):
        ordem = np.argsort(fit)
        nova = [pop[i] for i in ordem[:n_elite]]
        if OPC["selecao"] == "roleta":
            w = 1.0 / (1.0 + np.asarray(fit))
            w = w / w.sum()

        def torneio():
            if OPC["selecao"] == "roleta":
                return pop[rng.choice(pop_n, p=w)]
            a, b = rng.integers(pop_n, size=2)
            return pop[a] if fit[a] <= fit[b] else pop[b]

        while len(nova) < pop_n:
            f1, f2 = cruza(rng, torneio(), torneio())
            nova.append(muta(rng, f1, conjunto))
            if len(nova) < pop_n:
                nova.append(muta(rng, f2, conjunto))
        pop = nova
        fit = [aptidao(i, modo, dominio, rng, emar) for i in pop]  # all re-evaluated
        curva.append(min(fit))
    i = int(np.argmin(fit))
    return curva, pop[i], fit[i]
