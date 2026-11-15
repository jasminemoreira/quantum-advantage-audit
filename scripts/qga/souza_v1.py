"""Reimplementacao de Souza, Dardenne e Portugal (arXiv 2511.05254 v1) com os operadores do texto
atual (13/11/2026). Substitui os operadores de vies_centro.py, que seguiam o metodo marcado como
"OLD METHOD" (comentado) no fonte TeX.

Fixado pelo texto (Gate-Based-QGA.tex, secoes 3.1 a 3.8 e 4):
  - m registros de n = 8 qubits; decodificacao x = a + z/(2^n - 1)(b - a), b_1 mais significativo;
  - estado inicial de base aleatorio; G_cl = {I, X, CNOT, SWAP, Toffoli, Fredkin},
    G_qu = G_cl + {H, T, T+, S, S+, Y, Z}; no maximo uma porta por qubit por camada;
  - G_qu: primeira camada com H em k qubits (k uniforme em 1..mn), demais camadas sorteadas;
  - aptidao f(x*), x* = media amostral dos valores decodificados em 1024 shots;
  - elitismo 20% (copiados sem mutacao); o restante por cruzamento; a populacao inteira, elites
    inclusive, e reavaliada a cada geracao;
  - cruzamento com probabilidade 0,7 (senao os dois pais passam inalterados): uma camada delta
    sorteada na segunda metade do circuito; k uniforme em 1..n; trocam-se as portas que agem
    nos k qubits menos significativos, junto com as portas multiqubit inteiramente contidas
    nessa regiao; dois filhos;
  - mutacao por gene (porta numa posicao), probabilidade 0,3 cada, com os sete casos de troca
    de aridade do texto.

NAO fixado pelo texto (escolhas declaradas):
  - profundidade fixa d = 5; selecao dos pais por torneio binario;
  - "k qubits menos significativos": os k menos significativos de CADA registro;
  - no cruzamento, uma porta do outro pai que colida com uma porta que ficou (parcialmente na
    regiao) e descartada, e os qubits sem porta recebem I;
  - mutacao: a porta nova e sorteada uniformemente em G; a aridade dela define o caso; no caso
    2 -> 3 a ordem e (qubit mantido, os outros dois em ordem aleatoria); no caso 3 -> 2 o qubit
    que sobra recebe I; os genes da camada de H tambem podem mutar;
  - cada filho herda o estado inicial de base do pai de onde vem.

Emaranhamento entre individuos (secao 4.3): o texto diz que pares sorteados "compartilham estados
maximamente emaranhados" no inicio, sem dizer quais. Por nao sinalizacao, a distribuicao medida
num individuo A so depende do estado reduzido dele: rho_A = U_A rho_A0 U_A^dagger, com
rho_A0 = Tr_B |Phi><Phi|, qualquer que seja o circuito de B e o pareamento. Duas leituras:
  - "max": |Phi> = soma_z |z>_A |z>_B / 2^(mn/2) (emaranhamento maximo entre os dois sistemas);
    rho_A0 = I / 2^(mn), entao a saida de A e uniforme para QUALQUER circuito;
  - "bell": |Phi> = (|b_A b_B> + |~b_A ~b_B>)/sqrt(2) (um ebit, com ~ o complemento bit a bit);
    rho_A0 = (|b><b| + |~b><~b|)/2.
Nas duas, o individuo emaranhado equivale exatamente a um individuo com estado inicial classico
aleatorio. A reducao e conferida de forma exata em certifica_pares.py.
"""
from __future__ import annotations

import numpy as np

import vies_centro as VC

M, NB, NQ, DIM = VC.M, VC.NB, VC.NQ, VC.DIM
SHOTS, PROF = VC.SHOTS, VC.PROF
P_ELITE, P_CROSS, P_MUT = VC.P_ELITE, VC.P_CROSS, VC.P_MUT
G_CL, G_QU = VC.G_CL, VC.G_QU
UM = 1  # aridades
TODOS = (1 << NQ) - 1

# Escolhas nao fixadas pelo texto, como opcoes (padrao = as escolhas declaradas acima); usadas na
# analise de sensibilidade (sensibilidade_souza.py). Alternativas:
#   prof: inteiro; selecao: "torneio" | "roleta" (prob. proporcional a 1/(1+f));
#   regiao: "por_registro" | "global" (os k qubits menos significativos da cadeia inteira, que
#     ficam todos no ultimo registro); colisao: "descarta_vem" | "descarta_fica" (a porta que vem
#     do outro pai prevalece, a que ficou sai e os qubits dela recebem I);
#   sorteio: "uniforme" (porta uniforme em G) | "por_aridade" (aridade uniforme, depois porta);
#   h_muta: True | False (camada de H congelada); base: "herda" | "cruza" (no cruzamento os bits
#     do estado inicial na regiao trocada tambem sao trocados).
PADRAO = dict(prof=PROF, selecao="torneio", regiao="por_registro", colisao="descarta_vem",
              sorteio="uniforme", h_muta=True, base="herda", funcao="rastrigin")
OPC = dict(PADRAO)


# funcoes de teste (minimo global f = 0), formulas usuais (Jamil e Yang 2013)
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
OTIMOS = dict(rastrigin=0.0, esfera=0.0, ackley=0.0, griewank=0.0, rosenbrock=1.0)  # coordenada do minimo


def por_aridade(conjunto, a):
    return [g for g in conjunto if g[1] == a]


# ---------------- circuitos (camadas cheias: todo qubit tem uma porta, I inclusive) ----------------
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
    """Formato de vies_centro (sem portas I), usado pelos simuladores."""
    return {"base": ind["base"], "cams": [[g for g in cam if g[0] != "i"] for cam in ind["cams"]]}


def confere_camada(cam, nq=NQ):
    qs = [q for _, t in cam for q in t]
    return sorted(qs) == list(range(nq))


# ---------------- cruzamento ----------------
def regiao_baixa(k, m=M, nb=NB):
    """k qubits menos significativos de cada registro (qubit r*nb + 0 e o mais significativo)."""
    return {r * nb + nb - 1 - j for r in range(m) for j in range(k)}


def regiao_global(k, m=M, nb=NB):
    """k qubits menos significativos da cadeia inteira (registro m-1 e o ultimo)."""
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
    delta = int(rng.integers(d // 2, d))  # segunda metade: d=5 -> camadas 2, 3, 4 (base 0)
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


# ---------------- mutacao (sete casos) ----------------
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
        if a_novo == a:  # caso 1
            cam[i] = (nome_novo, qs)
        elif a == 1 and a_novo in (2, 3):  # casos 2 e 3
            precisa = a_novo - 1
            if len(livres1) < precisa:
                cam[i] = (sorteia(um)[0], qs)  # volta ao caso 1
                continue
            alvos = [livres1[j] for j in rng.choice(len(livres1), size=precisa, replace=False)]
            for h in alvos:
                cam.remove(h)
            cam[cam.index(g)] = (nome_novo, qs + tuple(h[1][0] for h in alvos))
        elif a_novo == 1:  # casos 4 e 5
            cam.pop(i)
            cam += [(sorteia(um)[0], (q,)) for q in qs]
        elif a == 2 and a_novo == 3:  # caso 6
            if not livres1:
                cam[i] = (sorteia(por_aridade(conjunto, 2))[0], qs)
                continue
            h = livres1[rng.integers(len(livres1))]
            cam.remove(h)
            j = int(rng.integers(2))
            resto = [qs[1 - j], h[1][0]]
            resto = [resto[t] for t in rng.permutation(2)]
            cam[cam.index(g)] = (nome_novo, (qs[j],) + tuple(resto))
        elif a == 3 and a_novo == 2:  # caso 7
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


# ---------------- avaliacao ----------------
def p_quantico(ind, emar=None):
    """Distribuicao exata (ponto flutuante) da saida medida de um individuo."""
    if emar == "max":
        return np.full(DIM, 1.0 / DIM)
    s = sem_i(ind)
    p = np.abs(VC.estado(s)) ** 2
    if emar == "bell":
        s2 = {"base": s["base"] ^ TODOS, "cams": s["cams"]}
        p = 0.5 * (p + np.abs(VC.estado(s2)) ** 2)
    return p / p.sum()


def amostra_bits(ind, rng, gemeo, emar=None):
    """Classico (G_cl) ou gemeo classico de G_qu, shot a shot, com estado inicial classico
    aleatorio quando emar != None (a equivalencia classica do individuo emaranhado)."""
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
                    raise ValueError("H no avaliador classico")
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
    """f(E[x]) sem ruido de shots (so para o modo quantico)."""
    p = p_quantico(ind, emar)
    return FUNCOES[OPC["funcao"]]((VC.decodifica(np.arange(DIM), dominio) * p[:, None]).sum(axis=0))


# ---------------- algoritmo genetico ----------------
def ga(modo, dominio, semente, pop_n=50, ger=30, emar=None):
    """Devolve (curva do melhor f avaliado por geracao, melhor individuo final, f dele)."""
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
        fit = [aptidao(i, modo, dominio, rng, emar) for i in pop]  # todos reavaliados
        curva.append(min(fit))
    i = int(np.argmin(fit))
    return curva, pop[i], fit[i]
