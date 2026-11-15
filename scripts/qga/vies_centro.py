"""Artigo 3 (11/11/2026): teste do "vies de centro" em Souza, Dardenne e Portugal (arXiv 2511.05254 v1).

Hipotese: a aptidao do artigo e f(media dos valores decodificados). Uma saida uniforme num
registro de n bits tem media decodificada exatamente no centro do dominio, e no Rastrigin em
[-5.12, 5.12] o centro e o otimo global. O ganho atribuido a superposicao e ao emaranhamento
poderia vir dai, e nao de efeito quantico.

O artigo nao tem codigo; isto e uma REIMPLEMENTACAO a partir do texto. O que o texto fixa:
m=2 variaveis, n=8 qubits por variavel, 1024 shots, |P|=50, elite 20%, cruzamento 70%,
mutacao 30%, G_cl = {I, X, CNOT, SWAP, Toffoli, Fredkin}, G_qu = G_cl + {H, Y, Z, T, T+, S, S+},
estado inicial de base aleatorio, primeira camada de G_qu com H em k qubits (k uniforme em
1..mn), no maximo uma porta por qubit por camada, aptidao f(x*) com x* a media decodificada,
decodificacao x = a + z/(2^n - 1)(b - a). O que o texto NAO fixa e foi escolhido aqui
(declarar no artigo): profundidade fixa d=5; selecao por torneio binario; cruzamento de um
ponto entre listas de camadas; mutacao = trocar uma camada por outra sorteada; elites mantem a
aptidao ja calculada; 30 geracoes; 100 repeticoes.

Tres avaliadores sobre os mesmos operadores geneticos:
  quantico  : G_qu simulado em vetor de estado exato (numpy) e amostrado com 1024 shots;
  classico  : G_cl (deterministico; "entropia zero" no artigo);
  gemeo     : G_qu com H trocada por uma moeda classica justa no bit, Y por X e as fases
              (Z, S, T) removidas. E um algoritmo classico aleatorio com a mesma estrutura.
Dois dominios: centrado [-5.12, 5.12] (otimo no centro) e deslocado [-2.12, 8.12] (otimo
x=0 longe do centro, que fica em x=3).

Uso (WSL): externos/QFIE/.venv/bin/python scripts/qga/vies_centro.py
Saida: resultados/qga/vies_centro-<data>.txt
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
BIT = [((IDX >> q) & 1).astype(bool) for q in range(NQ)]  # qubit q = bit q do indice
G_CL = [("i", 1), ("x", 1), ("cx", 2), ("swap", 2), ("ccx", 3), ("cswap", 3)]
G_QU = G_CL + [("h", 1), ("y", 1), ("z", 1), ("t", 1), ("tdg", 1), ("s", 1), ("sdg", 1)]
FASE = {"z": -1, "s": 1j, "sdg": -1j, "t": np.exp(1j * np.pi / 4), "tdg": np.exp(-1j * np.pi / 4)}


def rastrigin(x):
    x = np.asarray(x)
    return 10 * len(x) + float(np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


# ---------------- circuitos ----------------
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
    """Indice de origem para portas de permutacao (todas involucoes)."""
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
        for j in range(NB):  # b_1 (mais significativo) = qubit r*NB + 0
            z = (z << 1) | ((idx >> (r * NB + j)) & 1)
        xs.append(a + z / (2 ** NB - 1) * (b - a))
    return np.stack(xs, axis=1)


def amostra_quantico(ind, rng):
    p = np.abs(estado(ind)) ** 2
    return rng.choice(DIM, size=SHOTS, p=p / p.sum())


def amostra_bits(ind, rng, gemeo):
    """Classico (G_cl) ou gemeo classico de G_qu, shot a shot sobre bits."""
    bits = np.zeros((SHOTS, NQ), dtype=bool)
    for q in range(NQ):
        bits[:, q] = (ind["base"] >> q) & 1
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
            # z, s, sdg, t, tdg: so fase, sem efeito na base computacional
    return (bits.astype(np.int64) << np.arange(NQ)).sum(axis=1)


def aptidao(ind, modo, dominio, rng):
    if modo == "quantico":
        idx = amostra_quantico(ind, rng)
    else:
        idx = amostra_bits(ind, rng, gemeo=(modo == "gemeo"))
    xstar = decodifica(idx, dominio).mean(axis=0)
    return rastrigin(xstar)


# ---------------- algoritmo genetico ----------------
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


# ---------------- checagens ----------------
def checagens():
    rng = np.random.default_rng(SEMENTE)
    linhas = []
    # 1) simulador: circuitos so de permutacao batem com o avaliador classico
    erros = 0
    for _ in range(50):
        ind = individuo(rng, G_CL)
        psi = estado(ind)
        k = int(np.argmax(np.abs(psi)))
        erros += int(abs(abs(psi[k]) - 1) > 1e-12 or k != int(amostra_bits(ind, rng, False)[0]))
    linhas.append(f"checagem 1: 50 circuitos G_cl, vetor de estado = bits classicos? erros = {erros}")
    # 1b) simulador proprio contra o Statevector do Qiskit em circuitos G_qu sorteados
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
    linhas.append(f"checagem 1b: 20 circuitos G_qu, max|psi_proprio - psi_qiskit| = {dif:.2e}")
    # 2) H em todos os qubits: media decodificada exata = centro do dominio
    ind = {"base": 0, "cams": [[("h", (q,)) for q in range(NQ)]]}
    p = np.abs(estado(ind)) ** 2
    for dom in ((-5.12, 5.12), (-2.12, 8.12)):
        media = (decodifica(IDX, dom) * p[:, None]).sum(axis=0)
        linhas.append(f"checagem 2: H em todos, dominio {dom}: media exata = {media.round(12)}, "
                      f"f(media) = {rastrigin(media):.4f}")
    # 3) par maximamente emaranhado entre dois registros de 8 qubits: marginal uniforme
    ind = {"base": 0, "cams": [[("h", (q,)) for q in range(NB)], [("cx", (q, q + NB)) for q in range(NB)]]}
    p = (np.abs(estado(ind)) ** 2).reshape(2 ** NB, 2 ** NB)  # [registro 2, registro 1]
    marg = p.sum(axis=0)
    linhas.append(f"checagem 3: par emaranhado, marginal de um registro: max|p - 1/256| = "
                  f"{np.abs(marg - 1 / 256).max():.2e}")
    # 4) gerao 0: f(x*) de individuos iniciais sorteados, por avaliador e dominio
    for dom in ((-5.12, 5.12), (-2.12, 8.12)):
        for modo in ("quantico", "gemeo", "classico"):
            conj = G_CL if modo == "classico" else G_QU
            fs = [aptidao(individuo(rng, conj), modo, dom, rng) for _ in range(300)]
            linhas.append(f"checagem 4: geracao 0, dominio {dom}, {modo:8s}: f mediana {np.median(fs):7.3f} "
                          f"| fracao com f < 1: {np.mean(np.array(fs) < 1):.3f}")
    return linhas


def main():
    linhas = [f"Vies de centro em Souza/Dardenne/Portugal (reimplementacao) | semente {SEMENTE} | "
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
    linhas.append("melhor f por geracao (media sobre repeticoes; entre parenteses, erro-padrao)")
    gens = (0, 1, 2, 5, 10, 20, 30)
    linhas.append("dominio avaliador | " + " | ".join(f"g{g}" for g in gens))
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
