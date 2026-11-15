"""Exact certification of entanglement in Souza, Dardenne and Portugal (13 Nov 2026).

Question: are the individuals selected by the quantum GA (reimplementation in vies_centro.py)
actually entangled, and can the entanglement influence the fitness?

Analytical remark (to be checked against their article): the fitness is f(x*), x* = mean of the
decoded values over the shots. Hence f depends on the circuit only through the measured
distribution p(x) in the computational basis; the expected value of x* depends only on the
one-qubit marginals. An entangled state and a classical mixture with the same p(x) give exactly
the same fitness distribution.

For each individual (best of the final generation of the quantum GA, in both domains, and
randomly drawn initial individuals), in the exact sparse engine over Z[zeta_16]:
  (1) separable between the registers of the two variables (qubits 0-7 | 8-15)? Exact rank-1
      test: M[i,j] M[i0,j0] == M[i,j0] M[i0,j] in the ring for all i, j;
  (2) Schmidt rank across that cut: certified lower bound (rank modulo primes p = 1 mod 16,
      zeta -> primitive 16th root of unity in F_p; reduction never increases the rank) and the
      numerical value by SVD, for reference only;
  (3) how many qubits are entangled with the rest (exact rank-1 test on each one-qubit cut);
  (4) does p(x) factor between the registers? P(x1,x2) == P1(x1) P2(x2), exactly;
  (5) do the exact one-qubit marginals belong to {0, 1/2, 1}?

Usage (WSL): externos/QFIE/.venv/bin/python scripts/qga/certifica_souza.py
Output: resultados/qga/certifica_souza-<date>.txt
"""
from __future__ import annotations

import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "qcalc"))
import vies_centro as VC  # noqa: E402
from esparso import MotorEsparso, conj, mult  # noqa: E402

RAIZ = AQUI.parents[1]
SEMENTE = 20261113
REPS = 20
FASE_M = {"z": 8, "s": 4, "sdg": 12, "t": 2, "tdg": 14}
ZERO = (0,) * 8


# ---------------- circuit -> exact engine ----------------
def portas_exatas(ind):
    out = [("mcx", [], q) for q in range(VC.NQ) if (ind["base"] >> q) & 1]
    for cam in ind["cams"]:
        for nome, q in cam:
            if nome == "h":
                out.append(("h", q[0]))
            elif nome == "x":
                out.append(("mcx", [], q[0]))
            elif nome == "y":  # Y = i X Z
                out += [("fase", 8, [q[0]]), ("mcx", [], q[0]), ("fase", 4, [])]
            elif nome in FASE_M:
                out.append(("fase", FASE_M[nome], [q[0]]))
            elif nome == "cx":
                out.append(("mcx", [q[0]], q[1]))
            elif nome == "ccx":
                out.append(("mcx", [q[0], q[1]], q[2]))
            elif nome == "swap":
                a, b = q
                out += [("mcx", [a], b), ("mcx", [b], a), ("mcx", [a], b)]
            elif nome == "cswap":  # Fredkin = CX(b,a) CCX(c,a,b) CX(b,a)
                c, a, b = q
                out += [("mcx", [b], a), ("mcx", [c, a], b), ("mcx", [b], a)]
            else:
                raise ValueError(nome)
    return out


def estado_exato(ind):
    m = MotorEsparso(VC.NQ)
    for p in portas_exatas(ind):
        m.aplica(p)
    return m


# ---------------- exact tests ----------------
def posto1(amps, corte):
    """amps: {x: u}; corte: mask of the qubits on side A. Exactly rank 1?"""
    itens = list(amps.items())
    if not itens:
        return True
    x0, u0 = itens[0]
    a0, b0 = x0 & corte, x0 & ~corte
    for x, u in itens:
        a, b = x & corte, x & ~corte
        lhs = mult(u, u0)
        rhs = mult(amps.get(a | b0, ZERO), amps.get(a0 | b, ZERO))
        if lhs != rhs:
            return False
    # zero entries outside the support: M[a,b] = 0 requires M[a,b0] M[a0,b] = 0
    As = {x & corte for x in amps}
    Bs = {x & ~corte for x in amps}
    for a in As:
        if amps.get(a | b0, ZERO) == ZERO:
            continue
        for b in Bs:
            if (a | b) not in amps and amps.get(a0 | b, ZERO) != ZERO:
                return False
    return True


def primo_1mod16(perto):
    c = perto - (perto % 16) + 1
    while not VC_primo(c):
        c -= 16
    return c


def VC_primo(n):
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1 if d == 2 else 2
    return True


def raiz16(p):
    """Primitive 16th root of unity in F_p (p = 1 mod 16)."""
    for g in range(2, p):
        r = pow(g, (p - 1) // 16, p)
        if pow(r, 8, p) == p - 1:
            return r
    raise ValueError


PRIMOS = [primo_1mod16(2 ** 30), primo_1mod16(2 ** 29 + 12345)]
RAIZES = {p: raiz16(p) for p in PRIMOS}


def posto_mod(amps, lado_a, lado_b, p, r):
    pot = [pow(r, j, p) for j in range(8)]
    ia = {q: i for i, q in enumerate(lado_a)}
    ib = {q: i for i, q in enumerate(lado_b)}
    M = np.zeros((1 << len(lado_a), 1 << len(lado_b)), dtype=np.int64)
    for x, u in amps.items():
        a = sum(((x >> q) & 1) << ia[q] for q in lado_a)
        b = sum(((x >> q) & 1) << ib[q] for q in lado_b)
        M[a, b] = sum(c * w for c, w in zip(u, pot)) % p
    posto, lin = 0, 0
    M = M % p
    for col in range(M.shape[1]):
        piv = None
        for i in range(lin, M.shape[0]):
            if M[i, col]:
                piv = i
                break
        if piv is None:
            continue
        M[[lin, piv]] = M[[piv, lin]]
        inv = pow(int(M[lin, col]), p - 2, p)
        M[lin] = (M[lin] * inv) % p
        f = M[:, col].copy()
        f[lin] = 0
        M = (M - (f[:, None] * M[lin][None, :]) % p) % p
        lin += 1
        posto += 1
        if lin == M.shape[0]:
            break
    return posto


def posto_svd(amps, lado_a, lado_b, k):
    zeta = np.exp(1j * np.pi / 8) ** np.arange(8)
    ia = {q: i for i, q in enumerate(lado_a)}
    ib = {q: i for i, q in enumerate(lado_b)}
    M = np.zeros((1 << len(lado_a), 1 << len(lado_b)), dtype=complex)
    for x, u in amps.items():
        a = sum(((x >> q) & 1) << ia[q] for q in lado_a)
        b = sum(((x >> q) & 1) << ib[q] for q in lado_b)
        M[a, b] = np.dot(u, zeta) / np.sqrt(2) ** k
    s = np.linalg.svd(M, compute_uv=False)
    return int((s > 1e-10).sum())


def fatora(m, A, B):
    """P(x) == P_A(xA) P_B(xB) exactly."""
    pa = m.marginal(A)
    pb = m.marginal(B)
    pab = m.marginal(A + B)
    escala = (2 ** m.k,) + (0,) * 7
    for ya, na in pa.items():
        for yb, nb in pb.items():
            lhs = mult(pab.get(ya | (yb << len(A)), ZERO), escala)
            if lhs != mult(na, nb):
                return False
    return True


def marginais_meio(m):
    """Are all one-qubit marginals in {0, 1/2, 1}?"""
    ok = True
    for q in range(VC.NQ):
        mg = m.marginal([q])
        n1 = mg.get(1, ZERO)
        alvos = [ZERO, (2 ** (m.k - 1),) + (0,) * 7 if m.k >= 1 else None, (2 ** m.k,) + (0,) * 7]
        if n1 not in alvos:
            ok = False
    return ok


def certifica(args):
    rotulo, ind, f = args
    m = estado_exato(ind)
    amps = m.s
    A = list(range(VC.NB))
    B = list(range(VC.NB, VC.NQ))
    corte = sum(1 << q for q in A)
    sep = posto1(amps, corte)
    pinf = max(posto_mod(amps, A, B, p, RAIZES[p]) for p in PRIMOS)
    psvd = posto_svd(amps, A, B, m.k)
    emar = sum(not posto1(amps, 1 << q) for q in range(VC.NQ))
    return dict(rotulo=rotulo, f=f, k=m.k, suporte=len(amps), sep=sep, pinf=pinf, psvd=psvd, emar=emar,
                fatora=fatora(m, A, B), meio=marginais_meio(m))


# ---------------- GA that returns the best final individual ----------------
def roda_ga_ind(args):
    dominio, semente = args
    rng = np.random.default_rng(semente)
    conjunto = VC.G_QU
    pop = [VC.individuo(rng, conjunto) for _ in range(VC.POP)]
    fit = [VC.aptidao(i, "quantico", dominio, rng) for i in pop]
    n_elite = int(VC.P_ELITE * VC.POP)
    for _ in range(VC.GER):
        ordem = np.argsort(fit)
        nova = [pop[i] for i in ordem[:n_elite]]
        nfit = [fit[i] for i in ordem[:n_elite]]
        while len(nova) < VC.POP:
            a, b = rng.integers(VC.POP, size=2)
            p1 = pop[a] if fit[a] <= fit[b] else pop[b]
            a, b = rng.integers(VC.POP, size=2)
            p2 = pop[a] if fit[a] <= fit[b] else pop[b]
            cams = list(p1["cams"])
            if rng.random() < VC.P_CROSS:
                corte = int(rng.integers(1, VC.PROF))
                cams = list(p1["cams"][:corte]) + list(p2["cams"][corte:])
            if rng.random() < VC.P_MUT:
                cams[int(rng.integers(VC.PROF))] = VC.camada(rng, conjunto)
            filho = {"base": p1["base"], "cams": cams}
            nova.append(filho)
            nfit.append(VC.aptidao(filho, "quantico", dominio, rng))
        pop, fit = nova, nfit
    i = int(np.argmin(fit))
    return dominio, pop[i], fit[i]


def resumo(rotulo, rs):
    n = len(rs)
    sep = sum(r["sep"] for r in rs)
    fat = sum(r["fatora"] for r in rs)
    meio = sum(r["meio"] for r in rs)
    pinf = [r["pinf"] for r in rs]
    concorda = sum(r["pinf"] == r["psvd"] for r in rs)
    emar = [r["emar"] for r in rs]
    return (f"{rotulo} | n={n} | separable between registers: {sep} | Schmidt rank (certified bound) "
            f"median {int(np.median(pinf))}, max {max(pinf)} | bound mod p == SVD in {concorda}/{n} | "
            f"qubits entangled with the rest: median {int(np.median(emar))}, max {max(emar)} | "
            f"p(x) factors between registers: {fat}/{n} | marginals in {{0,1/2,1}}: {meio}/{n}")


def main():
    rng = np.random.default_rng(SEMENTE)
    linhas = [f"Souza/Dardenne/Portugal (reimplementation): exact certification of entanglement | "
              f"seed {SEMENTE} | {date.today()}", f"primes {PRIMOS}"]
    dominios = ((-5.12, 5.12), (-2.12, 8.12))
    tarefas = [(dom, SEMENTE + 1000 * r + k) for k, dom in enumerate(dominios) for r in range(REPS)]
    with Pool(16) as pool:
        finais = pool.map(roda_ga_ind, tarefas, chunksize=1)
        iniciais = [(f"inicial", VC.individuo(rng, VC.G_QU), float("nan")) for _ in range(100)]
        args = [(f"final {d}", ind, f) for d, ind, f in finais] + iniciais
        certs = pool.map(certifica, args, chunksize=1)
    for rot in [f"final {d}" for d in dominios] + ["inicial"]:
        rs = [c for c in certs if c["rotulo"] == rot]
        linhas.append(resumo(rot, rs))
        print(linhas[-1], flush=True)
    linhas.append("")
    linhas.append("best final individuals, one per line: domain | f | k | support | separable | rank >= | SVD rank | "
                  "entangled qubits | p factors | marginals in {0,1/2,1}")
    for c in certs:
        if c["rotulo"].startswith("final"):
            linhas.append(f"{c['rotulo']} | {c['f']:.4f} | {c['k']} | {c['suporte']} | {c['sep']} | {c['pinf']} | "
                          f"{c['psvd']} | {c['emar']} | {c['fatora']} | {c['meio']}")
    (RAIZ / "resultados" / "qga" / f"certifica_souza-{date.today()}.txt").write_text("\n".join(linhas) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
