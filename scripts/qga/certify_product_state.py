"""Structural certificate of Pattern 2 (product state): the circuit is equivalent to drawing independent bits (14 Nov 2026).

Lemma (project v2, generalized): if the state before the rotations is a product state, each qubit
receives arbitrary one-qubit gates while it has not been coupled to another one, and every coupling
(gate on two or more qubits) and every later gate on a coupled qubit is MONOMIAL (basis permutation
times phases), then the measured distribution is that of independent bits y_i ~ Bernoulli(p_i) pushed
forward by a Boolean function pi(y). The state may become entangled (CNOT on a superposition), but the
entanglement does not show up in a measurement in the computational basis. Holds for any angles and any
number of qubits.

The checker:
  (1) prefix: longest initial stretch of gates in the ring Z[zeta_16] (H, X, CNOT, MCX, phases k*pi/8),
      run on the exact sparse engine; decides EXACTLY whether the state is a product state (rank 1 in
      every one-qubit cut, via 2x2 minors in the ring);
  (2) rest: rules above, by gate name (list of monomial gates), descending into the definitions;
  (3) measurement only at the end.
Cross-check: for small n and randomly drawn angles, the distribution of the classical sampler built
by the checker is compared with the Qiskit statevector (TV). Negative controls must fail.

Circuits of the cases RECONSTRUCTED from the reading notes
(texto/busca-bases/registro/qga-artigo3-2026-11-11/leitura.md), not from the authors' code (none of
them publishes code). Check each reconstruction against the figures before the article.

Usage: python scripts/qga/certify_product_state.py
Output: results/qga/certify_product_state-<data>.txt
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "qcalc"))
from sparse_engine import MotorEsparso, mult  # noqa: E402
from qiskit_convert import converte  # noqa: E402

RAIZ = AQUI.parents[1]
SEMENTE = 20261119

FLIP1 = {"x", "y"}
DIAG1 = {"z", "s", "sdg", "t", "tdg", "rz", "p", "u1", "id"}
PERM = {"cx", "cy", "ccx", "mcx", "mcx_gray", "mcx_recursive", "mcx_vchain", "c3x", "c4x", "swap", "cswap"}
DIAGM = {"cz", "ccz", "cp", "crz", "cu1", "rzz", "mcphase", "mcp"}
IGNORA = {"barrier", "delay", "global_phase"}


def portas_planas(qc):
    """List of (name, qubits, operation), descending into the definitions of gates not in the lists."""
    out = []

    def anda(circ, mapa):
        for ins in circ.data:
            op = ins.operation
            qs = [mapa[circ.find_bit(q).index] for q in ins.qubits]
            nm = op.name
            if nm in IGNORA:
                continue
            if nm == "measure" or nm in FLIP1 | DIAG1 | PERM | DIAGM or len(qs) == 1 or op.definition is None:
                out.append((nm, qs, op))
            else:
                anda(op.definition, qs)
    anda(qc, list(range(qc.num_qubits)))
    return out


def produto_exato(m, n):
    """Is the sparse-engine state a product state? Rank 1 in every cut q | rest, via 2x2 minors in the ring."""
    for q in range(n):
        b = 1 << q
        zero = {x: u for x, u in m.s.items() if not x & b}
        um = {x ^ b: u for x, u in m.s.items() if x & b}
        if not zero or not um:
            continue
        if set(zero) != set(um):
            return False
        chaves = list(zero)
        a0, a1 = zero[chaves[0]], um[chaves[0]]
        for c in chaves[1:]:
            if mult(a0, um[c]) != mult(a1, zero[c]):
                return False
    return True


def classifica(qc):
    """Returns (verdict, detail, plan). plan is used by the classical sampler."""
    n = qc.num_qubits
    gs = portas_planas(qc)
    # (1) prefix in the ring
    i = 0
    m = MotorEsparso(n)
    while i < len(gs) and gs[i][0] != "measure":
        sub = QuantumCircuit(n)
        sub.append(gs[i][2], gs[i][1])
        conv = converte(sub, rota_exata=True)
        if conv is None:
            break
        for p in conv:
            m.aplica(p)
        i += 1
    prefixo = i
    if not produto_exato(m, n):
        return False, f"exact-prefix state ({prefixo} gates) is not a product state", None
    # (2) rest. States per qubit: basic (only monomial gates so far: definite basis value),
    # coupled (has already taken part in a gate on two or more qubits), locked (witness used).
    # Rule R2 (deferred measurement): a non-monomial gate on a coupled qubit q is accepted only if there
    # is a copy of the value of q in another qubit w (made by a CNOT with w still basic) and w never
    # receives a gate again. Then the branches q = 0 and q = 1 are orthogonal in w and do not interfere:
    # this is equivalent to measuring q and preparing it again.
    basico = {q: len(m.marginal([q])) == 1 for q in range(n)}
    classe = {q: {q} for q in range(n)}
    acoplado, travado, medido = set(), set(), set()
    usos_r2 = 0

    def sai_da_classe(q):
        classe[q].discard(q)
        classe[q] = {q}

    for j in range(prefixo, len(gs)):
        nm, qs, op = gs[j]
        if nm == "measure":
            medido.add(qs[0])
            continue
        if any(q in medido for q in qs):
            return False, f"gate {nm} after measurement", None
        if any(q in travado for q in qs):
            return False, f"gate {nm} on an already used witness (gate {j})", None
        if len(qs) == 1:
            q = qs[0]
            if nm in FLIP1 | DIAG1:
                continue
            basico[q] = False
            if q in acoplado:
                outros = classe[q] - {q}
                if not outros:
                    return False, f"gate non-monomial {nm} on a coupled qubit without a copy (gate {j})", None
                travado.update(outros)
                sai_da_classe(q)
                acoplado.discard(q)
                usos_r2 += 1
        else:
            if nm not in PERM | DIAGM:
                return False, f"coupling not monomial {nm} (gate {j})", None
            if nm == "cx" and basico[qs[1]] and qs[1] not in acoplado:
                c, t = qs
                sai_da_classe(t)
                classe[c].add(t)
                classe[t] = classe[c]
            elif nm in PERM:
                for t in (qs[-2:] if "swap" in nm else qs[-1:]):
                    sai_da_classe(t)
            for q in qs:
                basico[q] = False
            acoplado.update(qs)
    return True, (f"exact prefix {prefixo} gates, product; {len(gs) - prefixo} gates after; "
                  f"{len(acoplado)} coupled qubits; rule R2 used {usos_r2} times"), (gs, prefixo)


def _aplica_perm(bits, nm, qs):
    if nm in ("x", "y"):
        bits[qs[0]] ^= 1
    elif nm == "swap" or (nm == "cswap" and bits[qs[0]]):
        a, b = qs[-2:]
        bits[a], bits[b] = bits[b], bits[a]
    elif nm not in ("swap", "cswap") and all(bits[c] for c in qs[:-1]):
        bits[qs[-1]] ^= 1


def amostrador(qc, plano):
    """Exact distribution of the equivalent classical program (float), by enumeration of branches.

    A 'fresh' qubit holds a vector of 2 amplitudes (local state); when it enters a gate on several
    qubits it is drawn (branch 0 and branch 1) and becomes a classical bit. A non-monomial gate on a
    classical bit (only happens via R2) turns it back into a fresh qubit with vector U|b>.
    """
    gs, prefixo = plano
    n = qc.num_qubits
    pre = QuantumCircuit(n)
    for nm, qs, op in gs[:prefixo]:
        pre.append(op, qs)
    psi = Statevector(pre).data.reshape([2] * n)  # axis 0 = qubit n-1
    vet0 = {}
    for q in range(n):
        mat = np.moveaxis(psi, n - 1 - q, 0).reshape(2, -1)
        v = mat[:, np.argmax(np.abs(mat).sum(axis=0))]
        vet0[q] = v / np.linalg.norm(v)
    ramos = [(1.0, {}, vet0)]

    def sorteia(ramos, q):
        out = []
        for pr, bits, vet in ramos:
            if q not in vet:
                out.append((pr, bits, vet))
                continue
            for b in (0, 1):
                pb = abs(vet[q][b]) ** 2
                if pb > 1e-15:
                    v2 = dict(vet)
                    del v2[q]
                    out.append((pr * pb, {**bits, q: b}, v2))
        return out

    for nm, qs, op in gs[prefixo:]:
        if nm == "measure":
            continue
        if len(qs) == 1:
            q, U = qs[0], op.to_matrix()
            novos = []
            for pr, bits, vet in ramos:
                if q in vet:
                    novos.append((pr, bits, {**vet, q: U @ vet[q]}))
                elif nm in FLIP1 | DIAG1:
                    b2 = dict(bits)
                    if nm in FLIP1:
                        _aplica_perm(b2, nm, qs)
                    novos.append((pr, b2, vet))
                else:
                    e = np.zeros(2, complex)
                    e[bits[q]] = 1
                    b2 = dict(bits)
                    del b2[q]
                    novos.append((pr, b2, {**vet, q: U @ e}))
            ramos = novos
            continue
        for q in qs:
            ramos = sorteia(ramos, q)
        if nm in PERM:
            novos = []
            for pr, bits, vet in ramos:
                b2 = dict(bits)
                _aplica_perm(b2, nm, qs)
                novos.append((pr, b2, vet))
            ramos = novos
    for q in range(n):
        ramos = sorteia(ramos, q)
    dist = np.zeros(2 ** n)
    for pr, bits, _ in ramos:
        dist[sum(b << q for q, b in bits.items())] += pr
    return dist


# ---- reconstructed cases ----
def ballinas(n, th):
    """Ballinas and Montiel 2022, Fig. 2 (checked on 14 Nov): X on the ancilla; H on all; CNOT from each
    data qubit to the ancilla; H on the data only; Ry on all, including the ancilla; measures the data.
    Ballinas et al. 2024 says it uses the Deutsch-Jozsa circuit; figure not yet checked."""
    qc = QuantumCircuit(n + 1, n)
    qc.x(n)
    qc.h(range(n + 1))
    for i in range(n):
        qc.cx(i, n)
    qc.h(range(n))
    for i in range(n + 1):
        qc.ry(th[i % n], i)
    qc.measure(range(n), range(n))
    return qc


def ballinas2024(n, th, leitura):
    """Ballinas et al. 2024 (Axioms 13:48), Fig. 4 (checked on 14 Nov in PDF v3). The figure shows
    H on all, CNOT from each data qubit to the ancilla, H on the data, Ry on all, measures the data,
    WITHOUT the X gate on the ancilla. The text says that the ancilla must be in |1> and that it CONTROLS
    the CNOTs. Readings: 'figura' (no X, ancilla is target), 'texto_x' (with X, ancilla is target),
    'texto_controle' (with X, ancilla is control)."""
    qc = QuantumCircuit(n + 1, n)
    if leitura != "figura":
        qc.x(n)
    qc.h(range(n + 1))
    for i in range(n):
        if leitura == "texto_controle":
            qc.cx(n, i)
        else:
            qc.cx(i, n)
    qc.h(range(n))
    for i in range(n + 1):
        qc.ry(th[i % n], i)
    qc.measure(range(n), range(n))
    return qc


def _pares_eaqga(n, rng, cadeia):
    """Pairs (control, target). Star: a chain has one control and several targets (text of Sec. III-B,
    Fig. 4). Chain: the target of one pair may be the control of the next (allowed by the pseudocode)."""
    ordem = [int(x) for x in rng.permutation(n)]
    pares = []
    while len(ordem) >= 2:
        k = int(rng.integers(2, min(4, len(ordem)) + 1))
        grupo, ordem = ordem[:k], ordem[k:]
        if cadeia:
            pares += [(grupo[i], grupo[i + 1]) for i in range(k - 1)]
        else:
            pares += [(grupo[0], t) for t in grupo[1:]]
    return pares, ordem


def eaqga(n, th, rng, cadeia=False):
    """EAQGA, Algorithm 1 as printed (v1, checked on 14 Nov): for each pair, RY on the control,
    X on the target if the parent's bit at the control is 1, CNOT; unpaired qubits receive one RY; measures all.
    The RY is repeated on the control for each pair, so it may come after a CNOT on it (rule R2)."""
    qc = QuantumCircuit(n, n)
    pares, soltos = _pares_eaqga(n, rng, cadeia)
    pai = rng.integers(0, 2, n)
    for c, t in pares:
        qc.ry(th[c], c)
        if pai[c]:
            qc.x(t)
        qc.cx(c, t)
    for q in soltos:
        qc.ry(th[q], q)
    qc.measure(range(n), range(n))
    return qc


def eaqga_estrela(n, th, rng):
    return eaqga(n, th, rng, cadeia=False)


def eaqga_cadeia(n, th, rng):
    return eaqga(n, th, rng, cadeia=True)


def gamberi(n, th):
    """Gamberi and Bianchini 2026, Algorithm 3 (v1, checked on 14 Nov): uniform superposition in the
    initial generation (gate not named; H here), Y-rotations per qubit, measures. No 2-qubit gate."""
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for i in range(n):
        qc.ry(th[i], i)
    qc.measure(range(n), range(n))
    return qc


def saini(n, th):
    """Saini et al. 2025, IEEE Access, Fig. 3 (checked on 14 Nov on IEEE Xplore): RX per qubit and
    measurement; Fig. 5 (transpiled on ibm_nairobi) has only RZ and square root of X per qubit, no 2-qubit gate."""
    qc = QuantumCircuit(n, n)
    for i in range(n):
        qc.rx(th[i], i)
    qc.measure(range(n), range(n))
    return qc


# ---- negative controls ----
def neg_interferencia(n, th):
    qc = QuantumCircuit(n, n)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(range(n), range(n))
    return qc


def neg_qaoa(n, th):
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for i in range(n - 1):
        qc.rzz(th[i], i, i + 1)
    for i in range(n):
        qc.rx(th[i], i)
    qc.measure(range(n), range(n))
    return qc


def neg_prefixo(n, th):
    qc = QuantumCircuit(n, n)
    qc.h(0)
    qc.cx(0, 1)  # Bell state in the exact prefix, then rotation
    qc.ry(th[0], 0)
    qc.measure(range(n), range(n))
    return qc


def neg_r2_alvo_girado(n, th):
    """Copy made on a target that was already in superposition: it is not a witness."""
    qc = QuantumCircuit(n, n)
    qc.ry(th[0], 0)
    qc.ry(th[1], 1)
    qc.cx(0, 1)
    qc.ry(th[2], 0)
    qc.measure(range(n), range(n))
    return qc


def neg_r2_testemunha_mexida(n, th):
    """Good copy, but the witness receives H before the RY on the control."""
    qc = QuantumCircuit(n, n)
    qc.ry(th[0], 0)
    qc.cx(0, 1)
    qc.h(1)
    qc.ry(th[2], 0)
    qc.measure(range(n), range(n))
    return qc


def main():
    rng = np.random.default_rng(SEMENTE)
    L = [f"Structural certificate of Pattern 2 (product state) (classical sampling) | seed {SEMENTE} | {date.today()}",
         "circuits checked on 14 Nov against Ballinas 2022 Fig. 2, Ballinas 2024 Fig. 4 and text, EAQGA v1 Alg. 1, Gamberi v1 Alg. 3, Saini Figs. 3 and 5; cross-check: TV(sampler, statevector) with randomly drawn angles", ""]
    casos = [("Ballinas 2022 (Fig. 2)", ballinas, True),
             ("Ballinas 2024 (Fig. 4, no X)", lambda n, th: ballinas2024(n, th, "figura"), True),
             ("Ballinas 2024 (text: X on the ancilla)", lambda n, th: ballinas2024(n, th, "texto_x"), True),
             ("Ballinas 2024 (text: ancilla controls)", lambda n, th: ballinas2024(n, th, "texto_controle"), True),
             ("EAQGA, star (text)", lambda n, th: eaqga_estrela(n, th, rng), True),
             ("EAQGA, chain (pseudocode)", lambda n, th: eaqga_cadeia(n, th, rng), True),
             ("Gamberi 2026 (Algorithm 3)", gamberi, True),
             ("Saini 2025 (Fig. 3)", saini, True),
             ("control: R2 with target in superposition", neg_r2_alvo_girado, False),
             ("control: R2 with disturbed witness", neg_r2_testemunha_mexida, False),
             ("control: H-CNOT-H", neg_interferencia, False),
             ("control: QAOA layer", neg_qaoa, False),
             ("control: Bell state in the prefix", neg_prefixo, False)]
    tudo = True
    for nome, f, esperado in casos:
        tvs, vered = [], set()
        for n in (3, 5, 8):
            for _ in range(20):
                th = rng.uniform(0, 2 * np.pi, n)
                qc = f(n, th)
                ok, det, plano = classifica(qc)
                vered.add((ok, det if not ok else det.split(";")[0]))
                if ok:
                    sv = Statevector(qc.remove_final_measurements(inplace=False)).probabilities()
                    tvs.append(0.5 * np.abs(amostrador(qc, plano) - sv).sum())
        oks = {v[0] for v in vered}
        certo = oks == {esperado}
        tudo &= certo and (not tvs or max(tvs) < 1e-12)
        L.append(f"{nome}: classified as classical sampling = {sorted(oks)} (expected {esperado}) | "
                 + (f"TV max {max(tvs):.1e} over {len(tvs)} circuits" if tvs else "no sampler")
                 + f" | {'OK' if certo else 'FAIL'}")
        L.append("   " + "; ".join(sorted({v[1] for v in vered}))[:300])
    # scale: EAQGA with 100 qubits (the case the authors say is intractable to simulate)
    t0 = time.time()
    qc = eaqga_estrela(100, rng.uniform(0, 2 * np.pi, 100), rng)
    ok, det, _ = classifica(qc)
    L.append("")
    L.append(f"EAQGA with 100 qubits: classical sampling = {ok} ({det}) | {time.time() - t0:.2f}s")
    qc = ballinas(16, rng.uniform(0, 2 * np.pi, 16))  # exact prefix: support 2^17 after the H layer
    t0 = time.time()
    ok, det, _ = classifica(qc)
    L.append(f"Ballinas with 16 data qubits: classical sampling = {ok} ({det}) | {time.time() - t0:.2f}s")
    tudo &= ok
    L.append(f"\nall as expected: {tudo}")
    print("\n".join(L))
    (RAIZ / "results" / "qga" / f"certify_product_state-{date.today()}.txt").write_text("\n".join(L) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
