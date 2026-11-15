"""Certificado estrutural do padrao 1: o circuito equivale a sortear bits independentes (14/11/2026).

Lema (projeto v2, generalizado): se o estado antes das rotacoes e produto, cada qubit recebe portas
de um qubit quaisquer enquanto nao foi acoplado a outro, e todo acoplamento (porta de dois ou mais
qubits) e toda porta posterior num qubit acoplado sao MONOMIAIS (permutacao da base vezes fases),
entao a distribuicao medida e a de bits independentes y_i ~ Bernoulli(p_i) empurrada por uma funcao
booleana pi(y). O estado pode ficar emaranhado (CNOT sobre superposicao), mas o emaranhamento nao
aparece na medicao na base computacional. Vale para quaisquer angulos e qualquer numero de qubits.

O verificador:
  (1) prefixo: maior trecho inicial com portas do anel Z[zeta_16] (H, X, CNOT, MCX, fases k*pi/8),
      executado no motor exato esparso; decide EXATAMENTE se o estado e produto (posto 1 em cada
      corte de um qubit, por menores 2x2 no anel);
  (2) resto: regras acima, por nome de porta (lista de portas monomiais), descendo nas definicoes;
  (3) medicao so no fim.
Conferencia: para n pequeno e angulos sorteados, a distribuicao do amostrador classico construido
pelo verificador e comparada com o statevector do Qiskit (TV). Controles negativos devem falhar.

Circuitos dos casos RECONSTRUIDOS a partir das notas de leitura
(texto/busca-bases/registro/qga-artigo3-2026-11-11/leitura.md), nao do codigo dos autores (nenhum
publica codigo). Conferir cada reconstrucao contra as figuras antes do artigo.

Uso (WSL): externos/mqtbench/.venv/bin/python scripts/qga/certifica_padrao1.py
Saida: resultados/qga/certifica_padrao1-<data>.txt
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
from esparso import MotorEsparso, mult  # noqa: E402
from converte import converte  # noqa: E402

RAIZ = AQUI.parents[1]
SEMENTE = 20261119

FLIP1 = {"x", "y"}
DIAG1 = {"z", "s", "sdg", "t", "tdg", "rz", "p", "u1", "id"}
PERM = {"cx", "cy", "ccx", "mcx", "mcx_gray", "mcx_recursive", "mcx_vchain", "c3x", "c4x", "swap", "cswap"}
DIAGM = {"cz", "ccz", "cp", "crz", "cu1", "rzz", "mcphase", "mcp"}
IGNORA = {"barrier", "delay", "global_phase"}


def portas_planas(qc):
    """Lista (nome, qubits, operacao), descendo nas definicoes de portas fora das listas."""
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
    """Estado do motor esparso e produto? Posto 1 em cada corte q | resto, por menores 2x2 no anel."""
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
    """Devolve (veredito, detalhe, plano). plano serve ao amostrador classico."""
    n = qc.num_qubits
    gs = portas_planas(qc)
    # (1) prefixo no anel
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
        return False, f"estado apos o prefixo exato ({prefixo} portas) nao e produto", None
    # (2) resto. Estados por qubit: basico (so portas monomiais ate aqui: valor de base definido),
    # acoplado (ja participou de porta de dois ou mais qubits), travado (testemunha usada).
    # Regra R2 (medicao adiada): porta nao monomial num qubit acoplado q so e aceita se ha copia
    # do valor de q noutro qubit w (feita por CNOT com w ainda basico) e w nunca mais recebe porta.
    # Entao os ramos q = 0 e q = 1 ficam ortogonais em w e nao interferem: equivale a medir q e
    # prepara-lo de novo.
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
            return False, f"porta {nm} depois de medir", None
        if any(q in travado for q in qs):
            return False, f"porta {nm} numa testemunha ja usada (porta {j})", None
        if len(qs) == 1:
            q = qs[0]
            if nm in FLIP1 | DIAG1:
                continue
            basico[q] = False
            if q in acoplado:
                outros = classe[q] - {q}
                if not outros:
                    return False, f"porta nao monomial {nm} num qubit acoplado sem copia (porta {j})", None
                travado.update(outros)
                sai_da_classe(q)
                acoplado.discard(q)
                usos_r2 += 1
        else:
            if nm not in PERM | DIAGM:
                return False, f"acoplamento nao monomial {nm} (porta {j})", None
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
    return True, (f"prefixo exato {prefixo} portas, produto; {len(gs) - prefixo} portas depois; "
                  f"{len(acoplado)} qubits acoplados; regra R2 usada {usos_r2} vezes"), (gs, prefixo)


def _aplica_perm(bits, nm, qs):
    if nm in ("x", "y"):
        bits[qs[0]] ^= 1
    elif nm == "swap" or (nm == "cswap" and bits[qs[0]]):
        a, b = qs[-2:]
        bits[a], bits[b] = bits[b], bits[a]
    elif nm not in ("swap", "cswap") and all(bits[c] for c in qs[:-1]):
        bits[qs[-1]] ^= 1


def amostrador(qc, plano):
    """Distribuicao exata do programa classico equivalente (float), por enumeracao de ramos.

    Qubit 'fresco' guarda um vetor de 2 amplitudes (estado local); ao entrar numa porta de varios
    qubits ele e sorteado (ramo 0 e ramo 1) e vira bit classico. Porta nao monomial num bit
    classico (so ocorre via R2) o transforma de novo em qubit fresco com vetor U|b>.
    """
    gs, prefixo = plano
    n = qc.num_qubits
    pre = QuantumCircuit(n)
    for nm, qs, op in gs[:prefixo]:
        pre.append(op, qs)
    psi = Statevector(pre).data.reshape([2] * n)  # eixo 0 = qubit n-1
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


# ---- casos reconstruidos ----
def ballinas(n, th):
    """Ballinas e Montiel 2022, Fig. 2 (conferida em 14/11): X na ancila; H em todos; CNOT de cada
    qubit de dados para a ancila; H so nos dados; Ry em todos, inclusive a ancila; mede os dados.
    Ballinas et al. 2024 diz usar o circuito de Deutsch-Jozsa; figura ainda nao conferida."""
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
    """Ballinas et al. 2024 (Axioms 13:48), Fig. 4 (conferida em 14/11 no PDF v3). A figura mostra
    H em todos, CNOT de cada qubit de dados para a ancila, H nos dados, Ry em todos, mede os dados,
    SEM a porta X na ancila. O texto diz que a ancila deve estar em |1> e que ela CONTROLA os CNOT.
    Leituras: 'figura' (sem X, ancila alvo), 'texto_x' (com X, ancila alvo), 'texto_controle'
    (com X, ancila controle)."""
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
    """Pares (controle, alvo). Estrela: uma cadeia tem um controle e varios alvos (texto da Sec. III-B,
    Fig. 4). Cadeia: o alvo de um par pode ser o controle do seguinte (permitido pelo pseudocodigo)."""
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
    """EAQGA, Algorithm 1 como impresso (v1, conferido em 14/11): para cada par, RY no controle,
    X no alvo se o bit do pai no controle for 1, CNOT; qubits sem par recebem um RY; mede tudo.
    O RY e repetido no controle a cada par, entao pode vir depois de um CNOT nele (regra R2)."""
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
    """Gamberi e Bianchini 2026, Algorithm 3 (v1, conferido em 14/11): superposicao uniforme na
    geracao inicial (porta nao nomeada; H aqui), Y-rotations por qubit, mede. Sem porta de 2 qubits."""
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for i in range(n):
        qc.ry(th[i], i)
    qc.measure(range(n), range(n))
    return qc


def saini(n, th):
    """Saini et al. 2025, IEEE Access, Fig. 3 (conferida em 14/11 no IEEE Xplore): RX por qubit e
    medicao; a Fig. 5 (transpilado no ibm_nairobi) so tem RZ e raiz de X por qubit, sem porta de 2 qubits."""
    qc = QuantumCircuit(n, n)
    for i in range(n):
        qc.rx(th[i], i)
    qc.measure(range(n), range(n))
    return qc


# ---- controles negativos ----
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
    qc.cx(0, 1)  # Bell no prefixo exato, depois rotacao
    qc.ry(th[0], 0)
    qc.measure(range(n), range(n))
    return qc


def neg_r2_alvo_girado(n, th):
    """Copia feita num alvo que ja estava em superposicao: nao e testemunha."""
    qc = QuantumCircuit(n, n)
    qc.ry(th[0], 0)
    qc.ry(th[1], 1)
    qc.cx(0, 1)
    qc.ry(th[2], 0)
    qc.measure(range(n), range(n))
    return qc


def neg_r2_testemunha_mexida(n, th):
    """Copia boa, mas a testemunha recebe H antes do RY no controle."""
    qc = QuantumCircuit(n, n)
    qc.ry(th[0], 0)
    qc.cx(0, 1)
    qc.h(1)
    qc.ry(th[2], 0)
    qc.measure(range(n), range(n))
    return qc


def main():
    rng = np.random.default_rng(SEMENTE)
    L = [f"Certificado estrutural do padrao 1 (amostragem classica) | semente {SEMENTE} | {date.today()}",
         "circuitos conferidos em 14/11 contra Ballinas 2022 Fig. 2, Ballinas 2024 Fig. 4 e texto, EAQGA v1 Alg. 1, Gamberi v1 Alg. 3, Saini Figs. 3 e 5; conferencia: TV(amostrador, statevector) com angulos sorteados", ""]
    casos = [("Ballinas 2022 (Fig. 2)", ballinas, True),
             ("Ballinas 2024 (Fig. 4, sem X)", lambda n, th: ballinas2024(n, th, "figura"), True),
             ("Ballinas 2024 (texto: X na ancila)", lambda n, th: ballinas2024(n, th, "texto_x"), True),
             ("Ballinas 2024 (texto: ancila controla)", lambda n, th: ballinas2024(n, th, "texto_controle"), True),
             ("EAQGA, estrela (texto)", lambda n, th: eaqga_estrela(n, th, rng), True),
             ("EAQGA, cadeia (pseudocodigo)", lambda n, th: eaqga_cadeia(n, th, rng), True),
             ("Gamberi 2026 (Algorithm 3)", gamberi, True),
             ("Saini 2025 (Fig. 3)", saini, True),
             ("controle: R2 com alvo em superposicao", neg_r2_alvo_girado, False),
             ("controle: R2 com testemunha mexida", neg_r2_testemunha_mexida, False),
             ("controle: H-CNOT-H", neg_interferencia, False),
             ("controle: camada QAOA", neg_qaoa, False),
             ("controle: Bell no prefixo", neg_prefixo, False)]
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
        L.append(f"{nome}: classifica como amostragem classica = {sorted(oks)} (esperado {esperado}) | "
                 + (f"TV max {max(tvs):.1e} em {len(tvs)} circuitos" if tvs else "sem amostrador")
                 + f" | {'OK' if certo else 'FALHA'}")
        L.append("   " + "; ".join(sorted({v[1] for v in vered}))[:300])
    # escala: EAQGA com 100 qubits (o caso que os autores dizem intratavel de simular)
    t0 = time.time()
    qc = eaqga_estrela(100, rng.uniform(0, 2 * np.pi, 100), rng)
    ok, det, _ = classifica(qc)
    L.append("")
    L.append(f"EAQGA com 100 qubits: amostragem classica = {ok} ({det}) | {time.time() - t0:.2f}s")
    qc = ballinas(16, rng.uniform(0, 2 * np.pi, 16))  # prefixo exato: suporte 2^17 apos a camada de H
    t0 = time.time()
    ok, det, _ = classifica(qc)
    L.append(f"Ballinas com 16 qubits de dados: amostragem classica = {ok} ({det}) | {time.time() - t0:.2f}s")
    tudo &= ok
    L.append(f"\ntodos conforme o esperado: {tudo}")
    print("\n".join(L))
    (RAIZ / "resultados" / "qga" / f"certifica_padrao1-{date.today()}.txt").write_text("\n".join(L) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
