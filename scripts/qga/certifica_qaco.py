"""Padrao 1 no QACO da propria QMI (14/11/2026).

Caso: "Implementable hybrid quantum ant colony optimization algorithm", Quantum Machine Intelligence,
2022, doi 10.1007/s42484-022-00065-1 (Secoes 3.1.1 e 3.1.2 lidas no site da Springer em 14/11).
Circuito reconstruido do texto:
  - qubits de formiga: RY(theta_i) a partir de |0> (feromonio), 0 < theta_i < pi;
  - exploracao: qubit com RY(theta_e), theta_e = 2 arcsin(sqrt(beta_e)), controla uma porta e e
    reiniciado (reset) depois de cada porta controlada; aqui o reset + novo RY e representado por um
    qubit de exploracao novo para cada porta (os autores tambem "duplicam" esse qubit);
  - sem restricao: CNOT(exploracao -> formiga) em cada formiga;
  - com restricao (numero fixo de 1s): Fredkin(exploracao; t1, t2) em pares de formigas;
  - mede as formigas.
Previsao: classificado como amostragem classica (Teorema 1); amostrador = statevector.

Uso (WSL): externos/mqtbench/.venv/bin/python scripts/qga/certifica_qaco.py
Saida: resultados/qga/certifica_qaco-<data>.txt
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import certifica_padrao1 as P1  # noqa: E402

RAIZ = AQUI.parents[1]
SEMENTE = 20261122


def qaco(n, th, beta, restrito, rng):
    pares = [(i, j) for i in range(n) for j in range(i + 1, n)][: max(1, n * (n - 1) // 2 - 1)] if restrito else []
    ne = len(pares) if restrito else n
    qc = QuantumCircuit(n + ne, n)
    for i in range(n):
        qc.ry(th[i], i)
    te = 2 * np.arcsin(np.sqrt(beta))
    for k in range(ne):
        e = n + k
        qc.ry(te, e)
        if restrito:
            a, b = pares[k]
            qc.cswap(e, a, b)
        else:
            qc.cx(e, k)
    qc.measure(range(n), range(n))
    return qc


def main():
    rng = np.random.default_rng(SEMENTE)
    L = [f"QACO (QMI 2022, doi 10.1007/s42484-022-00065-1) no verificador do padrao 1 | semente {SEMENTE} | {date.today()}"]
    tudo = True
    for restrito in (False, True):
        tvs, oks = [], set()
        for n in (3, 4, 5):
            for _ in range(20):
                qc = qaco(n, rng.uniform(0.05, np.pi - 0.05, n), rng.uniform(0.01, 0.5), restrito, rng)
                ok, det, plano = P1.classifica(qc)
                oks.add(ok)
                if ok:
                    sv = Statevector(qc.remove_final_measurements(inplace=False)).probabilities()
                    am = P1.amostrador(qc, plano)
                    # marginal das formigas (qubits 0..n-1)
                    nq = qc.num_qubits
                    msv = np.zeros(2 ** n)
                    mam = np.zeros(2 ** n)
                    for x in range(2 ** nq):
                        msv[x & ((1 << n) - 1)] += sv[x]
                        mam[x & ((1 << n) - 1)] += am[x]
                    tvs.append(0.5 * np.abs(msv - mam).sum())
        tudo &= oks == {True} and max(tvs) < 1e-12
        L.append(f"{'com restricao (Fredkin)' if restrito else 'sem restricao (CNOT)'}: amostragem classica = {sorted(oks)} | "
                 f"TV max {max(tvs):.1e} em {len(tvs)} circuitos")
    L.append(f"todos conforme o esperado: {tudo}")
    print("\n".join(L))
    (RAIZ / "resultados" / "qga" / f"certifica_qaco-{date.today()}.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
