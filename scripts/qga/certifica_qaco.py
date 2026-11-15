"""Pattern 2 (product state) in the QACO published in QMI itself (14 Nov 2026).

Case: "Implementable hybrid quantum ant colony optimization algorithm", Quantum Machine Intelligence,
2022, doi 10.1007/s42484-022-00065-1 (Sections 3.1.1 and 3.1.2 read on the Springer site on 14 Nov).
Circuit reconstructed from the text:
  - ant qubits: RY(theta_i) from |0> (pheromone), 0 < theta_i < pi;
  - exploration: a qubit with RY(theta_e), theta_e = 2 arcsin(sqrt(beta_e)), controls one gate and is
    reset after each controlled gate; here the reset + new RY is represented by a new
    exploration qubit for each gate (the authors also "duplicate" this qubit);
  - unconstrained: CNOT(exploration -> ant) on each ant;
  - constrained (fixed number of 1s): Fredkin(exploration; t1, t2) on pairs of ants;
  - measures the ants.
Prediction: classified as classical sampling (Theorem 1); sampler = statevector.

Usage (WSL): externos/mqtbench/.venv/bin/python scripts/qga/certifica_qaco.py
Output: resultados/qga/certifica_qaco-<data>.txt
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
    L = [f"QACO (QMI 2022, doi 10.1007/s42484-022-00065-1) on the checker of Pattern 2 (product state) | seed {SEMENTE} | {date.today()}"]
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
                    # marginal of the ants (qubits 0..n-1)
                    nq = qc.num_qubits
                    msv = np.zeros(2 ** n)
                    mam = np.zeros(2 ** n)
                    for x in range(2 ** nq):
                        msv[x & ((1 << n) - 1)] += sv[x]
                        mam[x & ((1 << n) - 1)] += am[x]
                    tvs.append(0.5 * np.abs(msv - mam).sum())
        tudo &= oks == {True} and max(tvs) < 1e-12
        L.append(f"{'constrained (Fredkin)' if restrito else 'unconstrained (CNOT)'}: classical sampling = {sorted(oks)} | "
                 f"TV max {max(tvs):.1e} over {len(tvs)} circuits")
    L.append(f"all as expected: {tudo}")
    print("\n".join(L))
    (RAIZ / "resultados" / "qga" / f"certifica_qaco-{date.today()}.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
