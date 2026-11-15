"""Article 3 (11 Nov 2026): the measured output of the RQGA is uniform, whatever the number of
Grover iterations.

Uses the authors' public code, without rewriting the circuits:
  - Ardelean and Udrescu (PeerJ CS 2022), externos/rqga (commit a018d70), graph_coloring.ipynb:
    the functions get_ufit_instruction, get_oracle_instruction and get_grover_iteration_subcircuit
    are extracted from the notebook and run as they are. The only API adjustments: QuantumCircuit.mct
    (removed in Qiskit 1.0) becomes mcx, and the graph is generated with a seed instead of init_graph
    (nx.to_numpy_matrix was removed in networkx 3).
  - Viana and de Paula Neto (arXiv 2501.01058), externos/maxcut-qga (commit 23533c4): the class
    FragmentedQGA is imported as it is (the metis module, used only in preprocessing, is
    replaced by an empty module).
The circuit is built with the same lines as run_algorithm() / FragmentedQGA.run(), and Aer
(statevector) returns the exact distribution of the measured register (individuals) and, for contrast,
that of the fitness register, which Grover does change.

Usage (WSL): bash scripts/qga/rodar.sh
Output: resultados/qga/verifica_rqga-<data>.txt
"""
from __future__ import annotations

import json
import sys
import types
from datetime import date
from pathlib import Path

import networkx as nx
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

RAIZ = Path(__file__).resolve().parents[2]
EXT = RAIZ / "externos"
SEMENTE = 20261111
SIM = AerSimulator(method="statevector")
QuantumCircuit.mct = lambda self, controles, alvo: self.mcx(controles, alvo)


def tv(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def distribuicoes(qc, ind, fit):
    qc = qc.copy()
    qc.save_probabilities(list(ind), label="ind")
    qc.save_probabilities(list(fit), label="fit")
    r = SIM.run(transpile(qc, SIM, optimization_level=0)).result().data()
    return np.asarray(r["ind"]), np.asarray(r["fit"])


def amostra(qc, ind, shots, rng):
    qc = qc.copy()
    creg = ClassicalRegister(len(ind), "m")
    qc.add_register(creg)
    qc.measure(list(ind), creg)
    cont = SIM.run(transpile(qc, SIM, optimization_level=0), shots=shots,
                   seed_simulator=int(rng.integers(2**31))).result().get_counts()
    obs = np.zeros(2 ** len(ind))
    for k, v in cont.items():
        obs[int(k, 2)] = v
    esp = shots / len(obs)
    return float(((obs - esp) ** 2 / esp).sum()), len(obs) - 1


# ---------------- RQGA (Ardelean and Udrescu) ----------------
def carrega_rqga(n_nos, grafo):
    nb = json.loads((EXT / "rqga" / "graph_coloring.ipynb").read_text(encoding="utf-8"))
    celulas = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    ns = {"QuantumCircuit": QuantumCircuit, "QuantumRegister": QuantumRegister,
          "ClassicalRegister": ClassicalRegister}
    exec("import math, operator\nfrom numpy import flip,array,binary_repr,insert", ns)
    ns["NO_OF_NODES"] = n_nos
    for c in celulas:
        if c.lstrip().startswith("NO_OF_COLORS"):
            exec(c, ns)
        elif c.lstrip().startswith("def ") and "run_algorithm" not in c and "init_graph" not in c:
            exec(c, ns)
    ns["GRAPH"] = grafo
    return ns


def circuito_rqga(ns, iteracoes):
    # Same lines as run_algorithm(), without the measurement and without IBMQ.
    F = ns["NO_OF_QUBITS_FITNESS"]
    ind_qreg = QuantumRegister(ns["NO_OF_QUBITS_INDIVIDUAL"], "ireg")
    fit_qreg = QuantumRegister(F + 1, "freg")
    carry_qreg = QuantumRegister(9, "qcarry")
    oracle = QuantumRegister(1, "oracle")
    no_of_edges_qreg = QuantumRegister(F, "pos_max_qreg")
    qc = QuantumCircuit(ind_qreg, fit_qreg, carry_qreg, oracle, no_of_edges_qreg)
    qc.h(ind_qreg)
    qc.h(oracle)
    pos = ns["get_number_of_edges"](ns["GRAPH"])
    pos_value_bin = ns["to_binary"](pos, F, True)
    ufit = ns["get_ufit_instruction"]()
    orac = ns["get_oracle_instruction"](pos_value_bin)
    grov = ns["get_grover_iteration_subcircuit"]()
    qc.append(ufit, list(ind_qreg) + list(fit_qreg))
    for it in range(iteracoes):
        qc.append(orac, list(fit_qreg[:F]) + list(no_of_edges_qreg)
                  + [carry_qreg[0], carry_qreg[2 * it + 1], carry_qreg[2 * it + 2]] + [oracle[0]])
        qc.append(grov, list(fit_qreg) + [oracle[0]])
    return qc, ind_qreg, fit_qreg


# ---------------- MaxCut QGA (Viana and de Paula Neto) ----------------
def carrega_maxcut():
    sys.modules.setdefault("metis", types.ModuleType("metis"))
    sys.path.insert(0, str(EXT / "maxcut-qga"))
    import fqga  # noqa: E402
    from functions import to_binary  # noqa: E402
    return fqga.FragmentedQGA, to_binary


def circuito_maxcut(q, to_binary, iteracoes):
    # Same lines as FragmentedQGA.run(), without the measurement.
    q.weight_check()
    ind_qreg = QuantumRegister(q.individual_size, "ireg")
    fit_qreg = QuantumRegister(q.fitness_size + 1, "freg")
    carry_qreg = QuantumRegister(2 * q.grover_iterations + 1, "qcarry")
    oracle = QuantumRegister(1, "oracle")
    no_of_edges_qreg = QuantumRegister(q.fitness_size, "pos_max_qreg")
    qc = QuantumCircuit(ind_qreg, fit_qreg, carry_qreg, oracle, no_of_edges_qreg)
    qc.h(ind_qreg)
    qc.h(oracle)
    pos_value_bin = to_binary(q.total_edges, q.fitness_size, True)
    ufit = q.get_ufit_instruction()
    orac = q.get_oracle_instruction(pos_value_bin)
    grov = q.get_grover_iteration_subcircuit()
    qc.append(ufit, list(ind_qreg) + list(fit_qreg))
    for it in range(iteracoes):
        qc.append(orac, list(fit_qreg[:q.fitness_size]) + list(no_of_edges_qreg)
                  + [carry_qreg[0], carry_qreg[2 * it + 1], carry_qreg[2 * it + 2]] + [oracle[0]])
        qc.append(grov, list(fit_qreg) + [oracle[0]])
    return qc, ind_qreg, fit_qreg


def main():
    rng = np.random.default_rng(SEMENTE)
    linhas = [f"RQGA and MaxCut-QGA: distribution of the measured register | seed {SEMENTE} | {date.today()}",
              "case iter qubits | max|p_ind - 2^-N| | TV(p_ind, uniform) | TV(p_fit, p_fit[iter=1]) | "
              "chi2/df (8192 shots)"]
    casos = []
    for n in (3, 4):
        g = nx.erdos_renyi_graph(n, 0.6, seed=int(rng.integers(2**31)))
        adj = nx.to_numpy_array(g, dtype=int).tolist()
        ns = carrega_rqga(n, adj)
        casos.append((f"RQGA-coloring n={n} |E|={g.number_of_edges()}",
                      lambda it, ns=ns: circuito_rqga(ns, it)))
    Frag, to_binary = carrega_maxcut()
    for n in (4, 6, 8):
        g = nx.erdos_renyi_graph(n, 0.5, seed=int(rng.integers(2**31)))
        q = Frag(g, shots=1000)
        casos.append((f"MaxCut-QGA n={n} |E|={g.number_of_edges()}",
                      lambda it, q=q: circuito_maxcut(q, to_binary, it)))
    for rotulo, construtor in casos:
        ref_fit = None
        for it in range(1, 5):
            qc, ind, fit = construtor(it)
            p_ind, p_fit = distribuicoes(qc, ind, fit)
            unif = np.full_like(p_ind, 1 / len(p_ind))
            ref_fit = p_fit if ref_fit is None else ref_fit
            chi2, gl = amostra(qc, ind, 8192, rng)
            linhas.append(f"{rotulo} {it} {qc.num_qubits} | {np.abs(p_ind - unif).max():.2e} | "
                          f"{tv(p_ind, unif):.2e} | {tv(p_fit, ref_fit):.4f} | {chi2 / gl:.3f}")
            print(linhas[-1], flush=True)
    texto = "\n".join(linhas)
    destino = RAIZ / "resultados" / "qga"
    destino.mkdir(parents=True, exist_ok=True)
    (destino / f"verifica_rqga-{date.today()}.txt").write_text(texto + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
