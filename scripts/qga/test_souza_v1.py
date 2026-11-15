"""Tests of the operators in souza_v1.py (13 Nov 2026). Usage: python scripts/qga/test_souza_v1.py"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import souza_v1 as S  # noqa: E402
import center_bias as VC  # noqa: E402


def main():
    rng = np.random.default_rng(7)
    falhas = []
    # 1) layers remain partitions of the qubits after crossover and mutation; G_cl without H
    casos = Counter()
    for conj in (S.G_CL, S.G_QU):
        pop = [S.individuo(rng, conj) for _ in range(40)]
        for _ in range(2000):
            a, b = pop[rng.integers(40)], pop[rng.integers(40)]
            f1, f2 = S.cruza(rng, a, b)
            f1, f2 = S.muta(rng, f1, conj), S.muta(rng, f2, conj)
            for f in (f1, f2):
                for cam in f["cams"]:
                    if not S.confere_camada(cam):
                        falhas.append(("partition", cam))
                    if conj is S.G_CL and any(g[0] not in {n for n, _ in S.G_CL} for g in cam):
                        falhas.append(("gate outside G_cl", cam))
                    for g in cam:
                        casos[len(g[1])] += 1
            pop[rng.integers(40)] = f1
            pop[rng.integers(40)] = f2
    print("1) partitions ok" if not falhas else f"1) FAILURES {falhas[:3]}", "| gates per arity:", dict(casos))
    # 2) crossover with k = n swaps the whole layer; with an empty region nothing changes
    a, b = S.individuo(rng, S.G_QU), S.individuo(rng, S.G_QU)
    x, y = S.troca_camada(a["cams"][3], b["cams"][3], set(range(S.NQ)))
    ok2 = sorted(x) == sorted(b["cams"][3]) and sorted(y) == sorted(a["cams"][3])
    x, y = S.troca_camada(a["cams"][3], b["cams"][3], set())
    ok2 &= sorted(x) == sorted(a["cams"][3])
    print("2) crossover extremes:", ok2)
    falhas += [] if ok2 else ["crossover extremes"]
    # 3) low region: k=1 takes the last qubit of each register (least significant bit)
    ok3 = S.regiao_baixa(1) == {7, 15} and S.regiao_baixa(8) == set(range(16))
    print("3) low region:", ok3)
    falhas += [] if ok3 else ["region"]
    # 4) per-gene mutation rate ~ 0.3 (fraction of altered genes, only case 1 counts exactly; approximate)
    conj = S.G_QU
    mud = tot = 0
    for _ in range(300):
        ind = S.individuo(rng, conj)
        m = S.muta(rng, ind, conj)
        for c0, c1 in zip(ind["cams"], m["cams"]):
            tot += len(c0)
            mud += len(set(c0) - set(c1))
    print(f"4) fraction of altered genes: {mud / tot:.3f} (above 0.3 is expected: cases 2, 3 and 6 remove neighboring gates, and the draw can repeat the gate)")
    # 5) evaluator: G_cl quantum == classical bits; bell quantum and analog have the same support on G_cl
    err = 0
    for _ in range(30):
        ind = S.individuo(rng, S.G_CL)
        p = S.p_quantico(ind)
        if int(np.argmax(p)) != int(S.amostra_bits(ind, rng, False)[0]) or abs(p.max() - 1) > 1e-12:
            err += 1
        pb = S.p_quantico(ind, "bell")
        sup_q = set(np.flatnonzero(pb > 1e-12))
        sup_c = set(S.amostra_bits(ind, rng, False, "bell").tolist())
        if sup_q != sup_c or not np.allclose(pb[list(sup_q)], 1 / len(sup_q)):
            err += 1
    print("5) G_cl evaluators (without and with bell): errors =", err)
    falhas += [] if err == 0 else ["evaluators"]
    # 6) state against Qiskit on v1 individuals (with I and mutated)
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    dif = 0.0
    for _ in range(15):
        ind = S.muta(rng, S.individuo(rng, S.G_QU), S.G_QU)
        qc = QuantumCircuit(S.NQ)
        for q in range(S.NQ):
            if (ind["base"] >> q) & 1:
                qc.x(q)
        for cam in ind["cams"]:
            for nome, q in cam:
                getattr(qc, "id" if nome == "i" else nome)(*q)
        dif = max(dif, float(np.abs(Statevector(qc).data - VC.estado(S.sem_i(ind))).max()))
    print(f"6) max |psi - qiskit| = {dif:.2e}")
    falhas += [] if dif < 1e-10 else ["qiskit"]
    # 7) "max": uniform output; the analog with a uniform start too
    ind = S.individuo(rng, S.G_QU)
    idx = S.amostra_bits(ind, rng, True, "max")
    print(f"7) analog 'max': decoded mean {VC.decodifica(idx, (-5.12, 5.12)).mean(axis=0).round(3)} (center = 0)")
    # 8) sensitivity-analysis variants: layers remain partitions; G_cl without H; frozen H layer intact
    import sensitivity_souza as SS
    for var, opc in SS.VARIANTES.items():
        S.OPC.clear(); S.OPC.update(S.PADRAO, **opc)
        rv = np.random.default_rng(11)
        ruins = 0
        for conj in (S.G_CL, S.G_QU):
            pop = [S.individuo(rv, conj, prof=S.OPC["prof"]) for _ in range(20)]
            for _ in range(300):
                f1, f2 = S.cruza(rv, pop[rv.integers(20)], pop[rv.integers(20)])
                for f in (S.muta(rv, f1, conj), S.muta(rv, f2, conj)):
                    ruins += sum(not S.confere_camada(c) for c in f["cams"]) + (len(f["cams"]) != S.OPC["prof"])
                    ruins += not (0 <= f["base"] < 2 ** S.NQ)
                    if conj is S.G_CL:
                        ruins += any(g[0] == "h" for c in f["cams"] for g in c)
                    pop[rv.integers(20)] = f
            if conj is S.G_QU and not S.OPC["h_muta"]:
                ruins += any(g[0] not in ("h", "i") for p_ in pop for g in p_["cams"][0])
        _, mel, fm = S.ga("quantico", (-5.12, 5.12), 3, pop_n=10, ger=2)
        print(f"8) variant {var}: violations {ruins} | short ga ok, f = {fm:.3f}")
        falhas += [] if ruins == 0 else [f"variant {var}"]
    S.OPC.clear(); S.OPC.update(S.PADRAO)
    print("ALL OK" if not falhas else f"FAILURES: {falhas[:5]}")
    return 0 if not falhas else 1


if __name__ == "__main__":
    sys.exit(main())
