"""Conversion of Qiskit circuits into the gates of the exact engine (extracted from mede_kmin_mqt.py,
thesis repository, 13 Nov 2026; the function bodies are identical to the ones used for the results)."""
import math

FASE = {"z": 8, "s": 4, "sdg": 12, "t": 2, "tdg": 14}


def angulo_pi8(theta):
    m = theta / (math.pi / 8)
    r = round(m)
    return r % 16 if abs(m - r) < 1e-9 else None


MCX_NOMES = {"ccx", "mcx", "mcx_gray", "mcx_recursive", "mcx_vchain", "c3x", "c4x", "rcccx"}


def converte(qc, rota_exata=True):
    """QuantumCircuit -> list of engine gates; None if there is a phase that is not a multiple of pi/8.

    Recursively descends into the definitions of composite gates. On the exact route, MCX (any
    variant) and MCPhase with an angle that is a multiple of pi/8 are kept whole.
    """
    out = []

    def anda(circ, mapa):
        for ins in circ.data:
            op = ins.operation
            nm = op.name
            qs = [mapa[circ.find_bit(q).index] for q in ins.qubits]
            if nm in ("barrier", "measure", "id", "delay", "global_phase"):
                continue
            if nm == "h":
                out.append(("h", qs[0]))
            elif nm == "x":
                out.append(("mcx", [], qs[0]))
            elif nm == "y":  # Y = i X Z
                out.extend([("fase", 8, [qs[0]]), ("mcx", [], qs[0]), ("fase", 4, [])])
            elif nm in FASE:
                out.append(("fase", FASE[nm], [qs[0]]))
            elif nm == "cx":
                out.append(("mcx", [qs[0]], qs[1]))
            elif nm == "cz":
                out.append(("fase", 8, qs))
            elif nm == "swap":
                a, b = qs
                out.extend([("mcx", [a], b), ("mcx", [b], a), ("mcx", [a], b)])
            elif rota_exata and nm in MCX_NOMES and nm != "rcccx":
                out.append(("mcx", qs[:-1], qs[-1]))
            elif rota_exata and nm in ("p", "cp", "mcphase", "mcp", "u1", "cu1", "rz") and nm != "rz":
                m = angulo_pi8(float(op.params[0]))
                if m is None:
                    raise ValueError("phase not a multiple of pi/8")
                out.append(("fase", m, qs if nm != "p" and nm != "u1" else [qs[0]]))
            elif op.definition is not None:
                anda(op.definition, qs)
            else:
                raise ValueError(f"gate without definition: {nm}")

    try:
        anda(qc, list(range(qc.num_qubits)))
    except ValueError:
        return None
    return out
