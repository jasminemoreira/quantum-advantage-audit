# Where does the quantum advantage go? — reproduction package

Code and result files for the article *Where does the quantum advantage go? An audit of evolutionary
algorithms executed on quantum circuits* (manuscript in preparation for *Quantum Machine
Intelligence*).

Every number in the article comes from a dated text file in `resultados/qga/`, produced by one of the
scripts in `scripts/`. Each result file states the random seed in its header. File names, comments
and output headers are in Portuguese.

## Contents

| Script (`scripts/`) | Produces (`resultados/qga/`) | Article |
|---|---|---|
| `qcalc/esparso.py` | exact sparse simulator over Z[ζ₁₆] (library) | Sec. 3 |
| `qcalc/converte.py` | Qiskit circuit → simulator gates (library) | Sec. 3 |
| `qga/rqga2006_exato.py [--exemplo 1\|2]` | `rqga2006_exato_ex1-*.txt`, `rqga2006_exato-*.txt` | Sec. 4, Table 1 |
| `qga/rqga2006_exemplo.py` | `rqga2006_exemplo-*.txt` (same, floating point) | Sec. 4 |
| `qga/verifica_rqga.py` | `verifica_rqga-*.txt` (authors' code, Qiskit Aer) | Sec. 4 |
| `qga/certifica_rqga.py` | `certifica_rqga-*.txt` (exact uniformity, 20 cases) | Sec. 4 |
| `qga/confere_aer.py` | cross-check of exact vs. Aer marginals | Sec. 4 |
| `qga/certifica_padrao1.py` | `certifica_padrao1-*.txt` (verifier, cases, negative controls) | Sec. 5 |
| `qga/certifica_qaco.py` | `certifica_qaco-*.txt` | Sec. 5 |
| `qga/souza_v1.py`, `qga/vies_centro.py` | reimplementation of the ablation example (library) | Sec. 6, Suppl. S2 |
| `qga/ablacao_souza.py` | `ablacao_souza-*.txt`, `.npz` (protocol: `ablacao_protocolo-*.md`) | Sec. 6 |
| `qga/certifica_pares.py` | `certifica_pares-*.txt` (exact reduction of entanglement between individuals) | Sec. 6 |
| `qga/exp_souza_v1.py` | `exp_souza_v1-*.txt` | Sec. 6 |
| `qga/gemeo_exato.py`, `gemeo_exato_anel.py`, `gemeo_aptidao.py`, `gemeo_centro.py` | `gemeo_*-*.txt` | Suppl. S3 |
| `qga/sensibilidade_souza.py` | `sensibilidade_souza-*.txt` | Suppl. S4 |
| `qga/certifica_souza.py`, `certifica_souza_v1.py` | `certifica_souza*-*.txt` (Schmidt rank, separability) | Sec. 6 |
| `qga/entropia_funcoes_souza.py` | `entropia_funcoes_souza-*.txt` | Sec. 6 |
| `qga/confere_pinsker.py` | `confere_pinsker-*.txt` | Suppl. |
| `qga/teste_souza_v1.py` | unit tests of the reimplementation | — |

Scripts write a new file named with the current date; the files committed here are the ones cited in
the article.

## Environment

Python 3.12 with the packages in `requirements.txt` (the results were produced with Qiskit 2.6.2,
Qiskit Aer 0.18.2, NumPy 2.5.4, SciPy 1.18.1, NetworkX 3.7). Run the scripts from the repository
root, for example:

```
python scripts/qga/rqga2006_exato.py --exemplo 2
```

The longest runs are `ablacao_souza.py` (about 2 h on 16 cores), `sensibilidade_souza.py` (about
4.5 h) and `entropia_funcoes_souza.py` (about 3 h).

## Third-party code

`verifica_rqga.py` and `certifica_rqga.py` execute the public code of two published works, which is
not included here. Clone them into `externos/` at the commits used:

```
git clone https://github.com/sebastianardelean/graphcoloringusingrqga externos/rqga
git -C externos/rqga checkout a018d70
git clone https://github.com/pauloaviana/maxcut-qga externos/maxcut-qga
git -C externos/maxcut-qga checkout 23533c4
```

The only changes needed to run them are API updates (`mct` → `mcx`), a fixed graph seed and an empty
METIS partition; they are applied at run time by `verifica_rqga.py`, without editing the cloned code.

## Note on `converte.py`

In the original working repository, the circuit converter lived in a larger script that also
depended on MQT Bench. `converte.py` contains the same functions, unchanged, so that MQT Bench is not
needed. The outputs of `rqga2006_exemplo.py`, `rqga2006_exato.py` and `certifica_padrao1.py` were
re-run from this package and match the committed files (except for run dates and timings).

## License

Code: MIT License (`LICENSE`). Result files in `resultados/`: CC BY 4.0.
