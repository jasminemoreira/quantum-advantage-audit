# Classical sampling and Grover without coherence — reproduction package

Code and result files for the article *Classical sampling and Grover without coherence: a structural audit of evolutionary algorithms run on quantum circuits* (manuscript in preparation for *Quantum Machine
Intelligence*).

Every number in the article comes from a dated text file in `results/qga/`, produced by one of the
scripts in `scripts/`. Each result file states the random seed in its header.

## Contents

| Script (`scripts/`) | Produces (`results/qga/`) | Article |
|---|---|---|
| `qcalc/sparse_engine.py` | exact sparse simulator over Z[ζ₁₆] (library) | Sec. 3 |
| `qcalc/qiskit_convert.py` | Qiskit circuit → simulator gates (library) | Sec. 3 |
| `qga/rqga2006_exact.py [--exemplo 1\|2]` | `rqga2006_exact_ex1-*.txt`, `rqga2006_exact-*.txt` | Sec. 4, Table 1 |
| `qga/rqga2006_example_float.py` | `rqga2006_example_float-*.txt` (same, floating point) | Sec. 4 |
| `qga/run_rqga_authors_code.py` | `run_rqga_authors_code-*.txt` (authors' code, Qiskit Aer) | Sec. 4 |
| `qga/certify_rqga.py` | `certify_rqga-*.txt` (exact uniformity, 20 cases) | Sec. 4 |
| `qga/check_aer.py` | cross-check of exact vs. Aer marginals (prints to the terminal) | Sec. 4 |
| `qga/certify_product_state.py` | `certify_product_state-*.txt` (verifier, cases, negative controls) | Sec. 5 |
| `qga/certify_qaco.py` | `certify_qaco-*.txt` | Sec. 5 |
| `qga/simulation_cost.py` | `simulation_cost-*.txt` (state vector vs. per-shot classical sampler, 4 to 400 qubits) | Sec. 5, Fig. 2 |
| `figures/make_figures.py [--lang pt\|en]` | `results/figures/fig1_rqga2006.pdf`, `fig2_simulation_cost.pdf`, `fig3_corpus.pdf` (drawn from the result files; needs matplotlib only) | Figs. 1 to 3 |
| `qga/cases_acampora.py` | `cases_acampora-*.txt` (HQGA, QMO, Rosales et al.) | Sec. 5, application cases |
| `qga/cases_ant_colony_grover.py` | `cases_ant_colony_grover-*.txt` (Wu et al., Ghosh et al., Martinez et al., Gomez Cadavid et al.) | Secs. 4, 5, 7 |
| `qga/cases_other_gas.py` | `cases_other_gas-*.txt` (Wang and Wang, Sarracino et al., Abdolahi et al., Soloviev et al.) | Secs. 5, 7 |

Scripts write a new file named with the current date; the files committed here are the ones cited in
the article.

## Environment

Python 3.12 with the packages in `requirements.txt` (the results were produced with Qiskit 2.6.2,
Qiskit Aer 0.18.2, NumPy 2.5.4, SciPy 1.18.1, NetworkX 3.7). Run the scripts from the repository
root, for example:

```
python scripts/qga/rqga2006_exact.py --exemplo 2
```

Every script finishes in a few minutes on a laptop; the longest is `run_rqga_authors_code.py`
(about 4 min).

## Third-party code

`run_rqga_authors_code.py` and `certify_rqga.py` execute the public code of two published works, which
is not included here. Clone them into `external/` at the commits used:

```
git clone https://github.com/sebastianardelean/graphcoloringusingrqga external/rqga
git -C external/rqga checkout a018d70
git clone https://github.com/pauloaviana/maxcut-qga external/maxcut-qga
git -C external/maxcut-qga checkout 23533c4
```

The only changes needed to run them are API updates (`mct` → `mcx`), a fixed graph seed and an empty
METIS partition; they are applied at run time by `run_rqga_authors_code.py`, without editing the
cloned code.

## Note on `qiskit_convert.py`

In the original working repository, the circuit converter lived in a larger script that also
depended on MQT Bench. `qiskit_convert.py` contains the same functions, unchanged, so that MQT Bench
is not needed.

## Language of the code and original results

The code was written in Portuguese and later translated: comments, docstrings, output text and file
names are in English. Identifiers (variable and function names) and a few strings used as data were
kept in Portuguese, because renaming them would change the code that produced the results.

The results produced before the translation, with Portuguese file names and headers, are kept in
`results/qga-pt-original/`. Every script was re-run after the translation; the numbers in each new
file are identical to those in the corresponding original file, except for run dates and timings.

| Original name | Current name |
|---|---|
| `qcalc/esparso.py` | `qcalc/sparse_engine.py` |
| `qcalc/converte.py` | `qcalc/qiskit_convert.py` |
| `rqga2006_exato.py` | `rqga2006_exact.py` |
| `rqga2006_exemplo.py` | `rqga2006_example_float.py` |
| `verifica_rqga.py` | `run_rqga_authors_code.py` |
| `certifica_rqga.py` | `certify_rqga.py` |
| `confere_aer.py` | `check_aer.py` |
| `certifica_padrao1.py` | `certify_product_state.py` |
| `certifica_qaco.py` | `certify_qaco.py` |
| `acampora_casos.py` | `cases_acampora.py` |
| `formigas_grover.py` | `cases_ant_colony_grover.py` |
| `geneticos_outros.py` | `cases_other_gas.py` |

## License

Code: MIT License (`LICENSE`). Result files in `results/`: CC BY 4.0.
