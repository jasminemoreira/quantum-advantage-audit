"""Figures of the article, drawn from the result files in results/qga/ (15 Nov 2026).

Fig. 1  The 2006 worked example: probability of a marked fitness value, exact, per Grover iteration,
        for the two rounds, with the bound of Lemma 2 and the value claimed in 2006 (close to 1).
Fig. 2  Simulation cost of a Pattern 2 circuit: state vector versus the equivalent classical program.
Fig. 3  Corpus overview: verdicts of the development and application cases.

Usage: python scripts/figures/make_figures.py [--lang pt|en]   (needs matplotlib; no quantum package)
Output: results/figures/fig1_rqga2006.pdf, fig2_simulation_cost.pdf, fig3_corpus.pdf
"""
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
RES = RAIZ / "results" / "qga"
OUT = RAIZ / "results" / "figures"

# validated categorical slots (light surface), text inks, neutral for "no decision"
AZUL, LARANJA, AGUA = "#2a78d6", "#eb6834", "#1baf7a"
TEXTO, TEXTO2, GRADE, NEUTRO = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"

T = {
    "pt": dict(k="iterações de Grover", p="probabilidade de medir uma aptidão marcada",
               r1="rodada 1 (5 de 16 marcados)", r2="rodada 2 (2 de 16 marcados)",
               cota="cota do Lema 2", alegado="afirmado no artigo de 2006: próximo de 1",
               n="número de qubits", t="tempo (s)", sv="vetor de estado (Qiskit)",
               am="programa clássico, 1000 amostras", ver="verificador",
               mem="4 GiB de memória", dev="desenvolvimento", app="aplicação",
               p1="padrão 1", p2="padrão 2", inc="sem decisão", trab="trabalhos"),
    "en": dict(k="Grover iterations", p="probability of measuring a marked fitness",
               r1="round 1 (5 of 16 marked)", r2="round 2 (2 of 16 marked)",
               cota="bound of Lemma 2", alegado="claimed in the 2006 paper: close to 1",
               n="number of qubits", t="time (s)", sv="state vector (Qiskit)",
               am="classical program, 1000 samples", ver="verifier",
               mem="4 GiB of memory", dev="development", app="application",
               p1="pattern 1", p2="pattern 2", inc="no decision", trab="works"),
}


def ultimo(prefixo):
    arqs = sorted(RES.glob(f"{prefixo}-*.txt"))
    if not arqs:
        raise FileNotFoundError(prefixo)
    return arqs[-1].read_text(encoding="utf-8")


def estilo(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(TEXTO2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=TEXTO2, labelsize=8)
    ax.grid(True, color=GRADE, linewidth=0.6)
    ax.set_axisbelow(True)


def le_rqga(txt):
    ks, ps, cota = [], [], None
    for linha in txt.splitlines():
        m = re.match(r"\s*(\d+) \| (True|False) \| ([\d.]+) \| ([\d.]+) \|", linha)
        if m:
            ks.append(int(m.group(1)))
            ps.append(float(m.group(3)))
            cota = float(m.group(4))
    return ks, ps, cota


def fig1(L):
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    estilo(ax)
    for arq, cor, rot in (("rqga2006_exact_ex1", AZUL, L["r1"]), ("rqga2006_exact", LARANJA, L["r2"])):
        ks, ps, cota = le_rqga(ultimo(arq))
        ax.plot(ks, ps, color=cor, linewidth=2, marker="o", markersize=4, label=rot)
        ax.axhline(cota, color=cor, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(12.2, ps[-1], rot, color=TEXTO, fontsize=8, va="center")
        ax.text(12.2, cota + 0.012, L["cota"], color=TEXTO2, fontsize=7, va="bottom")
    ax.axhline(1.0, color=NEUTRO, linewidth=1.2, linestyle=(0, (1, 2)))
    ax.text(0, 0.97, L["alegado"], color=TEXTO2, fontsize=8, va="top")
    ax.set_xlim(-0.3, 12.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(L["k"], color=TEXTO, fontsize=9)
    ax.set_ylabel(L["p"], color=TEXTO, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.9))
    return fig


def fig2(L):
    txt = ultimo("simulation_cost")
    n, tv, ta, ts = [], [], [], []
    for linha in txt.splitlines():
        m = re.match(r"(\d+) \| ([\d.e+-]+) \| ([\d.e+-]+) \| ([\d.e+-]+|-)$", linha.strip())
        if m:
            n.append(int(m.group(1)))
            tv.append(float(m.group(2)))
            ta.append(float(m.group(3)))
            ts.append(None if m.group(4) == "-" else float(m.group(4)))
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    estilo(ax)
    nsv = [a for a, b in zip(n, ts) if b is not None]
    tsv = [b for b in ts if b is not None]
    ax.plot(nsv, tsv, color=LARANJA, linewidth=2, marker="o", markersize=4, label=L["sv"])
    ax.plot(n, ta, color=AZUL, linewidth=2, marker="o", markersize=4, label=L["am"])
    ax.plot(n, tv, color=AGUA, linewidth=2, marker="o", markersize=4, label=L["ver"])
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 28, 64, 128, 256, 400])
    ax.set_xticklabels(["4", "8", "16", "28", "64", "128", "256", "400"])
    ax.annotate(L["mem"], (nsv[-1], tsv[-1]), xytext=(6, -2), textcoords="offset points",
                color=TEXTO2, fontsize=7, va="center")
    ax.text(n[-1] * 1.04, ta[-1], L["am"].split(",")[0], color=TEXTO, fontsize=7, va="center")
    ax.text(n[-1] * 1.04, tv[-1], L["ver"], color=TEXTO, fontsize=7, va="center")
    ax.set_xlim(3.5, 900)
    ax.set_xlabel(L["n"], color=TEXTO, fontsize=9)
    ax.set_ylabel(L["t"], color=TEXTO, fontsize=9)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    return fig


def fig3(L):
    # counts by group; source: article Sections 3, 4, 5 and 7 (casos-novos RESUMO.md)
    grupos = [(L["dev"], {"p1": 4, "p2": 6, "inc": 1}), (L["app"], {"p1": 0, "p2": 11, "inc": 6})]
    cores = {"p1": AZUL, "p2": LARANJA, "inc": NEUTRO}
    fig, ax = plt.subplots(figsize=(5.6, 1.9))
    estilo(ax)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    for y, (nome, c) in enumerate(reversed(grupos)):
        x0 = 0
        for chave in ("p1", "p2", "inc"):
            v = c[chave]
            if v == 0:
                continue
            ax.barh(y, v - 0.08, left=x0 + 0.04, color=cores[chave], height=0.62)
            ax.text(x0 + v / 2, y, str(v), color="white" if chave != "inc" else TEXTO,
                    fontsize=8, ha="center", va="center", fontweight="bold")
            x0 += v
        ax.text(-0.4, y, f"{nome} ({x0})", ha="right", va="center", color=TEXTO, fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(0, 17.5)
    ax.set_xlabel(L["trab"], color=TEXTO, fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cores[k], label=L[k]) for k in ("p1", "p2", "inc")], frameon=False,
              fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="pt", choices=("pt", "en"))
    L = T[ap.parse_args().lang]
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Latin Modern Roman", "CMU Serif",
                         "DejaVu Serif"], "pdf.fonttype": 42})
    for nome, f in (("fig1_rqga2006", fig1), ("fig2_simulation_cost", fig2), ("fig3_corpus", fig3)):
        fig = f(L)
        fig.savefig(OUT / f"{nome}.pdf", bbox_inches="tight")
        fig.savefig(OUT / f"{nome}.png", bbox_inches="tight", dpi=160)
        plt.close(fig)
        print("ok", nome)


if __name__ == "__main__":
    main()
