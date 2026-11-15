> English translation of the Portuguese original `ablacao_protocolo-2026-11-14.md` (same folder), which remains the authoritative record. Decimal commas of the original are written here as decimal points.

# Ablation protocol for the Souza, Dardenne and Portugal case (fixed on 14 Nov 2026, before running)

Fixed before any execution of the script `scripts/qga/ablacao_souza.py`. Changes after this date
are entered as dated amendments at the end of this file. Responds to the lens 0 review
(`ArtigoQCalc/parecer-lente0-2026-11-14.md`, objection 1).

## Objective
Demonstrate the ablation control of the procedure: decide whether the advantage reported by the
preprint (arXiv:2511.05254 v1) comes from the circuit or from the design of the evaluation (fitness on
the decoded mean and optimum at the center of the domain).

## Design
- Reimplementation: `scripts/qga/souza_v1.py`, default choices (`PADRAO`), |P| = 50, G = 30, 1024
  measurements per evaluation, 2 registers of 8 qubits.
- Arms: quantum (G_qu, with H); classical analog ("gêmeo", i.e. twin: H replaced by XOR with a coin,
  Y by X, phases discarded); classical (G_cl, without H).
- Functions (usual domain of width W): Rastrigin (W = 10.24), Sphere (10.24), Ackley (65.536),
  Griewank (1200), Rosenbrock (4.096). Optimum o = 0, except Rosenbrock (o = 1).
- Domains: [o − αW, o + (1 − α)W] with α ∈ {0.50 (centered on the optimum), 0.355, 0.26, 0.16}. The
  values of α were chosen so that, on Rastrigin, the center of the domain lies near half-integers
  (shifts of 1.48; 2.46; 3.48), far from the local minima. No shift is an integer.
- Repetitions: 50 per arm, function and domain (3000 runs). Deterministic seeds (base
  20261123).
- Output per run: curve of the best evaluated f at generations 0 to 30.

## Measures and tests (fixed)
- Measure: log(f + 1e-9) of the best evaluated f.
- Equivalence margin: ratio of geometric means in [0.80; 1.25] (bioequivalence convention), that is,
  difference of log means within ±ln 1.25 = ±0.2231. Equivalence by TOST (two one-sided Welch tests,
  α = 0.05 each; equivalent to the 90% CI contained in the margin).
- Superiority: one-sided Welch test "quantum better (smaller log f)", α = 0.05, with Holm correction
  within each family.

## Predictions (H_A1 to H_A3)
- **H_A1, signature of the center bias (g0).** Quantum − classical (log) negative on the centered
  domain and non-negative on the three shifted ones, for each function. Family: 5 functions × 4
  domains. The direction and the 95% CI are reported.
- **H_A2, no final superiority (g30, shifted).** In none of the 15 pairs (5 functions × 3 shifted
  domains) is the quantum superior to the classical (Holm, family of 15). Likewise quantum against
  analog.
- **H_A3, final equivalence (g30, shifted).** Quantum and classical equivalent under the margin above;
  quantum and analog likewise. The fraction of pairs with demonstrated equivalence is reported; the
  others are "inconclusive" (neither difference nor equivalence), not "equal".
- Also reported, without prediction: the same comparisons on the centered domain and at all
  generations (curves), and where the analog lags behind the quantum.

## Reading criterion
- If H_A1 and H_A2 hold: the reported advantage is explained by the design of the evaluation.
- If H_A2 fails in some shifted pair: report it as an unexplained quantum advantage in that pair.
- An inconclusive H_A3 does not contradict H_A2; it says that the number of repetitions is not enough
  to claim a tie.

## Amendments
- 14/11/2026: functional test of the script with 2 repetitions per cell, only to check that it runs
  to the end; output deleted without reading the results. Nothing in the protocol was changed.
- 15/11/2026: full run finished on 14/11 (6378 s). Reading against the predictions in
  `ArtigoQCalc/plano-artigo-2026-11-14.md`, Section 4.7.2. The monotonicity with the shift was noticed
  after seeing the data and is exploratory.
