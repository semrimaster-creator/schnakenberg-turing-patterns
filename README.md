# Turing Pattern Formation in the Schnakenberg Reaction–Diffusion System

**A linear stability analysis validated against direct numerical simulation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![tests](https://github.com/semrimaster-creator/schnakenberg-turing-patterns/actions/workflows/tests.yml/badge.svg)](https://github.com/semrimaster-creator/schnakenberg-turing-patterns/actions/workflows/tests.yml)

Mohamed Semri — [semri.master@gmail.com](mailto:semri.master@gmail.com)

---

## Overview

This repository studies **diffusion-driven (Turing) instability** in the
Schnakenberg reaction–diffusion system,

```
∂u/∂t = D_u ∇²u + a − u + u²v
∂v/∂t = D_v ∇²v + b − u²v
```

on a doubly periodic square domain. The project has two independent, mutually
validating components:

1. **Linear stability analysis (LSA)** — an analytic dispersion relation
   λ(k) derived from linearizing the reaction–diffusion operator about the
   homogeneous steady state (u\*, v\*), giving closed-form Turing instability
   conditions and the critical wavenumber k_c.
2. **Direct numerical simulation (DNS)** — a pseudo-spectral solver
   (RK4 for the reaction term, exact integrating-factor propagation for
   diffusion in Fourier space) that evolves the full nonlinear PDE system
   from a noisy initial condition and lets patterns emerge on their own.

The DNS is then used to **measure**, independently of the theory, the growth
rate of each unstable Fourier mode and the dominant emergent wavelength — and
these measurements are compared quantitatively against the LSA prediction.
Agreement between the two (typically a few percent) is the validation result
this repository reports.

## Why this matters for transport/kinetic theory

The mathematical object at the center of this project — a system of coupled
reaction–diffusion PDEs, linearized about a steady state, with stability
governed by the sign of eigenvalues of a wavenumber-dependent dispersion
matrix — is the same structural object that appears in:

- **Neutron diffusion / multigroup transport theory**, where criticality is
  determined by the sign of the dominant eigenvalue of a diffusion-plus-
  reaction operator;
- **Reactor kinetics stability analysis**, where perturbations about a
  steady flux are linearized and classified by growth/decay of Fourier
  (or spatial-mode) components;
- **Pattern-forming instabilities in driven nuclear and plasma media**
  (e.g. void lattice formation under irradiation, which is itself modeled
  with reaction–diffusion equations of Turing type).

This project is a self-contained, fully validated testbed for exactly that
methodology — linearization, dispersion-relation derivation, and numerical
confirmation via direct simulation — applied to a simpler and better-
understood chemical system before being carried over to nuclear transport
and kinetics problems.

This work builds directly on two prior projects: an M.Sc. thesis modeling
reaction–diffusion dynamics of host–pathogen systems with the Finite Element
Method (Nanjing Agricultural University, 2019), and a Monte Carlo simulation
of radioactive decay validated against the analytic exponential law
(see [nuclear-decay-monte-carlo](https://github.com/semrimaster-creator/nuclear-decay-monte-carlo)).
Together the three form one coherent line of work: stochastic microscopic
rules and deterministic PDEs, both validated quantitatively against theory,
applied across biological, chemical, and nuclear-relevant systems.

## Repository structure

```
schnakenberg-turing-patterns/
├── src/schnakenberg/
│   ├── parameters.py      # SchnakenbergParams dataclass, steady state
│   ├── lsa.py              # Jacobian, dispersion relation, Turing conditions
│   ├── solver.py           # Pseudo-spectral DNS (RK4 + integrating factor)
│   └── diagnostics.py      # Growth-rate extraction, dominant wavelength
├── scripts/
│   ├── run_lsa.py           # Step 1: theoretical analysis + dispersion plot
│   ├── run_dns.py           # Step 2: full pattern-formation simulation
│   ├── run_validation.py    # Step 3: DNS vs LSA quantitative comparison
│   └── make_report.py       # Step 4: assembles results/scientific_report.pdf
├── tests/
│   └── test_schnakenberg.py # Unit tests (steady state, Turing conditions,
│                             #  dispersion relation limits, solver conservation)
├── figures/                 # Generated plots (git-ignored)
├── results/                 # Generated CSV/TXT/PDF outputs (git-ignored)
├── requirements.txt
├── LICENSE
└── CITATION.cff
```

## Method summary

**Steady state.** For a, b > 0, the unique homogeneous steady state is
u\* = a + b, v\* = b / (a + b)².

**Linearization.** The Jacobian of the reaction kinetics at (u\*, v\*) is

```
J = [ 2u*v* − 1      u*²   ]
    [ −2u*v*        −u*²   ]
```

**Dispersion relation.** Seeking perturbations ∝ exp(λt + ik·x) gives a
quadratic in λ for each wavenumber k, λ² − A(k)λ + B(k) = 0, with

```
A(k) = T − (D_u + D_v) k²
B(k) = Δ − M k² + D_u D_v k⁴
```

where T = tr(J), Δ = det(J), and M = D_v J₁₁ + D_u J₂₂. Diffusion-driven
(Turing) instability requires the reaction system to be stable on its own
(T < 0, Δ > 0) while diffusion destabilizes it at finite k (M > 0 and
M² > 4 D_u D_v Δ) — the four conditions checked by `lsa.turing_conditions`.

**Critical wavenumber.** k_c² = M / (2 D_u D_v) maximizes the growth rate
λ₊(k), and this is the wavenumber DNS is expected to select.

**DNS validation.** `diagnostics.mode_growth_rates` performs a log-linear
regression of |û(k, t)| over the early linear-growth window, for the
dominant Fourier modes, and compares the fitted rate against λ₊(k) from the
closed-form dispersion relation.

## Default parameter set

a = 0.1, b = 0.9, D_u = 1.0, D_v = 40.0, on a domain of size L = 100 —
a standard Turing-unstable regime for this system (see Murray, 2003, and
the original Schnakenberg 1979 reference below). With these values, k_c ≈
0.622 (wavelength ≈ 10.1) and λ(k_c) ≈ 0.305; all four Turing conditions are
satisfied.

## Reproducing the results

```bash
pip install -r requirements.txt

python scripts/run_lsa.py          # theory: dispersion relation + conditions
python scripts/run_dns.py          # simulation: pattern formation
python scripts/run_validation.py   # comparison: DNS growth rates vs LSA
python scripts/make_report.py      # assembles results/scientific_report.pdf

pytest tests/ -v                   # unit tests
```

Each script writes its numerical outputs to `results/` (CSV + TXT) and its
plots to `figures/` (PNG), so every figure in the final report is
reproducible from a checked-in, git-ignored-only-for-size artifact.

## Numerical method notes / limitations

- The diffusion step is solved **exactly** (no time-discretization error)
  via the spectral integrating factor `exp(-D k² dt)`; only the reaction
  step carries O(dt⁴) local truncation error from RK4.
- Periodic boundary conditions are required for the FFT-based spectral
  method; this is standard for studying pattern selection far from
  boundaries but is not appropriate for bounded/Neumann domains without
  modification (e.g. a cosine transform).
- Growth-rate fits use only the early-time window where |û(k,t)| is still
  in the linear (pre-saturation) regime; the amplitude bounds
  `amp_lo`/`amp_hi` in `diagnostics.mode_growth_rates` control this window
  and can be tightened for a stricter validation.
- A grid-resolution / domain-size convergence study is a natural next
  extension (see Future Work).

## Future work

- Resolution and domain-size convergence study (N = 128, 256, 512;
  L scaled to keep k_c · L fixed) to demonstrate the DNS result is
  grid-independent, not a discretization artifact.
- Weakly nonlinear (amplitude-equation) analysis near the Turing
  bifurcation to predict pattern *type* (spots vs. stripes), not just
  onset wavelength.
- Extension to a three-species or spatially heterogeneous diffusion
  coefficient, as a bridge toward multigroup transport/kinetics models.

## References

- Turing, A. M. (1952). The chemical basis of morphogenesis.
  *Philosophical Transactions of the Royal Society B*, 237(641), 37–72.
- Schnakenberg, J. (1979). Simple chemical reaction systems with limit
  cycle behaviour. *Journal of Theoretical Biology*, 81(3), 389–400.
- Murray, J. D. (2003). *Mathematical Biology II: Spatial Models and
  Biomedical Applications* (3rd ed.). Springer. — Chapter 2–3 for the
  general Turing-instability framework used here.
- Trefethen, L. N. (2000). *Spectral Methods in MATLAB*. SIAM. — for the
  pseudo-spectral / integrating-factor time-stepping approach.

## License

MIT — see [LICENSE](LICENSE).
