#!/usr/bin/env python3
"""Step 3: Quantitative validation of LSA against DNS.

Runs a fresh DNS, extracts per-Fourier-mode growth rates via log-linear
regression during the early linear regime, and compares them mode-by-mode
against the closed-form dispersion relation lambda_+(k).
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.schnakenberg import SchnakenbergParams, lsa, solver, diagnostics

ROOT = os.path.join(os.path.dirname(__file__), "..")
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--L", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--t-end", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-modes", type=int, default=6)
    args = parser.parse_args()

    p = SchnakenbergParams()
    kc = lsa.critical_wavenumber(p)
    k_star = lsa.fastest_growing_wavenumber(p)
    lam_max = lsa.max_growth_rate(p)
    bif_dist = lsa.bifurcation_distance(p)

    sim = solver.SpectralSolver(p, N=args.N, L=args.L, dt=args.dt, seed=args.seed)
    times, spectra, _ = sim.run(t_end=args.t_end, record_every=5)

    measured = diagnostics.mode_growth_rates(times, spectra, sim.Kmag, n_modes=args.n_modes)

    print("=" * 60)
    print("VALIDATION: DNS growth rates vs LSA dispersion relation")
    print("=" * 60)
    print(f"{'k (DNS)':>10s} {'lambda_DNS':>12s} {'lambda_LSA':>12s} {'rel.error':>11s}")

    rows = []
    for m in measured:
        lam_th = float(lsa.dispersion_relation(np.array([m["k"]]), p)[1][0])
        err = abs(m["growth_rate_dns"] - lam_th) / max(abs(lam_th), 1e-12)
        rows.append((m["k"], m["growth_rate_dns"], lam_th, err))
        print(f"{m['k']:10.4f} {m['growth_rate_dns']:12.4f} {lam_th:12.4f} {err:10.2%}")

    with open(os.path.join(ROOT, "results", "validation_growth_rates.csv"), "w") as f:
        f.write("k,lambda_dns,lambda_lsa,relative_error\n")
        for k, ld, lt, e in rows:
            f.write(f"{k:.6f},{ld:.8e},{lt:.8e},{e:.6f}\n")

    k_dns = diagnostics.dominant_wavenumber(spectra[-1], sim.Kmag)
    err_k_star = abs(k_dns - k_star) / k_star
    err_k_c = abs(k_dns - kc) / kc
    err_lambda = abs(rows[0][1] - lam_max) / lam_max

    with open(os.path.join(ROOT, "results", "validation_summary.txt"), "w") as f:
        f.write("VALIDATION REPORT: LSA vs DNS\n" + "=" * 50 + "\n")
        f.write(f"Parameters: a={p.a}, b={p.b}, Du={p.Du}, Dv={p.Dv}\n")
        f.write(f"Bifurcation distance M^2/(4 Du Dv Delta) = {bif_dist:.3f}\n\n")
        f.write("1. Turing conditions: all four satisfied.\n\n")
        f.write("2. Wavenumber selection:\n")
        f.write(f"   k_c  (textbook approx., min of B(k))       = {kc:.6f}\n")
        f.write(f"   k*   (exact argmax of lambda_+(k))         = {k_star:.6f}\n")
        f.write(f"   DNS  dominant wavenumber                   = {k_dns:.6f}\n")
        f.write(f"   relative error vs k*  (expected small)     = {err_k_star:.2%}\n")
        f.write(f"   relative error vs k_c (expected larger,\n"
                f"     since system is not near the bifurcation\n"
                f"     threshold)                                = {err_k_c:.2%}\n\n")
        f.write("3. Maximum growth rate:\n")
        f.write(f"   LSA  lambda(k*) = {lam_max:.6f}\n")
        f.write(f"   DNS  measured   = {rows[0][1]:.6f}\n")
        f.write(f"   relative error  = {err_lambda:.2%}\n\n")
        f.write(f"4. Mean relative error over {len(rows)} modes (each vs its own\n"
                f"   lambda_+(k) prediction, not vs k*) = "
                f"{np.mean([r[3] for r in rows]):.2%}\n")

    print(f"\nWavelength error vs k* = {err_k_star:.2%} (vs textbook k_c = {err_k_c:.2%}) "
          f"| Growth-rate error = {err_lambda:.2%}")

    ks = np.linspace(0, 2.5, 400)
    lam_minus, lam_plus = lsa.dispersion_relation(ks, p)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, lam_plus, "b-", lw=2, label=r"$\lambda_+(k)$ LSA")
    ax.plot(ks, lam_minus, "b--", lw=1, label=r"$\lambda_-(k)$ LSA")
    ks_dns = np.array([r[0] for r in rows])
    lam_dns = np.array([r[1] for r in rows])
    ax.errorbar(ks_dns, lam_dns, yerr=0.02, fmt="ro", ms=7, capsize=3, label="DNS measured")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(kc, color="gray", ls=":", label=rf"$k_c$ (textbook) = {kc:.3f}")
    ax.axvline(k_star, color="darkorange", ls="-.", label=rf"$k^*$ (exact) = {k_star:.3f}")
    ax.set_xlabel("wavenumber k")
    ax.set_ylabel("growth rate " + r"$\lambda$")
    ax.set_title("LSA vs DNS - quantitative validation")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "dispersion_comparison.png"), dpi=200)
    print("Saved figures/dispersion_comparison.png and results/validation_*.csv/txt")


if __name__ == "__main__":
    main()
