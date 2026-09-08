#!/usr/bin/env python3
"""Step 1: Linear stability analysis of the Schnakenberg system.

Computes the Turing instability conditions, the critical wavenumber k_c,
and the dispersion relation lambda(k); saves numeric results to results/
and a plot to figures/.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.schnakenberg import SchnakenbergParams, lsa

ROOT = os.path.join(os.path.dirname(__file__), "..")
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)


def main():
    p = SchnakenbergParams()

    print("=" * 60)
    print("TURING CONDITIONS (LSA)")
    print("=" * 60)
    conditions = lsa.turing_conditions(p)
    for name, val in conditions.items():
        if isinstance(val, bool):
            print(f"  {name:45s} -> {'PASS' if val else 'FAIL'}")
        else:
            print(f"  {name:45s} = {val: .4f}")

    kc = lsa.critical_wavenumber(p)
    k_star = lsa.fastest_growing_wavenumber(p)
    lam_max = lsa.max_growth_rate(p)
    bif_dist = lsa.bifurcation_distance(p)
    print(f"\n  k_c   (textbook critical wavenumber, min of B(k)) = {kc:.4f}")
    print(f"  k*    (numerically exact argmax of lambda_+(k))   = {k_star:.4f}")
    print(f"  wavelength 2*pi/k*                                = {2 * np.pi / k_star:.3f}")
    print(f"  lambda(k*) max growth                             = {lam_max:.4f}")
    print(f"  bifurcation distance M^2/(4 Du Dv Delta)          = {bif_dist:.3f}"
          f"  (1.0 = marginal; k_c == k* only at 1.0)")
    if bif_dist > 1.5:
        print("  NOTE: system is well inside the unstable regime, so k_c "
              "(textbook formula) and k* (true fastest-growing mode) differ "
              "meaningfully. DNS pattern selection should be compared against k*.")

    # --- dispersion relation data ---
    ks = np.linspace(0, 2.5, 500)
    lam_minus, lam_plus = lsa.dispersion_relation(ks, p)
    with open(os.path.join(ROOT, "results", "dispersion_data.csv"), "w") as f:
        f.write("k,lambda_minus,lambda_plus\n")
        for k, lm, lp in zip(ks, lam_minus, lam_plus):
            f.write(f"{k:.6f},{lm:.8e},{lp:.8e}\n")

    with open(os.path.join(ROOT, "results", "lsa_summary.txt"), "w") as f:
        f.write("LINEAR STABILITY ANALYSIS SUMMARY\n")
        f.write(f"Parameters: a={p.a}, b={p.b}, Du={p.Du}, Dv={p.Dv}\n")
        f.write(f"Steady state: u*={p.u_star:.6f}, v*={p.v_star:.6f}\n\n")
        for name, val in conditions.items():
            if isinstance(val, bool):
                f.write(f"  {name:45s} -> {'PASS' if val else 'FAIL'}\n")
            else:
                f.write(f"  {name:45s} = {val: .6f}\n")
        f.write(f"\nTextbook critical wavenumber : k_c = {kc:.6f}\n")
        f.write(f"Exact fastest-growing mode   : k*  = {k_star:.6f}\n")
        f.write(f"Wavelength 2*pi/k*           = {2 * np.pi / k_star:.6f}\n")
        f.write(f"Max growth rate lambda(k*)   = {lam_max:.6f}\n")
        f.write(f"Bifurcation distance M^2/(4DuDvDelta) = {bif_dist:.6f}\n")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(ks, lam_plus, "b-", lw=2.2, label=r"$\lambda_+(k)$")
    ax.plot(ks, lam_minus, "b--", lw=1.2, label=r"$\lambda_-(k)$")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(kc, color="gray", ls=":", label=rf"$k_c$ (textbook) $= {kc:.4f}$")
    ax.axvline(k_star, color="crimson", ls="-.", label=rf"$k^*$ (exact) $= {k_star:.4f}$")
    ax.fill_between(ks, 0, lam_plus, where=lam_plus > 0,
                     alpha=0.15, color="red", label="unstable band")
    ax.set_xlabel("wavenumber k")
    ax.set_ylabel("growth rate " + r"$\lambda$")
    ax.set_title("Schnakenberg dispersion relation")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "dispersion_relation.png"), dpi=200)
    print("\nSaved results/dispersion_data.csv, results/lsa_summary.txt, "
          "figures/dispersion_relation.png")


if __name__ == "__main__":
    main()
