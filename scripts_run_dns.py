#!/usr/bin/env python3
"""Step 2: Direct numerical simulation of Turing pattern formation.

Runs the pseudo-spectral solver from a noisy initial condition, records
snapshots for a montage figure, and saves diagnostics (dominant wavenumber
and pattern wavelength over time) comparing the emergent pattern to the
LSA prediction.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.schnakenberg import SchnakenbergParams, solver, diagnostics, lsa

ROOT = os.path.join(os.path.dirname(__file__), "..")
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=256, help="grid points per side")
    parser.add_argument("--L", type=float, default=100.0, help="domain size")
    parser.add_argument("--dt", type=float, default=0.05, help="time step")
    parser.add_argument("--t-end", type=float, default=80.0, help="final time")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed")
    args = parser.parse_args()

    p = SchnakenbergParams()
    # Compare against the numerically exact fastest-growing mode k*, not the
    # textbook critical-wavenumber approximation k_c: away from the Turing
    # bifurcation threshold the two differ (see lsa.bifurcation_distance),
    # and it is k* that a DNS pattern-selection experiment actually measures.
    k_star_analytic = lsa.fastest_growing_wavenumber(p)

    sim = solver.SpectralSolver(p, N=args.N, L=args.L, dt=args.dt, seed=args.seed)

    records = []
    u, v = sim.initial_condition()
    n_steps = int(round(args.t_end / args.dt))
    print(f"Integrating {n_steps} steps on a {args.N}x{args.N} grid ...")

    for n in range(n_steps + 1):
        t = n * args.dt
        if n % 20 == 0:
            records.append((t, u.copy(), np.fft.fft2(u)))
            if n % 200 == 0:
                print(f"  t = {t:.1f}")
        if n < n_steps:
            u, v = sim.step(u, v)
    print("Simulation complete.")

    with open(os.path.join(ROOT, "results", "dns_diagnostics.csv"), "w") as f:
        f.write("t,dominant_k,pattern_wavelength,mean_u,max_u\n")
        for t, ur, fu in records:
            k_dom = diagnostics.dominant_wavenumber(fu, sim.Kmag)
            lam = diagnostics.pattern_wavelength(k_dom) if k_dom > 0 else float("nan")
            f.write(f"{t:.4f},{k_dom:.6f},{lam:.6f},{ur.mean():.6f},{ur.max():.6f}\n")

    k_final = diagnostics.dominant_wavenumber(records[-1][2], sim.Kmag)
    lam_final = 2 * np.pi / k_final
    err_k = abs(k_final - k_star_analytic) / k_star_analytic

    with open(os.path.join(ROOT, "results", "dns_summary.txt"), "w") as f:
        f.write("DNS SUMMARY\n")
        f.write(f"Grid {args.N}x{args.N}, L={args.L}, dt={args.dt}, t_end={args.t_end}\n\n")
        f.write(f"LSA predicted k* (exact fastest-growing mode) : {k_star_analytic:.6f}\n")
        f.write(f"DNS dominant wavenumber   : {k_final:.6f}\n")
        f.write(f"DNS pattern wavelength    : {lam_final:.6f}\n")
        f.write(f"Relative wavelength error : {err_k:.2%}\n")

    print(f"\nk* (LSA) = {k_star_analytic:.4f}  |  k (DNS) = {k_final:.4f}  "
          f"|  relative error = {err_k:.2%}")

    snapshot_ts = [0.0, 5.0, 20.0, 40.0, args.t_end]
    fig, axes = plt.subplots(1, len(snapshot_ts), figsize=(17, 3.6))
    for ax, t_target in zip(axes, snapshot_ts):
        t, ur, _ = min(records, key=lambda r: abs(r[0] - t_target))
        im = ax.imshow(ur.T, origin="lower", extent=[0, args.L, 0, args.L], cmap="viridis")
        ax.set_title(f"t = {t:.0f}")
        ax.set_xticks([0, args.L])
        ax.set_yticks([0, args.L])
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("Turing pattern formation - u concentration", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "pattern_evolution.png"), dpi=200)
    print("Saved figures/pattern_evolution.png and results/dns_diagnostics.csv")


if __name__ == "__main__":
    main()
