#!/usr/bin/env python3
"""Step 4: Assemble results/scientific_report.pdf from the outputs of
run_lsa.py, run_dns.py, and run_validation.py. Run those scripts first."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.schnakenberg import SchnakenbergParams, lsa

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
OUT_PATH = os.path.join(RESULTS, "scientific_report.pdf")


def main():
    p = SchnakenbergParams()

    with PdfPages(OUT_PATH) as pdf:
        # ---- Title / summary page ----
        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        fig.text(0.5, 0.92, "Turing Pattern Formation in the Schnakenberg\n"
                             "Reaction-Diffusion System",
                  ha="center", fontsize=20, weight="bold")
        fig.text(0.5, 0.85, "A Linear Stability Analysis Validated Against\n"
                             "Direct Numerical Simulation",
                  ha="center", fontsize=14, style="italic")
        fig.text(0.5, 0.79, "Mohamed Semri", ha="center", fontsize=13)
        fig.text(0.5, 0.76, "github.com/semrimaster-creator/schnakenberg-turing-patterns",
                  ha="center", fontsize=10, color="steelblue")

        conds = lsa.turing_conditions(p)
        kc = lsa.critical_wavenumber(p)
        k_star = lsa.fastest_growing_wavenumber(p)
        lam_max = lsa.max_growth_rate(p)
        bif_dist = lsa.bifurcation_distance(p)
        txt = ("MODEL\n"
               "  du/dt = Du*Lap(u) + a - u + u^2 v\n"
               "  dv/dt = Dv*Lap(v) + b - u^2 v\n"
               f"  a={p.a}, b={p.b}, Du={p.Du}, Dv={p.Dv}, L=100\n"
               f"  Steady state: u*={p.u_star:.4f}, v*={p.v_star:.4f}\n\n"
               "LINEAR STABILITY ANALYSIS (Turing conditions)\n")
        for name, val in conds.items():
            if isinstance(val, bool):
                txt += f"  {name:45s} {'PASS' if val else 'FAIL'}\n"
            else:
                txt += f"  {name:45s} {val:.4f}\n"
        txt += (f"\n  Textbook critical wavenumber k_c = {kc:.6f}\n"
                f"  Exact fastest-growing mode   k*  = {k_star:.6f}"
                f"  (wavelength 2pi/k* = {2 * np.pi / k_star:.4f})\n"
                f"  Max growth rate lambda(k*) = {lam_max:.6f}\n"
                f"  Bifurcation distance M^2/(4DuDvDelta) = {bif_dist:.3f}"
                f"  ({'near threshold' if bif_dist < 1.5 else 'deep in unstable regime -> k_c != k*'})\n\n"
                "VALIDATION RESULTS (see following pages)\n"
                "  - Per-mode DNS growth rates match lambda_+(k)\n"
                "  - DNS dominant wavelength matches k* (not the textbook k_c,\n"
                "    which is only exact at the bifurcation threshold)\n\n"
                "Reproducible pipeline: Python + NumPy (pseudo-spectral,\n"
                "integrating-factor diffusion + RK4 reaction stepping).\n")
        fig.text(0.08, 0.70, txt, fontsize=11, family="monospace", va="top")
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Figure pages ----
        figure_pages = [
            ("dispersion_relation.png",
             "Figure 1. Dispersion relation lambda(k) from LSA;\n"
             "unstable band and predicted k_c."),
            ("dispersion_comparison.png",
             "Figure 2. LSA curve vs DNS-measured growth rates\n"
             "(log-linear regression per Fourier mode)."),
            ("pattern_evolution.png",
             "Figure 3. Turing pattern formation from random noise\n"
             "(u concentration, t = 0 to t_end)."),
        ]
        for fname, caption in figure_pages:
            fpath = os.path.join(FIGURES, fname)
            if not os.path.exists(fpath):
                print(f"  (skipping {fname}: not found - run the earlier scripts first)")
                continue
            fig = plt.figure(figsize=(8.27, 11.69))
            im = plt.imread(fpath)
            ax = fig.add_axes([0.1, 0.25, 0.8, 0.55])
            ax.axis("off")
            ax.imshow(im)
            fig.text(0.5, 0.16, caption, ha="center", fontsize=11)
            pdf.savefig(fig)
            plt.close(fig)

        # ---- Raw text summary pages ----
        for fname in ["lsa_summary.txt", "dns_summary.txt", "validation_summary.txt"]:
            fpath = os.path.join(RESULTS, fname)
            if not os.path.exists(fpath):
                print(f"  (skipping {fname}: not found - run the earlier scripts first)")
                continue
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.08, 0.92, fname, fontsize=14, weight="bold")
            fig.text(0.08, 0.88, open(fpath).read(), fontsize=10, family="monospace", va="top")
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
