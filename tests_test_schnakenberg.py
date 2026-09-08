import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.schnakenberg import SchnakenbergParams, lsa, solver, diagnostics


# ---------------------------------------------------------------------------
# parameters.py
# ---------------------------------------------------------------------------

def test_steady_state_matches_definition():
    p = SchnakenbergParams(a=0.1, b=0.9)
    assert p.u_star == pytest.approx(1.0)
    assert p.v_star == pytest.approx(0.9)


def test_steady_state_is_fixed_point_of_reaction_kinetics():
    """At (u*, v*), the reaction terms a - u + u^2 v and b - u^2 v vanish."""
    p = SchnakenbergParams(a=0.1, b=0.9)
    u, v = p.u_star, p.v_star
    assert (p.a - u + u ** 2 * v) == pytest.approx(0.0, abs=1e-12)
    assert (p.b - u ** 2 * v) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# lsa.py
# ---------------------------------------------------------------------------

def test_default_params_satisfy_turing_conditions():
    p = SchnakenbergParams()
    conditions = lsa.turing_conditions(p)
    for name, val in conditions.items():
        if isinstance(val, bool):
            assert val, f"Turing condition failed: {name}"


def test_turing_condition_booleans_are_plain_python_bool():
    """Regression test: dispersion_coefficients() fields are numpy.float64,
    so comparisons like `c.M > 0.0` produce numpy.bool_, not Python bool.
    `isinstance(numpy.bool_(True), bool)` is False, so any caller that
    branches on `isinstance(val, bool)` (as the report/print scripts do,
    to separate PASS/FAIL flags from numeric fields) would silently print
    the flags as 1.0/0.0 instead of PASS/FAIL unless turing_conditions()
    casts them to plain bool explicitly."""
    p = SchnakenbergParams()
    conditions = lsa.turing_conditions(p)
    for name, val in conditions.items():
        if name in ("T", "Delta", "M"):
            continue
        assert type(val) is bool, f"{name} is {type(val)}, expected plain bool"


def test_no_turing_instability_without_differential_diffusion():
    """With Du == Dv, diffusion cannot destabilize a reaction-stable
    steady state (M <= 0), so Turing instability should not occur."""
    p = SchnakenbergParams(a=0.1, b=0.9, Du=1.0, Dv=1.0)
    conditions = lsa.turing_conditions(p)
    assert not conditions["M_squared_condition (M^2 > 4 Du Dv Delta)"] or conditions["M"] <= 0


def test_dispersion_relation_at_k_zero_matches_reaction_jacobian_eigenvalues():
    p = SchnakenbergParams()
    lam_minus, lam_plus = lsa.dispersion_relation(np.array([0.0]), p)
    J = lsa.jacobian(p)
    eigs = np.sort(np.linalg.eigvals(J).real)
    assert lam_minus[0] == pytest.approx(eigs[0], abs=1e-8)
    assert lam_plus[0] == pytest.approx(eigs[1], abs=1e-8)


def test_fastest_growing_wavenumber_maximizes_growth_rate():
    """k* (the numerically exact maximizer) must beat lambda_+ at nearby k,
    including near the textbook k_c -- which, away from the bifurcation
    threshold, is *not* itself the true maximizer (see
    test_critical_wavenumber_differs_from_fastest_growing_deep_in_regime)."""
    p = SchnakenbergParams()
    k_star = lsa.fastest_growing_wavenumber(p)
    lam_at_kstar = lsa.dispersion_relation(np.array([k_star]), p)[1][0]
    ks_nearby = np.array([k_star - 0.05, k_star + 0.05, lsa.critical_wavenumber(p)])
    lam_nearby = lsa.dispersion_relation(ks_nearby, p)[1]
    assert lam_at_kstar >= lam_nearby.max()


def test_critical_wavenumber_equals_fastest_growing_at_bifurcation_threshold():
    """At marginal stability (M^2 == 4 Du Dv Delta exactly), the textbook
    k_c formula IS the exact maximizer of lambda_+(k), since lambda_+ = 0
    there and the maximum of B(k) alone determines where instability first
    turns on."""
    # Tune Dv so that M^2 == 4 Du Dv Delta for fixed a, b, Du.
    a, b, Du = 0.1, 0.9, 1.0
    from scipy.optimize import brentq

    def f(Dv):
        p = SchnakenbergParams(a=a, b=b, Du=Du, Dv=Dv)
        c = lsa.dispersion_coefficients(p)
        return c.M ** 2 - 4 * Du * Dv * c.Delta

    Dv_threshold = brentq(f, 5.0, 100.0)
    p = SchnakenbergParams(a=a, b=b, Du=Du, Dv=Dv_threshold)
    assert lsa.bifurcation_distance(p) == pytest.approx(1.0, rel=1e-4)
    kc = lsa.critical_wavenumber(p)
    k_star = lsa.fastest_growing_wavenumber(p)
    assert kc == pytest.approx(k_star, rel=1e-3)


def test_critical_wavenumber_differs_from_fastest_growing_deep_in_regime():
    """For the default (deep-in-the-unstable-regime) parameters, the
    textbook k_c and the true fastest-growing k* are meaningfully
    different -- this is the behaviour a DNS validation must account for."""
    p = SchnakenbergParams()
    assert lsa.bifurcation_distance(p) > 2.0
    kc = lsa.critical_wavenumber(p)
    k_star = lsa.fastest_growing_wavenumber(p)
    assert abs(kc - k_star) / k_star > 0.1


def test_max_growth_rate_is_positive_for_turing_unstable_params():
    p = SchnakenbergParams()
    assert lsa.max_growth_rate(p) > 0.0


def test_max_growth_rate_uses_fastest_growing_not_textbook_kc():
    """max_growth_rate should report lambda_+ at k* (true max), which for
    the default deep-in-regime parameters is strictly larger than
    lambda_+ at the textbook k_c."""
    p = SchnakenbergParams()
    lam_at_kc = lsa.dispersion_relation(np.array([lsa.critical_wavenumber(p)]), p)[1][0]
    assert lsa.max_growth_rate(p) > lam_at_kc


# ---------------------------------------------------------------------------
# solver.py
# ---------------------------------------------------------------------------

def test_solver_rejects_non_power_of_two_grid():
    p = SchnakenbergParams()
    with pytest.raises(ValueError):
        solver.SpectralSolver(p, N=100)


def test_solver_run_returns_consistent_shapes():
    p = SchnakenbergParams()
    sim = solver.SpectralSolver(p, N=32, L=50.0, dt=0.1, seed=1)
    times, spectra, snapshots = sim.run(t_end=2.0, record_every=2, snapshot_times=[0.0, 1.0])
    assert times.shape[0] == spectra.shape[0]
    assert spectra.shape[1:] == (32, 32)
    assert len(snapshots) >= 1


def test_uniform_field_is_a_fixed_point_of_the_solver():
    """Starting exactly at the homogeneous steady state (no noise), the
    field should remain (numerically) constant after one step, since both
    the reaction term and the Laplacian vanish identically."""
    p = SchnakenbergParams()
    sim = solver.SpectralSolver(p, N=16, L=20.0, dt=0.05, seed=0)
    u0 = np.full((16, 16), p.u_star)
    v0 = np.full((16, 16), p.v_star)
    u1, v1 = sim.step(u0, v0)
    assert np.allclose(u1, p.u_star, atol=1e-10)
    assert np.allclose(v1, p.v_star, atol=1e-10)


# ---------------------------------------------------------------------------
# diagnostics.py
# ---------------------------------------------------------------------------

def test_pattern_wavelength_is_two_pi_over_k():
    assert diagnostics.pattern_wavelength(2 * np.pi) == pytest.approx(1.0)


def test_dominant_wavenumber_excludes_dc_component():
    """Regression test: the k=0 (DC / spatial-mean) component of fft2(u)
    is essentially always the largest-magnitude entry, since it scales
    with mean(u) * N^2 rather than the pattern amplitude. A naive global
    argmax over |spectrum| would therefore report k=0 (undefined
    wavelength) even when a clear pattern is present elsewhere in the
    spectrum; dominant_wavenumber must mask it out."""
    N = 16
    spectrum = np.zeros((N, N), dtype=complex)
    spectrum[0, 0] = 1000.0          # huge DC component
    spectrum[2, 3] = 50.0            # the actual pattern mode
    kmag = np.zeros((N, N))
    kmag[2, 3] = 0.77                # some nonzero wavenumber
    # kmag[0, 0] stays 0.0, as it does for a real Kmag grid.
    result = diagnostics.dominant_wavenumber(spectrum, kmag)
    assert result == pytest.approx(0.77)


def test_dns_growth_rate_matches_lsa_within_tolerance():
    """End-to-end regression test: run a short DNS and check that the
    measured growth rate of the dominant mode is close to the LSA
    prediction lambda_+(k)."""
    p = SchnakenbergParams()
    sim = solver.SpectralSolver(p, N=64, L=100.0, dt=0.1, seed=7)
    times, spectra, _ = sim.run(t_end=40.0, record_every=2)
    measured = diagnostics.mode_growth_rates(times, spectra, sim.Kmag, n_modes=3)
    assert len(measured) > 0
    for m in measured:
        lam_th = float(lsa.dispersion_relation(np.array([m["k"]]), p)[1][0])
        rel_err = abs(m["growth_rate_dns"] - lam_th) / max(abs(lam_th), 1e-12)
        assert rel_err < 0.15, f"mode k={m['k']:.3f}: rel. error {rel_err:.2%} too high"
