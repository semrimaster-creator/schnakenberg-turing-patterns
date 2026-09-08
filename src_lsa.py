"""Linear stability analysis of the Schnakenberg system."""
from dataclasses import dataclass
import numpy as np
from .parameters import SchnakenbergParams


def steady_state(p: SchnakenbergParams):
    return p.u_star, p.v_star


def jacobian(p: SchnakenbergParams) -> np.ndarray:
    u, v = steady_state(p)
    return np.array([[2.0 * u * v - 1.0, u * u],
                      [-2.0 * u * v, -u * u]])


@dataclass(frozen=True)
class DispersionCoefficients:
    T: float
    Delta: float
    S: float
    M: float


def dispersion_coefficients(p: SchnakenbergParams) -> DispersionCoefficients:
    J = jacobian(p)
    return DispersionCoefficients(
        T=float(np.trace(J)),
        Delta=float(np.linalg.det(J)),
        S=p.Du + p.Dv,
        M=p.Dv * J[0, 0] + p.Du * J[1, 1],
    )


def dispersion_relation(k, p: SchnakenbergParams):
    """Return (lambda_minus, lambda_plus) for wavenumber array k."""
    c = dispersion_coefficients(p)
    k = np.asarray(k, dtype=float)
    A = c.T - c.S * k ** 2
    B = c.Delta - c.M * k ** 2 + p.Du * p.Dv * k ** 4
    disc = np.maximum(A ** 2 - 4.0 * B, 0.0)
    root = np.sqrt(disc)
    return 0.5 * (A - root), 0.5 * (A + root)


def turing_conditions(p: SchnakenbergParams) -> dict:
    """Note: comparisons are cast to plain `bool` explicitly, since
    `dispersion_coefficients` fields are `numpy.float64` and
    `numpy.float64 > 0` returns `numpy.bool_`, which is NOT an instance of
    Python's built-in `bool` (`isinstance(np.bool_(True), bool)` is False).
    Callers that branch on `isinstance(val, bool)` to distinguish the
    PASS/FAIL entries from the numeric ones would otherwise silently
    mis-format the booleans as numbers.
    """
    c = dispersion_coefficients(p)
    return {
        "trace_negative (T < 0)": bool(c.T < 0.0),
        "det_positive (Delta > 0)": bool(c.Delta > 0.0),
        "M_positive (M > 0)": bool(c.M > 0.0),
        "M_squared_condition (M^2 > 4 Du Dv Delta)":
            bool(c.M ** 2 > 4.0 * p.Du * p.Dv * c.Delta),
        "T": c.T,
        "Delta": c.Delta,
        "M": c.M,
    }


def critical_wavenumber(p: SchnakenbergParams) -> float:
    """The wavenumber that minimizes B(k) = Delta - M k^2 + Du Dv k^4.

    This is the standard textbook "critical wavenumber" (e.g. Murray,
    *Mathematical Biology II*, Ch. 2): k_c^2 = M / (2 Du Dv). It is the
    EXACT most-unstable mode only at marginal stability, i.e. exactly on
    the Turing bifurcation threshold where M^2 = 4 Du Dv Delta. Away from
    that threshold it is an approximation, because it maximizes only the
    k-dependent part of B(k) and ignores the k-dependence of A(k) = T - S k^2
    in lambda_+(k) = 0.5[A(k) + sqrt(A(k)^2 - 4B(k))]. See
    `fastest_growing_wavenumber` for the numerically exact maximizer of
    lambda_+(k), which is what a DNS pattern-selection experiment actually
    measures once the system is well inside the unstable regime.
    """
    c = dispersion_coefficients(p)
    kc2 = c.M / (2.0 * p.Du * p.Dv)
    return float(np.sqrt(max(kc2, 0.0)))


# Backward-compatible alias: earlier versions of this module called the
# textbook approximation "most_unstable_wavenumber". Kept for compatibility;
# prefer `critical_wavenumber` (exact name) or `fastest_growing_wavenumber`
# (numerically exact maximizer) in new code.
most_unstable_wavenumber = critical_wavenumber


def fastest_growing_wavenumber(p: SchnakenbergParams, k_max: float = 5.0) -> float:
    """Numerically exact maximizer of lambda_+(k), found by direct 1-D
    optimization rather than the closed-form approximation in
    `critical_wavenumber`. This is the wavenumber whose growth rate DNS
    should reproduce, and it can differ substantially from
    `critical_wavenumber` when the system sits well inside the Turing-
    unstable region (M^2 >> 4 Du Dv Delta) rather than near the
    bifurcation threshold (M^2 approx 4 Du Dv Delta).
    """
    from scipy.optimize import minimize_scalar

    def neg_lambda_plus(k):
        return -float(dispersion_relation(np.array([k]), p)[1][0])

    res = minimize_scalar(neg_lambda_plus, bounds=(1e-6, k_max), method="bounded",
                           options={"xatol": 1e-10})
    return float(res.x)


def max_growth_rate(p: SchnakenbergParams) -> float:
    """Growth rate at the numerically exact fastest-growing wavenumber."""
    k_star = fastest_growing_wavenumber(p)
    return float(dispersion_relation(np.array([k_star]), p)[1][0])


def bifurcation_distance(p: SchnakenbergParams) -> float:
    """Ratio M^2 / (4 Du Dv Delta). Equals 1.0 exactly at the Turing
    bifurcation threshold; values >> 1 mean the system is deep inside the
    unstable regime, where `critical_wavenumber` and
    `fastest_growing_wavenumber` are expected to diverge."""
    c = dispersion_coefficients(p)
    return float(c.M ** 2 / (4.0 * p.Du * p.Dv * c.Delta))
