"""Diagnostics: growth rates and wavelength extracted from a DNS run."""
import numpy as np


def mode_growth_rates(times, spectra, kmag, n_modes: int = 8,
                       amp_lo: float = 1e-4, amp_hi: float = 1e-1):
    """Estimate the exponential growth rate of the strongest Fourier modes
    by a log-linear fit over their early, still-linear growth window."""
    final_power = np.abs(spectra[-1])
    flat_idx = np.argsort(final_power.ravel())[::-1]
    results, seen = [], set()

    for idx in flat_idx:
        if len(results) >= n_modes:
            break
        iy, ix = divmod(idx, spectra.shape[-1])
        km = kmag[iy, ix]
        if not np.isfinite(km) or km == 0.0:
            # k=0 is the spatial mean, not a growing/decaying pattern mode.
            continue
        key = round(float(km), 3)
        if key in seen:
            continue
        seen.add(key)

        A = np.abs(spectra[:, iy, ix])
        mask = (A > amp_lo * A.max()) & (A < amp_hi * A.max()) & (A > 1e-12)
        if mask.sum() < 5:
            continue
        slope, _ = np.polyfit(times[mask], np.log(A[mask]), 1)
        results.append({"k": float(km), "growth_rate_dns": float(slope),
                         "n_points": int(mask.sum())})
    return results


def dominant_wavenumber(spectrum, kmag) -> float:
    """Wavenumber of the largest-amplitude Fourier mode, excluding k=0.

    The k=0 (DC) component encodes the spatial mean of the field, not a
    pattern; its magnitude is proportional to the mean concentration times
    the number of grid points and is essentially always the single largest
    entry in |fft2(u)|, regardless of whether a Turing pattern has formed.
    Including it would make this function report k=0 (an undefined,
    infinite "wavelength") for every field. It is therefore masked out
    here, together with any non-finite wavenumbers.
    """
    power = np.abs(spectrum).copy()
    mask = ~np.isfinite(kmag) | (kmag == 0.0)
    power[mask] = 0.0
    iy, ix = np.unravel_index(np.argmax(power), power.shape)
    return float(kmag[iy, ix])


def pattern_wavelength(k_dominant: float) -> float:
    return 2.0 * np.pi / k_dominant
