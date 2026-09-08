"""Pseudo-spectral DNS of the Schnakenberg system.

Reaction term integrated with classical RK4; diffusion term integrated
exactly in Fourier space via an integrating factor (Strang-style
operator splitting). Periodic boundary conditions on [0, L]^2.
"""
import numpy as np
from .parameters import SchnakenbergParams


class SpectralSolver:
    def __init__(self, params: SchnakenbergParams, N: int = 256,
                 L: float = 100.0, dt: float = 0.1, seed: int = 42):
        if N & (N - 1):
            raise ValueError("N must be a power of 2.")
        self.p, self.N, self.L, self.dt = params, N, L, dt

        x = np.arange(N) * L / N
        self.X, self.Y = np.meshgrid(x, x, indexing="ij")

        k1d = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
        self.KX, self.KY = np.meshgrid(k1d, k1d, indexing="ij")
        self.K2 = self.KX ** 2 + self.KY ** 2
        self.Kmag = np.sqrt(self.K2)

        self.rng = np.random.default_rng(seed)

    def initial_condition(self, amplitude: float = 1e-3):
        def noisy(field):
            noise = self.rng.standard_normal((self.N, self.N))
            return field + amplitude * (noise - noise.mean())
        return noisy(self.p.u_star), noisy(self.p.v_star)

    def _reaction_rhs(self, u, v):
        p = self.p
        uv2 = u * u * v
        return p.a - u + uv2, p.b - uv2

    def step(self, u, v):
        dt = self.dt
        # --- Reaction: classical RK4 ---
        r1u, r1v = self._reaction_rhs(u, v)
        r2u, r2v = self._reaction_rhs(u + 0.5 * dt * r1u, v + 0.5 * dt * r1v)
        r3u, r3v = self._reaction_rhs(u + 0.5 * dt * r2u, v + 0.5 * dt * r2v)
        r4u, r4v = self._reaction_rhs(u + dt * r3u, v + dt * r3v)
        u = u + dt / 6.0 * (r1u + 2 * r2u + 2 * r3u + r4u)
        v = v + dt / 6.0 * (r1v + 2 * r2v + 2 * r3v + r4v)

        # --- Diffusion: exact integrating factor in Fourier space ---
        u = np.real(np.fft.ifft2(np.fft.fft2(u) * np.exp(-self.p.Du * self.K2 * dt)))
        v = np.real(np.fft.ifft2(np.fft.fft2(v) * np.exp(-self.p.Dv * self.K2 * dt)))
        return u, v

    def run(self, t_end: float, record_every: int = 5, snapshot_times=()):
        """Integrate to t_end. Returns (times, spectra_of_u, snapshots).

        snapshots is a list of (t, u_copy) close to each requested time.
        """
        u, v = self.initial_condition()
        times, spectra, snapshots = [], [], []
        n_steps = int(round(t_end / self.dt))

        for n in range(n_steps + 1):
            t = n * self.dt
            if n % record_every == 0:
                times.append(t)
                spectra.append(np.fft.fft2(u))
            for ts in snapshot_times:
                if abs(t - ts) < self.dt / 2 and (
                        not snapshots or abs(snapshots[-1][0] - ts) > self.dt):
                    snapshots.append((t, u.copy()))
            if n < n_steps:
                u, v = self.step(u, v)

        return np.array(times), np.array(spectra), snapshots
