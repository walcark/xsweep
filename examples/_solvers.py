"""Small radiative-transfer solvers used as the engine in every example.

The gallery needs an engine that is *genuinely* awkward to vectorise,
otherwise it would be arguing against xsweep: a closed-form expression over a
parameter grid is a numpy one-liner and needs no sweep library at all. Two
solvers are provided, and every example calls one of them.

``mc_reflectance``
    A Monte-Carlo photon transport solver. One photon is a ``while`` loop
    whose length depends on the random numbers it draws, so there is no array
    shape to broadcast over: this is the case xsweep exists for. Its cost is
    set by ``n_photons``, which the gallery keeps small enough to build a
    website with. A production run means seconds to minutes per point, and
    that regime is what makes a cache, a resume path and a cost estimate
    worth having.

``spectral_doubling``
    A two-stream doubling solver. Every operation in it is elementwise, so it
    costs the same on a 500-wavelength spectrum as on one wavelength: hand it
    the whole axis. It is a sequential recursion in optical thickness, and a
    real engine takes one atmospheric configuration at a time, so
    configurations stay separate calls. That mixture, one axis handed over
    whole and the rest swept point by point, is what ``vec`` and ``loop``
    exist to express.

None of this is calibrated against a published dataset or a real instrument.
It is textbook physics at textbook accuracy, chosen because it is short
enough to read and because its limits can be checked (see
``verify_solvers``), not because it would pass for an operational model.

References
----------
Henyey, L. G. & Greenstein, J. L. (1941). Diffuse radiation in the Galaxy.
*The Astrophysical Journal*, 93, 70-83.

van de Hulst, H. C. (1980). *Multiple Light Scattering: Tables, Formulas and
Applications*. Academic Press. (The doubling method and the adding
equations.)

Bodhaine, B. A., Wood, N. B., Dutton, E. G. & Slusser, J. R. (1999). On
Rayleigh optical depth calculations. *Journal of Atmospheric and Oceanic
Technology*, 16, 1854-1861.

Angstrom, A. (1929). On the atmospheric transmission of sun radiation and on
dust in the air. *Geografiska Annaler*, 11, 156-166.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "aerosol_optical_depth",
    "mc_reflectance",
    "rayleigh_optical_depth",
    "spectral_doubling",
]

_WEIGHT_CUTOFF = 1.0e-4
_ROULETTE_SURVIVAL = 0.25


def mc_reflectance(
    tau: float,
    ssa: float,
    g: float,
    mu0: float = 0.5,
    *,
    n_photons: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Trace photons through a homogeneous plane-parallel layer.

    Photons enter at the top travelling downwards with direction cosine
    ``mu0``. Between collisions the optical path is sampled as
    ``-log(u)``; absorption is handled by implicit capture (the weight is
    multiplied by the single-scattering albedo at every collision) and the
    scattering angle is drawn from the Henyey-Greenstein phase function by
    inverse-CDF sampling. Low-weight photons are terminated by Russian
    roulette, which keeps the estimator unbiased.

    The loop is the point: its length depends on the random draws, so this
    cannot be turned into an array operation over a parameter grid.

    Parameters
    ----------
    tau
        Optical thickness of the layer.
    ssa
        Single-scattering albedo, in [0, 1].
    g
        Henyey-Greenstein asymmetry factor, in (-1, 1). Zero is isotropic,
        positive is forward-scattering.
    mu0
        Cosine of the incidence angle, in (0, 1].
    n_photons
        Number of photons traced. The Monte-Carlo error falls as its square
        root, so this is the accuracy/cost knob.
    seed
        Seed of the per-call random generator, which makes a call
        reproducible and therefore cacheable.

    Returns
    -------
    dict of str to float
        ``reflectance`` and ``transmittance``, each the fraction of the
        incident flux leaving through the top and the bottom of the layer.
    """
    rng = np.random.default_rng(seed)
    reflected = 0.0
    transmitted = 0.0

    for _ in range(n_photons):
        depth = 0.0
        mu = mu0
        weight = 1.0
        while True:
            depth += -np.log(rng.random()) * mu
            if depth < 0.0:
                reflected += weight
                break
            if depth > tau:
                transmitted += weight
                break

            weight *= ssa
            if weight < _WEIGHT_CUTOFF:
                if rng.random() > _ROULETTE_SURVIVAL:
                    break
                weight /= _ROULETTE_SURVIVAL

            mu = _scatter(mu, g, rng)

    return {
        "reflectance": reflected / n_photons,
        "transmittance": transmitted / n_photons,
    }


def _scatter(mu: float, g: float, rng: np.random.Generator) -> float:
    """Return the direction cosine after one Henyey-Greenstein scattering.

    The scattering cosine is drawn by inverting the Henyey-Greenstein
    cumulative distribution, then rotated onto the vertical axis with a
    uniform azimuth.
    """
    u = rng.random()
    if abs(g) < 1.0e-6:
        cos_theta = 2.0 * u - 1.0
    else:
        ratio = (1.0 - g * g) / (1.0 - g + 2.0 * g * u)
        cos_theta = (1.0 + g * g - ratio * ratio) / (2.0 * g)

    phi = 2.0 * np.pi * rng.random()
    sin_mu = np.sqrt(max(0.0, 1.0 - mu * mu))
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    return float(mu * cos_theta + sin_mu * sin_theta * np.cos(phi))


def spectral_doubling(
    tau: Any,
    ssa: Any,
    g: Any,
    *,
    n_doublings: int = 16,
) -> dict[str, Any]:
    """Solve a homogeneous layer by two-stream doubling.

    A layer thin enough for single scattering has, to first order in its
    optical thickness ``dt``, reflectance and transmittance

    .. math::

        R = \\frac{\\omega \\beta\\, dt}{\\bar\\mu}, \\qquad
        T = 1 - \\frac{(1 - \\omega (1 - \\beta))\\, dt}{\\bar\\mu}

    with :math:`\\beta = (1 - g)/2` the backscattered fraction and
    :math:`\\bar\\mu = 1/2` the mean cosine of an isotropic diffuse flux.
    Stacking two identical layers is the adding recursion

    .. math::

        R_2 = R + \\frac{T^2 R}{1 - R^2}, \\qquad
        T_2 = \\frac{T^2}{1 - R^2}

    so ``n`` doublings starting from ``tau / 2**n`` reach the full layer.

    Every operation above is elementwise, so a whole spectrum costs what one
    wavelength costs: pass ``tau`` as an array and the recursion runs on all
    of it at once. This is what a ``vec`` axis is for.

    Parameters
    ----------
    tau
        Optical thickness, scalar or array (for instance one value per
        wavelength).
    ssa
        Single-scattering albedo, broadcastable against ``tau``.
    g
        Asymmetry factor, broadcastable against ``tau``.
    n_doublings
        Number of doublings, which sets the thin layer the recursion starts
        from at ``tau / 2**n_doublings``. Sixteen leaves the answer converged
        to about 1e-5 for the optical thicknesses used in this gallery.

    Returns
    -------
    dict of str to ndarray
        ``reflectance`` and ``transmittance``, shaped like ``tau``.

    Notes
    -----
    Two-stream is an approximation: the phase function enters only through
    the backscattered fraction, and the radiation field is reduced to an
    up-flux and a down-flux. It conserves energy exactly when ``ssa`` is 1,
    which is the check that matters most here, and it drifts from a full
    Monte-Carlo solution by a few percent of reflectance once scattering is
    strongly forward-peaked.

    The doubling count is a fixed number rather than one derived from
    ``tau``, so that slicing the wavelength axis cannot change the answer: a
    ``vec`` callee has to be safe to run on pieces of its axis, since that is
    exactly what a batch size does.
    """
    tau_arr = np.asarray(tau, dtype=float)
    ssa_arr = np.asarray(ssa, dtype=float)
    g_arr = np.asarray(g, dtype=float)

    dt = tau_arr / 2.0**n_doublings

    beta = 0.5 * (1.0 - g_arr)
    mu_bar = 0.5
    r = ssa_arr * beta * dt / mu_bar
    t = 1.0 - (1.0 - ssa_arr * (1.0 - beta)) * dt / mu_bar

    for _ in range(n_doublings):
        denom = 1.0 - r * r
        r, t = r + t * t * r / denom, t * t / denom

    return {"reflectance": r, "transmittance": t}


def rayleigh_optical_depth(wl_um: Any, pressure_hpa: float = 1013.25) -> Any:
    """Return the Rayleigh optical depth from the Bodhaine et al. (1999) fit.

    Parameters
    ----------
    wl_um
        Wavelength in micrometers, scalar or array.
    pressure_hpa
        Surface pressure in hPa; the fit is given at 1013.25 hPa and scales
        linearly with pressure.

    Returns
    -------
    ndarray
        Rayleigh optical depth, shaped like ``wl_um``.
    """
    wl = np.asarray(wl_um, dtype=float)
    numerator = 0.002152 * (1.0455996 - 341.29061 * wl**-2 - 0.90230850 * wl**2)
    denominator = 1.0 + 0.0027059889 * wl**-2 - 85.968563 * wl**2
    return numerator / denominator * (pressure_hpa / 1013.25)


def aerosol_optical_depth(wl_um: Any, aot550: float, alpha: float = 1.3) -> Any:
    """Return the aerosol optical depth from the Angstrom (1929) power law.

    Parameters
    ----------
    wl_um
        Wavelength in micrometers, scalar or array.
    aot550
        Aerosol optical depth at 550 nm.
    alpha
        Angstrom exponent; 1.3 is a common continental value.

    Returns
    -------
    ndarray
        Aerosol optical depth, shaped like ``wl_um``.
    """
    return aot550 * (np.asarray(wl_um, dtype=float) / 0.55) ** -alpha
