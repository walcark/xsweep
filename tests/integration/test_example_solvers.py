"""Physical checks on the toy solvers the documentation gallery runs on.

The gallery's whole argument is that its engine is a real, if small,
radiative-transfer solver rather than a stub. That claim is only worth
anything if the physics stays right, so the limits that must hold are
asserted here rather than checked once by hand.

The solvers live in ``examples/`` because they are documentation, not
library code; this is the only place the test suite reaches into them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))

from _solvers import (  # noqa: E402
    aerosol_optical_depth,
    mc_reflectance,
    rayleigh_optical_depth,
    spectral_doubling,
)


def test_monte_carlo_conserves_energy_without_absorption() -> None:
    """With ssa = 1 every photon must leave, so R + T is exactly 1."""
    for tau in (0.1, 1.0, 5.0):
        out = mc_reflectance(tau, 1.0, 0.0, 0.5, n_photons=400, seed=0)
        total = out["reflectance"] + out["transmittance"]
        assert total == pytest.approx(1.0, abs=1e-12)


def test_monte_carlo_reflects_nothing_without_scattering() -> None:
    """No scattering means no way back out of the top of the layer."""
    out = mc_reflectance(1.0, 0.0, 0.5, 0.5, n_photons=400, seed=0)
    assert out["reflectance"] == 0.0


def test_monte_carlo_transmits_everything_through_a_void() -> None:
    """A layer of no optical thickness cannot interact with a photon."""
    out = mc_reflectance(1e-9, 0.9, 0.5, 0.5, n_photons=400, seed=0)
    assert out["reflectance"] == 0.0
    assert out["transmittance"] == pytest.approx(1.0)


def test_monte_carlo_is_reproducible_from_its_seed() -> None:
    """A cached point has to be worth caching, so the seed must fix the run."""
    first = mc_reflectance(1.0, 0.9, 0.5, 0.5, n_photons=200, seed=7)
    second = mc_reflectance(1.0, 0.9, 0.5, 0.5, n_photons=200, seed=7)
    other = mc_reflectance(1.0, 0.9, 0.5, 0.5, n_photons=200, seed=8)
    assert first == second
    assert first != other


def test_monte_carlo_error_falls_as_the_square_root_of_the_photon_count() -> None:
    """Sixteen times the photons must halve the spread at least twice over."""
    spreads = []
    for n_photons in (250, 4000):
        values = [
            mc_reflectance(1.0, 0.9, 0.5, 0.5, n_photons=n_photons, seed=s)[
                "reflectance"
            ]
            for s in range(6)
        ]
        spreads.append(float(np.std(values)))
    assert spreads[1] < spreads[0] / 2.0


def test_doubling_conserves_energy_without_absorption() -> None:
    """The adding recursion preserves R + T = 1, which is why it is trusted."""
    tau = np.array([0.1, 1.0, 5.0, 20.0])
    out = spectral_doubling(tau, 1.0, 0.0)
    total = out["reflectance"] + out["transmittance"]
    np.testing.assert_allclose(total, 1.0, atol=1e-12)


def test_doubling_reflects_nothing_without_scattering() -> None:
    """A purely absorbing layer has no reflectance at any thickness."""
    out = spectral_doubling(np.array([0.1, 1.0, 5.0]), 0.0, 0.5)
    np.testing.assert_allclose(out["reflectance"], 0.0, atol=1e-15)


def test_doubling_is_elementwise_so_slicing_the_axis_is_free() -> None:
    """A vec callee must give the same answer on a slice as on the whole axis.

    This is what lets a wavelength axis be batched: if the recursion depth
    were derived from the data it would differ per batch, and a policy knob
    would silently move a value.
    """
    wl = np.linspace(0.4, 2.4, 64)
    tau = rayleigh_optical_depth(wl) + aerosol_optical_depth(wl, 0.3)
    whole = spectral_doubling(tau, 0.95, 0.6)["reflectance"]
    halves = np.concatenate(
        [
            spectral_doubling(tau[:32], 0.95, 0.6)["reflectance"],
            spectral_doubling(tau[32:], 0.95, 0.6)["reflectance"],
        ]
    )
    np.testing.assert_array_equal(whole, halves)


def test_doubling_has_converged_at_its_default_depth() -> None:
    """The default depth must be indistinguishable from a much deeper one."""
    shallow = spectral_doubling(np.array([2.0]), 0.9, 0.5, n_doublings=16)
    deep = spectral_doubling(np.array([2.0]), 0.9, 0.5, n_doublings=22)
    np.testing.assert_allclose(shallow["reflectance"], deep["reflectance"], atol=1e-5)


def test_the_two_solvers_agree_where_two_stream_is_nearly_exact() -> None:
    """Isotropic conservative scattering is the case two-stream handles well.

    They are two different approximations, so this is agreement rather than
    validation; for strongly forward-peaked scattering they part company by
    a few percent of reflectance, which is the two-stream closure error and
    not a defect in either solver.
    """
    for tau in (0.2, 0.5, 1.0):
        monte_carlo = mc_reflectance(tau, 1.0, 0.0, 0.5, n_photons=4000, seed=0)
        doubling = spectral_doubling(np.array([tau]), 1.0, 0.0)
        assert doubling["reflectance"][0] == pytest.approx(
            monte_carlo["reflectance"], abs=0.02
        )


def test_rayleigh_optical_depth_matches_the_published_value() -> None:
    """Bodhaine et al. (1999) give about 0.097 at 550 nm and sea level."""
    assert float(rayleigh_optical_depth(0.55)) == pytest.approx(0.0971, abs=5e-4)


def test_rayleigh_optical_depth_scales_with_pressure() -> None:
    """The fit is stated at 1013.25 hPa and is linear in pressure."""
    sea_level = float(rayleigh_optical_depth(0.55))
    altitude = float(rayleigh_optical_depth(0.55, pressure_hpa=506.625))
    assert altitude == pytest.approx(sea_level / 2.0)


def test_aerosol_optical_depth_is_anchored_at_550_nm() -> None:
    """The Angstrom law is written so that its reference value is exact."""
    assert float(aerosol_optical_depth(0.55, 0.2)) == pytest.approx(0.2)
    assert float(aerosol_optical_depth(1.10, 0.2)) < 0.2
