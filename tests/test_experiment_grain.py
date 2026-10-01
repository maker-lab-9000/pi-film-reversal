import numpy as np

from pifilm._cv2 import require_cv2
from pifilm.experiments.grain import (
    amplitude_curve,
    fit_clump_mix,
    fit_clump_sigma,
    flat_patches,
    radial_psd,
    synth_grain,
)

cv2 = require_cv2()


def _grain_field(shape, sigma_px, sd, seed=0):
    noise = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigma_px)
    return noise / noise.std() * sd


def test_flat_patches_skip_structure_and_keep_flat_grain():
    y = np.full((512, 512), 0.5, np.float32) + _grain_field((512, 512), 1.5, 0.03)
    y[:, :256] += np.linspace(0, 0.4, 512, dtype=np.float32)[:, None]  # a strong ramp
    y[:256, 256:] += (np.indices((256, 256)).sum(0) % 64 < 32) * 0.2  # stripes: scene detail
    patches = flat_patches(y, size=128)
    assert patches, "the flat quadrant must yield patches"
    for p in patches:
        assert p.y0 >= 256 and p.x0 >= 256
        assert abs(p.mean - 0.5) < 0.02


def test_residual_sd_recovers_the_grain_amplitude():
    y = 0.4 + _grain_field((512, 512), 1.5, 0.025)
    sds = [p.sd for p in flat_patches(y, size=128)]
    assert abs(np.median(sds) - 0.025) < 0.004


def test_clump_sigma_is_recovered_from_the_spectrum():
    for sigma in (1.0, 2.0, 3.0):
        y = 0.5 + _grain_field((1024, 1024), sigma, 0.03, seed=int(sigma * 10))
        f, p = radial_psd([pt.residual for pt in flat_patches(y, size=256)])
        assert abs(fit_clump_sigma(f, p) - sigma) / sigma < 0.15


def test_amplitude_curve_bins_by_patch_mean():
    rng = np.random.default_rng(1)
    means = rng.uniform(0.1, 0.9, 400)
    sds = 0.01 + 0.04 * means  # grain grows with brightness here
    centres, curve = amplitude_curve(means, sds, bins=np.linspace(0, 1, 11))
    assert np.all(np.diff(curve[~np.isnan(curve)]) > 0)
    assert abs(curve[np.searchsorted(centres, 0.55)] - (0.01 + 0.04 * 0.55)) < 0.003


def test_synth_grain_has_the_requested_sigma_and_amplitude():
    lum = np.full((512, 512), 0.5, np.float32)
    g = synth_grain(lum, sigma_px=2.0, curve_x=np.array([0.0, 1.0]),
                    curve_sd=np.array([0.03, 0.03]), rng=np.random.default_rng(3))
    assert abs(g.std() - 0.03) < 0.002
    f, p = radial_psd([pt.residual for pt in flat_patches(0.5 + g, size=256)])
    assert abs(fit_clump_sigma(f, p) - 2.0) / 2.0 < 0.15


def test_two_scale_mixture_is_recovered():
    """HP5's spectrum falls off like fine plus coarse clumps, not one Gaussian."""
    rng = np.random.default_rng(7)
    shape = (1024, 1024)
    fine = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), 0.6)
    coarse = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), 2.5)
    g = np.sqrt(0.6) * fine / fine.std() + np.sqrt(0.4) * coarse / coarse.std()
    f, p = radial_psd([pt.residual for pt in flat_patches(0.5 + 0.03 * g, size=256)])
    s1, s2, w1 = fit_clump_mix(f, p)
    assert abs(s1 - 0.6) < 0.2 and abs(s2 - 2.5) / 2.5 < 0.25
    assert abs(w1 - 0.6) < 0.15


def test_synth_grain_mixture_matches_requested_sd():
    lum = np.full((512, 512), 0.5, np.float32)
    g = synth_grain(lum, sigma_px=(0.6, 2.5), weights=(0.6, 0.4), curve_x=np.array([0.0, 1.0]),
                    curve_sd=np.array([0.04, 0.04]), rng=np.random.default_rng(1))
    assert abs(g.std() - 0.04) < 0.003
