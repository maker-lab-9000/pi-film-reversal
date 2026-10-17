import numpy as np
import pytest

from pifilm.color import srgb_to_oklab
from pifilm.imageio import save_jpeg
from pifilm.normalize import NormalizeParams
from pifilm.train.dataset import (
    CorpusTooSmall,
    PixelPool,
    SampleConfig,
    build_corpus,
    build_pool,
    corpus_sha1,
    crop_and_resize,
    prepare_image,
    sample_pixels,
    split_paths,
)


def _write_images(dir_path, n, seed=0):
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        p = dir_path / f"{i:03d}.jpg"
        save_jpeg(rng.integers(30, 220, (40, 60, 3), dtype=np.uint8), p)
        paths.append(p)
    return paths


def test_crop_and_resize_geometry():
    assert crop_and_resize(np.zeros((500, 1000, 3), np.uint8), 0.06, 512).shape == (256, 512, 3)
    assert crop_and_resize(np.zeros((1000, 500, 3), np.uint8), 0.0, 100).shape == (100, 50, 3)


def test_crop_removes_border():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[10:90, 10:90] = 200
    assert crop_and_resize(img, 0.1, 80).min() >= 190


def test_prepare_image_normalises_and_returns_gains():
    # A mild cast, dark enough that neither gain clamps: median linear
    # luminance lands near the 0.18 target, so the exposure gain is 0.99.
    # (190, 170, 150) looks similar but sits at luminance 0.42, which drives
    # the raw exposure gain to 0.43 and clamps it against the 0.5 floor.
    img = np.full((100, 100, 3), (130, 116, 102), dtype=np.uint8)
    out, gains = prepare_image(img, NormalizeParams(), SampleConfig(crop_frac=0.0, max_side=50))
    assert out.dtype == np.float32 and out.shape == (50, 50, 3)
    assert np.allclose(out[..., 0], out[..., 1], atol=1 / 255)
    assert gains.clamped == {"wb": False, "exposure": False}


def test_sample_pixels_respects_bounds_and_count():
    img = np.zeros((10, 20, 3), dtype=np.float32)
    img[:, 10:] = 0.5
    rng = np.random.default_rng(0)
    px = sample_pixels(img, 1000, 0.02, 0.98, rng)
    assert px.shape == (100, 3) and np.allclose(px, 0.5)
    assert sample_pixels(img, 30, 0.02, 0.98, rng).shape == (30, 3)


@pytest.mark.parametrize(
    "n, frac, expected_val", [(10, 0.2, 2), (5, 0.2, 1), (3, 0.5, 1), (1, 0.2, 0)]
)
def test_split_sizes(n, frac, expected_val):
    paths = [f"{i}.jpg" for i in range(n)]
    train, val = split_paths(paths, frac, seed=0)
    assert len(val) == expected_val
    assert len(train) == n - expected_val
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(paths)


def test_split_is_seeded_and_stable():
    paths = [f"{i}.jpg" for i in range(20)]
    assert split_paths(paths, 0.2, seed=3) == split_paths(paths, 0.2, seed=3)
    assert split_paths(paths, 0.2, seed=3) != split_paths(paths, 0.2, seed=4)


def test_build_pool_collects_diagnostics(tmp_path):
    paths = _write_images(tmp_path / "src", 3)
    cfg = SampleConfig(crop_frac=0.0, max_side=60, pixels_per_image=500, max_pixels=800)
    pool = build_pool(paths, NormalizeParams(), cfg)
    assert isinstance(pool, PixelPool)
    assert pool.n_images == 3
    assert pool.srgb.shape == (800, 3) and pool.srgb.dtype == np.float32
    assert np.allclose(pool.lab, srgb_to_oklab(pool.srgb), atol=1e-6)
    assert len(pool.wb_gains) == 3 and len(pool.exposure_gains) == 3
    assert 0.0 <= pool.clamp_rate <= 1.0
    assert pool.profiles  # profile name -> count


def test_no_validation_pixel_appears_in_training(tmp_path):
    """The split must happen before sampling, or held-out metrics are meaningless."""
    paths = _write_images(tmp_path / "src", 10, seed=5)
    cfg = SampleConfig(crop_frac=0.0, max_side=60, pixels_per_image=2000, val_fraction=0.3)
    split = build_corpus(paths, NormalizeParams(), cfg, minimum=1, label="source")
    assert len(split.val_paths) == 3 and len(split.train_paths) == 7
    assert set(split.train_paths).isdisjoint(split.val_paths)

    # Rebuild each side independently: the pools must match what build_corpus produced,
    # which is only possible if neither drew pixels from the other's images.
    expected_val = build_pool(split.val_paths, NormalizeParams(), cfg)
    assert np.array_equal(split.val_pool.srgb, expected_val.srgb)

    # And the mirror, which matters more. A validation pool contaminated by
    # training images inflates the score; a TRAINING pool contaminated by
    # held-out images corrupts the fit itself and then reports on data it was
    # fitted to. Verified that the assertion above stays green through exactly
    # that leak, so it cannot stand in for this one.
    expected_train = build_pool(split.train_paths, NormalizeParams(), cfg)
    assert np.array_equal(split.train_pool.srgb, expected_train.srgb)


def test_corpus_too_small_names_the_escape_hatch(tmp_path):
    paths = _write_images(tmp_path / "src", 4)
    cfg = SampleConfig(crop_frac=0.0, max_side=60, pixels_per_image=100)
    with pytest.raises(CorpusTooSmall, match="--allow-small"):
        build_corpus(paths, NormalizeParams(), cfg, minimum=30, label="source")
    split = build_corpus(
        paths, NormalizeParams(), cfg, minimum=30, label="source", allow_small=True
    )
    assert split.train_pool.n_images >= 1


def test_corpus_sha1_tracks_content_not_names(tmp_path):
    paths = _write_images(tmp_path / "a", 2)
    first = corpus_sha1(paths)
    assert first == corpus_sha1(paths)
    assert len(first) == 40
    save_jpeg(np.full((40, 60, 3), 7, dtype=np.uint8), paths[0])  # same name, new content
    assert corpus_sha1(paths) != first


def test_invalid_sample_config_names_the_field():
    with pytest.raises(ValueError, match="val_fraction"):
        SampleConfig(val_fraction=1.5)
    with pytest.raises(ValueError, match="max_side"):
        SampleConfig(max_side=0)
    with pytest.raises(ValueError, match="crop_frac"):
        SampleConfig(crop_frac=0.5)


# --- per-image weights (weights.json beside the images) ---------------------

def test_a_weighted_image_gives_that_many_times_the_pixels(tmp_path):
    import json

    from pifilm.train.dataset import load_weights

    paths = _write_images(tmp_path / "refs", 2)
    (tmp_path / "refs" / "weights.json").write_text(json.dumps({paths[0].name: 3}))
    weights = load_weights(tmp_path / "refs")
    cfg = SampleConfig(crop_frac=0.0, pixels_per_image=200, l_min=0.0, l_max=1.0)
    plain = build_pool(paths[1:], NormalizeParams(), cfg)
    heavy = build_pool(paths[:1], NormalizeParams(), cfg, weights=weights)
    assert len(plain.srgb) == 200
    assert len(heavy.srgb) == 600


def test_no_weights_file_means_every_image_counts_once(tmp_path):
    from pifilm.train.dataset import load_weights

    _write_images(tmp_path / "refs", 2)
    assert load_weights(tmp_path / "refs") == {}


@pytest.mark.parametrize(
    "content, message",
    [
        ('{"missing.jpg": 2}', "not in"),
        ('{"000.jpg": 0}', "positive"),
        ('{"000.jpg": -1}', "positive"),
        ('{"000.jpg": "2"}', "number"),
        ('{"000.jpg": 11}', "at most"),
        ('["000.jpg"]', "object"),
    ],
)
def test_a_bad_weights_file_is_refused(tmp_path, content, message):
    from pifilm.train.dataset import load_weights

    _write_images(tmp_path / "refs", 2)
    (tmp_path / "refs" / "weights.json").write_text(content)
    with pytest.raises(ValueError, match=message):
        load_weights(tmp_path / "refs")


def test_weights_change_the_corpus_identity_and_are_carried(tmp_path):
    cfg = SampleConfig(crop_frac=0.0, pixels_per_image=50, l_min=0.0, l_max=1.0)
    _write_images(tmp_path / "refs", 6)
    before = build_corpus(tmp_path / "refs", NormalizeParams(), cfg, 1, "target")
    (tmp_path / "refs" / "weights.json").write_text('{"000.jpg": 2.5}')
    after = build_corpus(tmp_path / "refs", NormalizeParams(), cfg, 1, "target")
    assert before.weights == {}
    assert after.weights == {"000.jpg": 2.5}
    assert before.corpus_sha1 != after.corpus_sha1
