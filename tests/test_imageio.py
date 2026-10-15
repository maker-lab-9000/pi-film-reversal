import numpy as np
import pytest
from PIL import Image, ImageCms

from pifilm.imageio import ImageMeta, list_images, load_rgb, save_jpeg, srgb_profile


def test_save_and_load_roundtrip(tmp_path):
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    img[..., 0] = 200  # red-dominant: proves channel order survives
    out = save_jpeg(img, tmp_path / "nested" / "a.jpg", quality=100)
    assert out.exists()
    back, meta = load_rgb(out)
    assert back.shape == (10, 20, 3) and back.dtype == np.uint8
    assert back[..., 0].mean() > 150 and back[..., 2].mean() < 30
    assert isinstance(meta, ImageMeta) and meta.width == 20 and meta.height == 10


def test_saved_jpeg_carries_an_srgb_profile(tmp_path):
    out = save_jpeg(np.full((8, 8, 3), 120, dtype=np.uint8), tmp_path / "p.jpg")
    with Image.open(out) as im:
        assert im.info.get("icc_profile")
    _, meta = load_rgb(out)
    assert "sRGB" in meta.profile or "embedded" in meta.profile


def test_exif_orientation_is_applied(tmp_path):
    # A 4x2 image tagged orientation=6 (rotate 90 CW on display) must come back 2x4.
    base = Image.new("RGB", (4, 2), (10, 20, 30))
    exif = base.getexif()
    exif[274] = 6  # Orientation
    path = tmp_path / "rot.jpg"
    base.save(path, exif=exif)
    arr, meta = load_rgb(path)
    assert arr.shape[:2] == (4, 2)  # height, width swapped
    assert meta.oriented is True


def test_icc_profile_is_converted_not_ignored(tmp_path, wide_gamut_icc):
    """The same bytes under a wide-gamut profile must not decode to the same pixels."""
    pixels = np.full((8, 8, 3), (200, 60, 60), dtype=np.uint8)
    srgb_path = tmp_path / "srgb.jpg"
    wide_path = tmp_path / "wide.jpg"
    Image.fromarray(pixels).save(
        srgb_path, quality=100, icc_profile=ImageCms.ImageCmsProfile(srgb_profile()).tobytes()
    )
    Image.fromarray(pixels).save(wide_path, quality=100, icc_profile=wide_gamut_icc)

    srgb_arr, srgb_meta = load_rgb(srgb_path)
    wide_arr, wide_meta = load_rgb(wide_path)

    assert wide_meta.profile_error is None
    assert "Wide Gamut" in wide_meta.profile
    assert not np.array_equal(srgb_arr, wide_arr), "a non-sRGB profile must change the pixels"
    # Verified against the real Adobe RGB (1998) profile: (200, 60, 60) lands on
    # (231, 57, 56) in sRGB. Allow a little slack for JPEG and littlecms rounding.
    assert abs(int(wide_arr[0, 0, 0]) - 231) <= 3
    assert abs(int(wide_arr[0, 0, 2]) - 56) <= 3
    # An sRGB-tagged image must come back essentially unchanged.
    assert np.abs(srgb_arr.astype(int) - pixels.astype(int)).max() <= 2


def test_malformed_profile_falls_back_and_reports(tmp_path):
    path = tmp_path / "bad.jpg"
    Image.fromarray(np.full((8, 8, 3), 90, dtype=np.uint8)).save(path, icc_profile=b"not-a-profile")
    arr, meta = load_rgb(path)
    assert arr.shape == (8, 8, 3)
    assert meta.profile == "invalid"
    assert meta.profile_error


def test_colour_manage_can_be_disabled(tmp_path, wide_gamut_icc):
    """The toggle must be provable, so the fixture needs a profile that would change pixels.

    An image with no embedded profile passes this test whether or not the
    toggle does anything, which is no test at all.
    """
    path = tmp_path / "wide.jpg"
    pixels = np.full((8, 8, 3), (200, 60, 60), dtype=np.uint8)
    Image.fromarray(pixels).save(path, quality=100, icc_profile=wide_gamut_icc)

    managed, _ = load_rgb(path)
    unmanaged, meta = load_rgb(path, colour_manage=False)

    assert not np.array_equal(managed, unmanaged), "the toggle did not suppress conversion"
    # Measured: managed (231, 57, 56); unmanaged stays at the stored values.
    assert np.abs(unmanaged.astype(int) - pixels.astype(int)).max() <= 2
    assert "assumed" in meta.profile and "off" in meta.profile


def test_a_lab_source_is_converted_not_mislabelled(tmp_path):
    """Regression test for a bug an earlier draft of this module contained.

    Converting the image to RGB before ``profileToProfile`` makes a LAB or
    CMYK transform unbuildable, so littlecms raises and the fallback brands a
    perfectly valid profile "invalid" — silently skipping colour management
    on exactly the archival scans that need it. This pins that it does not
    happen.
    """
    lab_profile = ImageCms.createProfile("LAB")
    lab = Image.new("LAB", (8, 8))
    lab.putdata([(180, 160, 140)] * 64)
    path = tmp_path / "lab.tif"
    lab.save(path, icc_profile=ImageCms.ImageCmsProfile(lab_profile).tobytes())

    arr, meta = load_rgb(path)
    assert arr.shape == (8, 8, 3) and arr.dtype == np.uint8
    assert meta.profile != "invalid", "a valid LAB profile was mislabelled"
    assert meta.profile_error is None


def test_load_converts_modes(tmp_path):
    Image.new("L", (8, 8), 77).save(tmp_path / "grey.png")
    Image.new("RGBA", (8, 8), (1, 2, 3, 255)).save(tmp_path / "rgba.png")
    assert load_rgb(tmp_path / "grey.png")[0].shape == (8, 8, 3)
    assert load_rgb(tmp_path / "rgba.png")[0].shape == (8, 8, 3)


def test_saved_jpeg_keeps_full_chroma_resolution(tmp_path):
    """4:2:0 would destroy chroma detail in the project's only output writer."""
    columns = np.zeros((64, 64, 3), dtype=np.uint8)
    columns[:, ::2] = (220, 30, 30)
    columns[:, 1::2] = (30, 200, 60)
    path = save_jpeg(columns, tmp_path / "chroma.jpg")

    with Image.open(path) as im:
        # layer[0] carries the luma sampling factors; (1, 1, 1, 0) means 4:4:4.
        assert im.layer[0][1:3] == (1, 1), f"expected 4:4:4, got layers {im.layer}"

    back, _ = load_rgb(path)
    # Measured: 4:2:0 gives a max error of about 118 here; 4:4:4 gives 2.
    assert np.abs(back.astype(int) - columns.astype(int)).max() <= 8


def test_loaded_arrays_are_writeable(tmp_path):
    """Callers draw on preview frames; a read-only return value breaks them."""
    save_jpeg(np.full((8, 8, 3), 120, dtype=np.uint8), tmp_path / "w.jpg")
    arr, _ = load_rgb(tmp_path / "w.jpg")
    assert arr.flags.writeable
    arr[0, 0, 0] = 5  # must not raise


def test_list_images_filters_and_sorts(tmp_path):
    for name in ["b.JPG", "a.jpeg", "c.png", "notes.txt", "d.tif"]:
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in list_images(tmp_path)] == ["a.jpeg", "b.JPG", "c.png", "d.tif"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rgb(tmp_path / "nope.jpg")


@pytest.mark.parametrize("fourth", [0, 255, "noise"])
def test_a_four_byte_image_saves_as_the_same_jpeg(tmp_path, fourth):
    """The GPU hands its result back as four bytes a pixel; the fourth is not
    part of the picture, whatever it holds."""
    rng = np.random.default_rng(11)
    rgb = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
    rgbx = np.empty((37, 53, 4), np.uint8)
    rgbx[..., :3] = rgb
    rgbx[..., 3] = rng.integers(0, 256, (37, 53)) if fourth == "noise" else fourth
    three = save_jpeg(rgb, tmp_path / "three.jpg").read_bytes()
    four = save_jpeg(rgbx, tmp_path / "four.jpg").read_bytes()
    assert four == three
    plain = save_jpeg(rgb, tmp_path / "a.jpg", quality=80, embed_srgb=False, subsampling=2)
    packed = save_jpeg(rgbx, tmp_path / "b.jpg", quality=80, embed_srgb=False, subsampling=2)
    assert packed.read_bytes() == plain.read_bytes()


def test_a_four_byte_view_that_is_not_contiguous_still_saves_the_same(tmp_path):
    rng = np.random.default_rng(12)
    rgbx = rng.integers(0, 256, (40, 60, 4), dtype=np.uint8)
    view = rgbx[::2, ::3]
    want = save_jpeg(np.ascontiguousarray(view[..., :3]), tmp_path / "w.jpg").read_bytes()
    assert save_jpeg(view, tmp_path / "g.jpg").read_bytes() == want


@pytest.mark.parametrize("array", [
    np.zeros((8, 8, 2), np.uint8),
    np.zeros((8, 8, 5), np.uint8),
    np.zeros((8, 8, 4), np.float32),
    np.zeros((8, 8, 4), np.uint16),
])
def test_save_jpeg_refuses_what_is_not_a_picture(tmp_path, array):
    with pytest.raises((ValueError, TypeError)):
        save_jpeg(array, tmp_path / "bad.jpg")
    assert not (tmp_path / "bad.jpg").exists()
