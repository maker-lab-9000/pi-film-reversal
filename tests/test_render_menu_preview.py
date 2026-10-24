import numpy as np
import pytest
from PIL import Image

from pifilm.display.menu import ROW_HEIGHT, ROW_X0, ROW_X1, ROW_Y0
from scripts import render_menu_preview as script

STEMS = ["live", "live-menu-pressed", "menu-1-first", "menu-2-camera", "menu-3-pipeline",
         "menu-4-network", "menu-5-image", "menu-6-stabilise"]


@pytest.mark.parametrize("scale", [1, 2])
def test_writes_every_screen_at_the_panel_size_and_a_contact_sheet(tmp_path, scale):
    assert script.main([str(tmp_path), "--scale", str(scale)]) == 0
    for stem in STEMS:
        with Image.open(tmp_path / f"{stem}.png") as image:
            assert image.size == (320 * scale, 240 * scale)
    with Image.open(tmp_path / "contact-sheet.png") as sheet:
        assert sheet.width > 4 * 320 * scale and sheet.height > 2 * 240 * scale


def test_the_pages_come_from_the_viewfinder_tree():
    rendered = script.render_pages(1, script.sample_frame())
    assert [stem for stem, *_ in rendered] == STEMS
    images = {stem: np.asarray(image) for stem, _, image in rendered}
    # The MENU key lights when pressed; every page differs from the others.
    assert not np.array_equal(images["live"], images["live-menu-pressed"])
    pages = [images[stem] for stem in STEMS[2:]]
    assert all(not np.array_equal(a, b) for i, a in enumerate(pages) for b in pages[i + 1:])
    # The first page has no title: its first row is drawn on the first line; a
    # category page's first line is its amber title (red well above blue).
    first_line = (slice(ROW_Y0, ROW_Y0 + ROW_HEIGHT), slice(ROW_X0, ROW_X1))
    camera = images["menu-2-camera"][first_line].reshape(-1, 3).astype(int)
    assert ((camera[:, 0] - camera[:, 2]) > 100).any()
    first = images["menu-1-first"][first_line].reshape(-1, 3).astype(int)
    assert not ((first[:, 0] - first[:, 2]) > 100).any()


def test_a_frame_that_cannot_be_read_is_an_error(tmp_path, capsys):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not a jpeg")
    assert script.main([str(tmp_path / "out"), "--frame", str(bad)]) == 2
    assert "cannot read" in capsys.readouterr().err
