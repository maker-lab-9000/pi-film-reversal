import json

import numpy as np
import pytest

from pifilm.artifacts import Artifacts
from pifilm.capture.batch import (
    load_capture_log,
    main,
    output_path,
    process_dir,
    select_inputs,
)
from pifilm.imageio import load_rgb, save_jpeg
from pifilm.pipeline import Pipeline


def _img(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (24, 32, 3), dtype=np.uint8)


def _capture_dir(tmp_path):
    """Looks like a real capture folder: originals plus already-graded siblings."""
    d = tmp_path / "shots"
    d.mkdir()
    for stem in ("120001", "120002"):
        save_jpeg(_img(1), d / f"{stem}_original.jpg")
        save_jpeg(_img(2), d / f"{stem}_graded.jpg")
    (d / "captures.jsonl").write_text("{}\n")
    return d


def test_select_inputs_prefers_originals_and_always_skips_graded(tmp_path):
    d = _capture_dir(tmp_path)
    chosen = [p.name for p in select_inputs(sorted(d.glob("*.jpg")))]
    assert chosen == ["120001_original.jpg", "120002_original.jpg"]


def test_select_inputs_all_still_skips_graded(tmp_path):
    d = _capture_dir(tmp_path)
    save_jpeg(_img(3), d / "loose.jpg")
    chosen = [p.name for p in select_inputs(sorted(d.glob("*.jpg")), all_files=True)]
    assert "loose.jpg" in chosen
    assert not any("_graded" in n for n in chosen)


def test_plain_folder_processes_everything(tmp_path):
    d = tmp_path / "plain"
    d.mkdir()
    save_jpeg(_img(1), d / "a.jpg")
    save_jpeg(_img(2), d / "b.jpg")
    assert len(select_inputs(sorted(d.glob("*.jpg")))) == 2


def test_capture_dir_is_not_double_graded(tmp_path):
    d = _capture_dir(tmp_path)
    result = process_dir(d, tmp_path / "out")
    assert [p.name for p in result.written] == [
        "120001_original_graded.jpg",
        "120002_original_graded.jpg",
    ]
    assert result.skipped_graded == 2


def test_same_stem_different_extensions_do_not_collide(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    save_jpeg(_img(1), d / "a.jpg")
    from PIL import Image

    Image.fromarray(_img(2)).save(d / "a.png")
    written = {p.name for p in process_dir(d, tmp_path / "out").written}
    assert written == {"a_jpg_graded.jpg", "a_png_graded.jpg"}


def test_output_path_without_disambiguation():
    from pathlib import Path

    assert output_path(Path("x/a.jpg"), Path("out"), False).name == "a_graded.jpg"
    assert output_path(Path("x/a.jpg"), Path("out"), True).name == "a_jpg_graded.jpg"


def test_existing_outputs_are_skipped_then_overwritten(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    save_jpeg(_img(1), d / "a.jpg")
    out = tmp_path / "out"
    first = process_dir(d, out)
    assert len(first.written) == 1
    second = process_dir(d, out)
    assert second.written == [] and second.skipped_existing == 1
    third = process_dir(d, out, overwrite=True)
    assert len(third.written) == 1


def test_nested_or_identical_output_is_refused(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    save_jpeg(_img(1), d / "a.jpg")
    with pytest.raises(ValueError, match="inside"):
        process_dir(d, d)
    with pytest.raises(ValueError, match="inside"):
        process_dir(d, d / "sub")


def test_main_uses_the_packaged_default_from_any_cwd(tmp_path, monkeypatch, capsys):
    d = tmp_path / "in"
    d.mkdir()
    save_jpeg(_img(1), d / "a.jpg")
    monkeypatch.chdir(tmp_path)
    assert main([str(d), str(tmp_path / "out")]) == 0
    assert "1 image" in capsys.readouterr().out


def test_main_reports_empty_input(tmp_path, capsys):
    (tmp_path / "in").mkdir()
    assert main([str(tmp_path / "in"), str(tmp_path / "out")]) == 1
    assert "no images" in capsys.readouterr().err.lower()


def test_main_reports_bad_artifacts(tmp_path, capsys):
    d = tmp_path / "in"
    d.mkdir()
    save_jpeg(_img(1), d / "a.jpg")
    assert main([str(d), str(tmp_path / "out"), "--artifacts", str(tmp_path / "none")]) == 2
    assert "params.json" in capsys.readouterr().err


# --- captures.jsonl: logged EV and grain seed ------------------------------

def _logged_dir(tmp_path, records, extra_lines=()):
    d = tmp_path / "day"
    d.mkdir()
    for r in records:
        save_jpeg(_img(len(r["original"])), d / r["original"])
    lines = [json.dumps(r) for r in records] + list(extra_lines)
    (d / "captures.jsonl").write_text("\n".join(lines) + "\n")
    return d


def _expected(src, *, ev, seed):
    rgb, _ = load_rgb(src)
    graded, _ = Pipeline(Artifacts.resolve(None)).process(
        rgb, rng=np.random.default_rng(seed), ev=ev)
    return graded


def test_logged_ev_and_grain_seed_reproduce_the_pipeline(tmp_path):
    d = _logged_dir(tmp_path, [
        {"original": "120001_original.jpg", "ev_comp": -0.67, "grain_seed": 1234},
        {"original": "120002_original.jpg", "ev_comp": 0.0, "grain_seed": 99},
    ])
    result = process_dir(d, tmp_path / "out")
    assert result.from_log == 2 and result.ev_applied == 1
    for name, ev, seed in (("120001", -0.67, 1234), ("120002", 0.0, 99)):
        out, _ = load_rgb(tmp_path / "out" / f"{name}_original_graded.jpg")
        want = _expected(d / f"{name}_original.jpg", ev=ev, seed=seed)
        save_jpeg(want, tmp_path / "want.jpg")
        assert np.array_equal(out, load_rgb(tmp_path / "want.jpg")[0])


def test_logged_ev_darkens_compared_with_ignoring_the_log(tmp_path):
    d = _logged_dir(tmp_path, [
        {"original": "120001_original.jpg", "ev_comp": -2.0, "grain_seed": 1}])
    process_dir(d, tmp_path / "a", grain=False)
    ignored = process_dir(d, tmp_path / "b", grain=False, use_log=False)
    assert ignored.from_log == 0
    a = load_rgb(tmp_path / "a" / "120001_original_graded.jpg")[0].mean()
    b = load_rgb(tmp_path / "b" / "120001_original_graded.jpg")[0].mean()
    assert a < b - 10


def test_no_log_keeps_ev_zero(tmp_path):
    d = tmp_path / "plain"
    d.mkdir()
    save_jpeg(_img(5), d / "120001_original.jpg")
    result = process_dir(d, tmp_path / "out", grain=False)
    assert result.from_log == 0
    out, _ = load_rgb(tmp_path / "out" / "120001_original_graded.jpg")
    rgb, _ = load_rgb(d / "120001_original.jpg")
    want, _ = Pipeline(Artifacts.resolve(None)).process(rgb, grain=False)
    save_jpeg(want, tmp_path / "want.jpg")
    assert np.array_equal(out, load_rgb(tmp_path / "want.jpg")[0])


def test_log_skips_malformed_lines_and_records_without_a_file(tmp_path):
    d = _logged_dir(
        tmp_path,
        [{"original": "120001_original.jpg", "ev_comp": -1.0, "grain_seed": 7}],
        extra_lines=["{not json", "[1, 2]", json.dumps({"original": "gone_original.jpg"})],
    )
    log, bad = load_capture_log(d)
    assert bad == 2
    assert set(log) == {"120001_original.jpg", "gone_original.jpg"}
    assert process_dir(d, tmp_path / "out").from_log == 1


def test_double_exposure_frames_use_their_own_ev_and_fresh_grain(tmp_path):
    d = _logged_dir(tmp_path, [
        {"original": "120001_original.jpg", "ev_comp": -1.0,
         "double": {"index": 1, "of": 2}},
        {"original": "120002_original.jpg", "ev_comp": 0.0, "grain_seed": 42,
         "double": {"index": 2, "of": 2, "ev_comp": [-1.0, -0.5]}},
    ])
    log, _ = load_capture_log(d)
    assert (log["120001_original.jpg"].ev, log["120001_original.jpg"].seed) == (-1.0, None)
    # Exposure 2's seed is the composite's grain, not this frame's.
    assert (log["120002_original.jpg"].ev, log["120002_original.jpg"].seed) == (-0.5, None)
    assert process_dir(d, tmp_path / "out").doubles == 2


def test_main_reports_logged_frames_and_accepts_ignore_log(tmp_path, capsys):
    d = _logged_dir(tmp_path, [
        {"original": "120001_original.jpg", "ev_comp": -0.33, "grain_seed": 3}])
    assert main([str(d), str(tmp_path / "out")]) == 0
    assert "1 from captures.jsonl (1 with EV)" in capsys.readouterr().out
    assert main([str(d), str(tmp_path / "out2"), "--ignore-log"]) == 0
    assert "captures.jsonl" not in capsys.readouterr().out
