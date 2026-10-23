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


# --- records the Pi graded on its GPU ---------------------------------------

GPU_NOTE = "on its GPU"


def test_a_gpu_graded_record_is_noted_once_and_changes_no_file(tmp_path, capsys):
    """The regrade is the CPU's. For a record the Pi graded on its GPU that is
    close to the Pi's file, not the same bytes, and the tool has to say so."""
    records = [
        {"original": "120001_original.jpg", "ev_comp": 0.0, "grain_seed": 5},
        {"original": "120002_original.jpg", "ev_comp": 0.0, "grain_seed": 6},
    ]
    for name in ("cpu", "gpu", "old"):
        (tmp_path / name).mkdir()
    cpu = _logged_dir(tmp_path / "cpu", [{**r, "pipeline_backend": "cpu"} for r in records])
    gpu = _logged_dir(tmp_path / "gpu", [{**r, "pipeline_backend": "gpu-1"} for r in records])
    old = _logged_dir(tmp_path / "old", records)

    assert main([str(gpu), str(tmp_path / "out-gpu")]) == 0
    out = capsys.readouterr().out
    assert out.count(GPU_NOTE) == 1 and "not byte for byte" in out
    assert "2 of these" in out
    for folder, dest in ((cpu, "out-cpu"), (old, "out-old")):
        assert main([str(folder), str(tmp_path / dest)]) == 0
        assert GPU_NOTE not in capsys.readouterr().out
    for name in ("120001_original_graded.jpg", "120002_original_graded.jpg"):
        want = (tmp_path / "out-old" / name).read_bytes()
        assert (tmp_path / "out-gpu" / name).read_bytes() == want
        assert (tmp_path / "out-cpu" / name).read_bytes() == want


def test_the_gpu_note_counts_only_what_was_regraded_from_the_log(tmp_path, capsys):
    d = _logged_dir(tmp_path, [
        {"original": "120001_original.jpg", "grain_seed": 5, "pipeline_backend": "gpu-1"},
        {"original": "120002_original.jpg", "grain_seed": 6, "pipeline_backend": "cpu"},
        {"original": "gone_original.jpg", "grain_seed": 7, "pipeline_backend": "gpu-1"},
    ])
    (d / "gone_original.jpg").unlink()
    result = process_dir(d, tmp_path / "out")
    assert result.gpu_graded == 1 and result.from_log == 2
    assert process_dir(d, tmp_path / "ignored", use_log=False).gpu_graded == 0
    assert main([str(d), str(tmp_path / "out2"), "--ignore-log"]) == 0
    assert GPU_NOTE not in capsys.readouterr().out


def test_a_backend_that_is_not_text_is_not_a_gpu_record(tmp_path):
    d = _logged_dir(tmp_path, [
        {"original": "120001_original.jpg", "grain_seed": 5, "pipeline_backend": 1},
        {"original": "120002_original.jpg", "grain_seed": 6, "pipeline_backend": None},
    ])
    log, bad = load_capture_log(d)
    assert bad == 0 and not any(entry.gpu for entry in log.values())


# --- halation switched off in the menu ----------------------------------------


def _halation_look_dir(tmp_path):
    from pifilm.artifacts import write_artifact
    from pifilm.halation import HalationParams

    art = Artifacts.default()
    return write_artifact(tmp_path / "look", art.lut, art.normalize, art.grain,
                          halation=HalationParams(enabled=True))


def _with_a_lamp(seed=0):
    """A frame with a bright patch, so a glow shows."""
    rgb = np.full((48, 64, 3), 40, np.uint8)
    rgb[:24] = 70
    rgb[19:29, 27:37] = 255
    return rgb


def test_a_record_with_halation_off_is_regraded_without_it(tmp_path, capsys):
    look = _halation_look_dir(tmp_path)
    d = tmp_path / "day"
    d.mkdir()
    for name in ("120001_original.jpg", "120002_original.jpg"):
        save_jpeg(_with_a_lamp(), d / name)
    (d / "captures.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"original": "120001_original.jpg", "ev_comp": 0.0, "grain_seed": 5,
         "halation": "off"},
        {"original": "120002_original.jpg", "ev_comp": 0.0, "grain_seed": 5,
         "halation_sha1": "x"},
    ]) + "\n")
    result = process_dir(d, tmp_path / "out", look)
    assert result.halation_off == 1
    pipe = Pipeline(Artifacts.resolve(look))
    rgb, _ = load_rgb(d / "120001_original.jpg")
    for name, halation in (("120001", False), ("120002", True)):
        want, _ = pipe.process(rgb, rng=np.random.default_rng(5), halation=halation)
        save_jpeg(want, tmp_path / "want.jpg")
        got = load_rgb(tmp_path / "out" / f"{name}_original_graded.jpg")[0]
        assert np.array_equal(got, load_rgb(tmp_path / "want.jpg")[0])
    # The two differ: the glow is the look's and was left out of one.
    a = load_rgb(tmp_path / "out" / "120001_original_graded.jpg")[0].astype(int)
    b = load_rgb(tmp_path / "out" / "120002_original_graded.jpg")[0].astype(int)
    assert np.abs(a - b).max() > 5
    # --ignore-log keeps the look's halation for both.
    ignored = process_dir(d, tmp_path / "ignored", look, use_log=False)
    assert ignored.halation_off == 0
    assert main([str(d), str(tmp_path / "out2"), "--artifacts", str(look)]) == 0
    assert "1 without halation, as shot" in capsys.readouterr().out


def test_only_an_off_value_turns_halation_off(tmp_path):
    d = _logged_dir(tmp_path, [
        {"original": "120001_original.jpg", "grain_seed": 5, "halation": "off"},
        {"original": "120002_original.jpg", "grain_seed": 6},
        {"original": "120003_original.jpg", "grain_seed": 7, "halation": False},
        {"original": "120004_original.jpg", "double": {"index": 2, "ev_comp": [0, 0]},
         "halation": "off"},
    ])
    log, bad = load_capture_log(d)
    assert bad == 0
    assert [log[f"12000{i}_original.jpg"].halation for i in range(1, 5)] == [
        False, True, True, False]


# --- multi-frame photographs (a record's ``stack``) ---------------------------


def _burst_scene(seed=0, h=96, w=128):
    """A smooth scene, so the frames align, plus each frame's own noise."""
    from pifilm._cv2 import require_cv2

    cv2 = require_cv2()
    rng = np.random.default_rng(100)
    base = cv2.GaussianBlur(rng.random((h, w, 3), dtype=np.float32), (0, 0), 4)
    base = 128.0 + 60.0 * (base - base.mean()) / base.std()
    noise = np.random.default_rng(seed).normal(0.0, 12.0, (h, w, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _stack_dir(tmp_path, n=3, missing=()):
    d = tmp_path / "day"
    d.mkdir()
    names = ["120001_original.jpg"] + [f"120001_frame{i}.jpg" for i in range(2, n + 1)]
    for i, name in enumerate(names):
        if name not in missing:
            save_jpeg(_burst_scene(i), d / name)
    save_jpeg(_img(9), d / "120001_graded.jpg")
    record = {"original": "120001_original.jpg", "ev_comp": -0.33, "grain_seed": 77,
              "shutter_delay_s": 0.0,
              "stack": {"frames": names, "reference": names[0], "exposure_us": 5000,
                        "gain": 2.0, "split": True, "method": "ecc-euclidean"}}
    (d / "captures.jsonl").write_text(json.dumps(record) + "\n")
    return d, names


def _graded(rgb, *, ev, seed, tmp_path):
    want, _ = Pipeline(Artifacts.resolve(None)).process(
        rgb, rng=np.random.default_rng(seed), ev=ev)
    return load_rgb(save_jpeg(want, tmp_path / "want.jpg"))[0]


def test_a_stack_regrades_the_merge_of_its_frames(tmp_path, capsys):
    from pifilm.stack import merge

    d, names = _stack_dir(tmp_path)
    result = process_dir(d, tmp_path / "out")
    assert [p.name for p in result.written] == ["120001_original_graded.jpg"]
    assert result.stacks == 1 and result.stack_frames_missing == 0
    merged, info = merge([load_rgb(d / n)[0] for n in names], reference=0)
    assert all(f["used"] for f in info["frames"])  # a real merge, not the reference alone
    got = load_rgb(tmp_path / "out" / "120001_original_graded.jpg")[0]
    assert np.array_equal(got, _graded(merged, ev=-0.33, seed=77, tmp_path=tmp_path))
    alone = _graded(load_rgb(d / names[0])[0], ev=-0.33, seed=77, tmp_path=tmp_path)
    assert not np.array_equal(got, alone)
    assert main([str(d), str(tmp_path / "out2")]) == 0
    assert "1 multi-frame merged" in capsys.readouterr().out


def test_a_stack_with_a_frame_missing_grades_the_reference_alone(tmp_path, capsys):
    d, names = _stack_dir(tmp_path, n=4, missing=("120001_frame3.jpg",))
    result = process_dir(d, tmp_path / "out")
    assert result.stacks == 0 and result.stack_frames_missing == 1
    got = load_rgb(tmp_path / "out" / "120001_original_graded.jpg")[0]
    want = _graded(load_rgb(d / names[0])[0], ev=-0.33, seed=77, tmp_path=tmp_path)
    assert np.array_equal(got, want)  # the logged EV and seed still apply
    assert main([str(d), str(tmp_path / "out2")]) == 0
    assert "1 multi-frame with frames missing, reference graded alone" in (
        capsys.readouterr().out)


def test_ignore_log_grades_a_stack_reference_alone(tmp_path):
    d, names = _stack_dir(tmp_path)
    # Grain is fresh without the log, so compare grain-free grades.
    result = process_dir(d, tmp_path / "out", use_log=False, grain=False)
    assert result.stacks == 0 and result.stack_frames_missing == 0
    assert [p.name for p in result.written] == ["120001_original_graded.jpg"]
    got = load_rgb(tmp_path / "out" / "120001_original_graded.jpg")[0]
    want, _ = Pipeline(Artifacts.resolve(None)).process(load_rgb(d / names[0])[0], grain=False)
    assert np.array_equal(got, load_rgb(save_jpeg(want, tmp_path / "w.jpg"))[0])


def test_burst_frames_are_never_inputs(tmp_path):
    d, _ = _stack_dir(tmp_path)
    every = sorted(d.glob("*.jpg"))
    for all_files in (False, True):
        chosen = [p.name for p in select_inputs(every, all_files=all_files)]
        assert chosen == ["120001_original.jpg"]
    # Without any original in the folder, the frames are still not picked up.
    (d / "120001_original.jpg").unlink()
    save_jpeg(_img(1), d / "loose.jpg")
    assert [p.name for p in select_inputs(sorted(d.glob("*.jpg")))] == ["loose.jpg"]
    assert [p.name for p in process_dir(d, tmp_path / "out", all_files=True).written] == [
        "loose_graded.jpg"]


def test_a_bad_stack_is_an_unreadable_log_line(tmp_path):
    d = _logged_dir(tmp_path, [
        {"original": "120001_original.jpg", "stack": {"frames": "x"}},
        {"original": "120002_original.jpg", "stack": {"frames": ["../a.jpg", "b.jpg"]}},
        {"original": "120003_original.jpg", "grain_seed": 3,
         "stack": {"frames": ["120003_original.jpg"]}},
    ])
    log, bad = load_capture_log(d)
    assert bad == 2
    assert log["120003_original.jpg"].frames == ("120003_original.jpg",)
