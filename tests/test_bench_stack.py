import json
import re

import numpy as np
import pytest

from pifilm._cv2 import require_cv2
from pifilm.imageio import save_jpeg
from scripts import bench_stack as script


def _frame(seed, h=96, w=128):
    """A smooth scene, so the frames align, plus each frame's own noise."""
    cv2 = require_cv2()
    rng = np.random.default_rng(100)
    base = cv2.GaussianBlur(rng.random((h, w, 3), dtype=np.float32), (0, 0), 4)
    base = 128.0 + 60.0 * (base - base.mean()) / base.std()
    noise = np.random.default_rng(seed).normal(0.0, 12.0, (h, w, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def day(tmp_path):
    d = tmp_path / "day"
    d.mkdir()
    names = ["120001_original.jpg", "120001_frame2.jpg", "120001_frame3.jpg"]
    for i, name in enumerate(names):
        save_jpeg(_frame(i), d / name)
    save_jpeg(_frame(9), d / "120002_original.jpg")
    records = [
        {"original": "120001_original.jpg", "ev_comp": 0.0, "grain_seed": 1,
         "stack": {"frames": names, "reference": names[0]}},
        {"original": "120002_original.jpg", "ev_comp": 0.0, "grain_seed": 2},
    ]
    (d / "captures.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return d, names


def _median(out):
    m = re.search(r"median of (\d+): align_ms ([\d.]+), merge_ms ([\d.]+), total_ms ([\d.]+)",
                  out)
    assert m, out
    return int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))


@pytest.mark.parametrize("shot", ["120001", "120001_original.jpg"])
def test_a_record_s_frames_are_timed(day, capsys, shot):
    d, _names = day
    assert script.main(["--folder", str(d), "--shot", shot]) == 0
    out = capsys.readouterr().out
    assert "3 frames of 128x96, reference 120001_original.jpg" in out
    assert len(re.findall(r"^run \d: ", out, re.M)) == 3
    runs, align, merge, total = _median(out)
    assert runs == 3 and total >= align + merge - 1.0
    assert "frames used: 3 of 3" in out
    assert re.search(r"peak RSS: \d+ MB", out)


def test_listed_jpegs_are_timed_the_first_as_reference(day, capsys):
    d, names = day
    paths = [str(d / n) for n in reversed(names)]
    assert script.main(["--frames", *paths, "--runs", "1"]) == 0
    out = capsys.readouterr().out
    assert "reference 120001_frame3.jpg" in out
    assert _median(out)[0] == 1


@pytest.mark.parametrize(
    "argv_tail, message",
    [
        (["--shot", "120002"], "not a multi-frame photograph"),
        (["--shot", "120009"], "no record for 120009_original.jpg"),
    ],
)
def test_a_record_without_a_stack_is_refused(day, capsys, argv_tail, message):
    d, _names = day
    assert script.main(["--folder", str(d), *argv_tail]) == 2
    assert message in capsys.readouterr().err


def test_a_missing_frame_is_refused(day, capsys):
    d, _names = day
    (d / "120001_frame3.jpg").unlink()
    assert script.main(["--folder", str(d), "--shot", "120001"]) == 2
    assert "120001_frame3.jpg" in capsys.readouterr().err


def test_a_folder_without_a_log_is_refused(tmp_path, capsys):
    assert script.main(["--folder", str(tmp_path), "--shot", "120001"]) == 2
    assert "no captures.jsonl" in capsys.readouterr().err


def test_one_listed_frame_is_refused(day, capsys):
    d, names = day
    assert script.main(["--frames", str(d / names[0])]) == 2
    assert "at least two" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--folder", "x"],
        ["--frames", "a.jpg", "b.jpg", "--shot", "1"],
        ["--frames", "a.jpg", "b.jpg", "--runs", "0"],
        ["--folder", "x", "--shot", "1", "--frames", "a.jpg"],
    ],
)
def test_bad_arguments_are_usage_errors(argv):
    with pytest.raises(SystemExit) as exc:
        script.main(argv)
    assert exc.value.code == 2
