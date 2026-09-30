# tests/test_double_capture.py
import json
import shutil
from datetime import datetime

import numpy as np
import pytest

from pifilm.artifacts import Artifacts, write_artifact
from pifilm.capture.app import CaptureSession, _announce
from pifilm.capture.camera import CameraError, FakeCamera
from pifilm.double import composite
from pifilm.grain import GrainParams
from pifilm.lut import LUT3D
from pifilm.normalize import NormalizeParams
from pifilm.pipeline import Pipeline

DARK = np.full((90, 160, 3), 10, dtype=np.uint8)
BRIGHT = np.full((90, 160, 3), 230, dtype=np.uint8)


@pytest.fixture
def pipeline(tmp_path):
    d = tmp_path / "art"
    write_artifact(d, LUT3D.identity(9), NormalizeParams(), GrainParams())
    return Pipeline(Artifacts.load(d))


def _session(tmp_path, pipeline, frames=(DARK, BRIGHT)):
    camera = FakeCamera([f.copy() for f in frames])
    return CaptureSession(camera, pipeline, tmp_path / "shots",
                          seed_rng=np.random.default_rng(0))


def _records(session):
    lines = []
    for f in sorted(session.out_root.glob("*/captures.jsonl")):
        lines += [json.loads(line) for line in f.read_text().splitlines()]
    return lines


def test_mode_is_off_by_default_and_single_shots_are_unchanged(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    assert session.double_state == (False, 0)
    result = session.capture()
    assert result.exposure is None
    assert result.pifilm.name.endswith("_graded.jpg")
    assert "double" not in _records(session)[0]


def test_exposure_one_saves_only_the_original_and_returns_the_card(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    session.set_double_exposure(True)
    result = session.capture()
    assert result.exposure == (1, 2)
    assert session.double_state == (True, 1)
    assert result.original.exists()
    day = result.original.parent
    assert not list(day.glob("*_graded.jpg"))
    # The card is served to the Stick but never lands in the synced photo folder.
    assert result.pifilm.exists()
    assert session.out_root not in result.pifilm.parents
    (record,) = _records(session)
    assert record["double"] == {"index": 1, "of": 2}
    assert "pifilm" not in record and record["original"] == result.original.name


def test_exposure_two_grades_the_composite_once(tmp_path, pipeline, monkeypatch):
    session = _session(tmp_path, pipeline)
    calls = []
    real = session.pipeline.process

    def spy(rgb, **kw):
        calls.append((rgb.copy(), kw))
        return real(rgb, **kw)

    monkeypatch.setattr(session.pipeline, "process", spy)
    session.set_double_exposure(True)
    first = session.capture()
    second = session.capture()

    assert len(calls) == 1  # graded once: one normalisation, one LUT, one grain pass
    graded_input, kw = calls[0]
    assert np.array_equal(graded_input, composite(DARK, BRIGHT))
    assert kw["ev"] == 0.0
    assert second.exposure == (2, 2)
    assert second.pifilm.name.endswith("_double_graded.jpg") and second.pifilm.exists()
    assert session.double_state == (True, 0)
    record = _records(session)[1]
    assert record["pifilm"] == second.pifilm.name
    assert record["double"]["index"] == 2 and record["double"]["method"] == "linear_mean"
    assert record["double"]["originals"] == [
        first.original.relative_to(session.out_root).as_posix(),
        second.original.relative_to(session.out_root).as_posix(),
    ]
    assert record["double"]["ev_comp"] == [0.0, 0.0]
    assert isinstance(record["grain_seed"], int) and "lut_sha1" in record


def test_each_exposure_keeps_the_ev_it_was_shot_at(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    session.set_double_exposure(True)
    session.camera.set_ev(-1.0)
    session.capture()
    session.camera.set_ev(0.0)
    session.capture()
    assert _records(session)[1]["double"]["ev_comp"] == [-1.0, 0.0]


def test_the_count_cycles_back_for_the_next_pair(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    session.set_double_exposure(True)
    assert [session.capture().exposure for _ in range(4)] == [(1, 2), (2, 2), (1, 2), (2, 2)]


def test_toggling_off_at_one_of_two_drops_the_pending_frame(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    session.set_double_exposure(True)
    session.capture()
    session.set_double_exposure(False)
    assert session.double_state == (False, 0)
    session.set_double_exposure(True)
    assert session.capture().exposure == (1, 2)


def test_toggling_on_again_keeps_the_pending_frame(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    session.set_double_exposure(True)
    session.capture()
    session.set_double_exposure(True)
    assert session.double_state == (True, 1)
    assert session.capture().exposure == (2, 2)


def test_a_camera_error_on_exposure_two_keeps_one_of_two(tmp_path, pipeline, monkeypatch):
    session = _session(tmp_path, pipeline)
    session.set_double_exposure(True)
    session.capture()
    real = session.camera.read

    def broken(*, full=True):
        raise CameraError("no frame")

    monkeypatch.setattr(session.camera, "read", broken)
    with pytest.raises(CameraError):
        session.capture()
    assert session.double_state == (True, 1)
    monkeypatch.setattr(session.camera, "read", real)
    assert session.capture().exposure == (2, 2)


def test_a_failure_after_exposure_two_is_read_returns_to_zero(tmp_path, pipeline, monkeypatch):
    session = _session(tmp_path, pipeline)
    session.set_double_exposure(True)
    session.capture()

    def boom(*a, **k):
        raise RuntimeError("grade failed")

    monkeypatch.setattr(session.pipeline, "process", boom)
    with pytest.raises(RuntimeError):
        session.capture()
    assert session.double_state == (True, 0)
    assert len(list(session.out_root.glob("*/*_original.jpg"))) == 0  # decoded source
    assert len(list(session.out_root.glob("*/*_ungraded.jpg"))) == 2  # both kept


def test_frames_of_different_sizes_fail_and_reset(tmp_path, pipeline):
    small = np.full((45, 80, 3), 128, dtype=np.uint8)
    session = _session(tmp_path, pipeline, frames=(DARK, small))
    session.set_double_exposure(True)
    session.capture()
    with pytest.raises(ValueError):
        session.capture()
    assert session.double_state == (True, 0)


def test_announce_handles_an_exposure_one_result(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    session.set_double_exposure(True)
    lines = []
    _announce(session.capture(), lines.append)
    assert len(lines) == 1 and "exposure 1/2" in lines[0]
    _announce(session.capture(), lines.append)
    assert "_double_graded.jpg" in lines[1]


def test_a_pair_across_midnight_names_both_day_folders(tmp_path, pipeline):
    times = iter([datetime(2026, 9, 30, 23, 59, 58), datetime(2026, 10, 1, 0, 0, 3)])
    camera = FakeCamera([DARK.copy(), BRIGHT.copy()])
    session = CaptureSession(camera, pipeline, tmp_path / "shots", now=lambda: next(times),
                             seed_rng=np.random.default_rng(0))
    session.set_double_exposure(True)
    session.capture()
    second = session.capture()
    first_name, second_name = second.record["double"]["originals"]
    assert first_name == "2026-09-30/235958_ungraded.jpg"
    assert second_name == "2026-10-01/000003_ungraded.jpg"
    assert (session.out_root / first_name).exists() and (session.out_root / second_name).exists()


def test_exposure_one_survives_the_card_directory_being_cleaned(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    session.set_double_exposure(True)
    card = session.capture().pifilm
    session.capture()
    shutil.rmtree(card.parent)  # systemd-tmpfiles ages /tmp while the service runs
    result = session.capture()
    assert result.exposure == (1, 2)
    assert result.pifilm.exists()


def test_a_card_failure_fails_exposure_one_without_a_log_line(tmp_path, pipeline, monkeypatch):
    session = _session(tmp_path, pipeline)
    session.set_double_exposure(True)

    def boom(*a, **k):
        raise OSError("card failed")

    monkeypatch.setattr("pifilm.capture.app.render_exposure_card", boom)
    with pytest.raises(OSError):
        session.capture()
    assert _records(session) == []
    assert session.double_state == (True, 0)
