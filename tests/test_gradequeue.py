import json
import stat
import threading
from pathlib import Path

import pytest

from pifilm.capture import gradequeue
from pifilm.capture.gradequeue import (
    ENTRY_VERSION,
    MAX_ATTEMPTS,
    GradeJob,
    GradeQueue,
    default_state_dir,
    note_failure,
    read_entries,
    remove_entry,
    write_entry,
)

WAIT = 5.0  # the ceiling on every wait; a healthy run never gets near it
BLOCKED = 0.1  # how long a call must stay blocked before the test believes it


class Recorder:
    """A ``run`` that records payloads and can hold chosen ones on an event."""

    def __init__(self, hold=()):
        self.ran = []
        self.jobs = []
        self.started = {name: threading.Event() for name in hold}
        self.release = {name: threading.Event() for name in hold}
        self.done = threading.Event()
        self.expect = None

    def __call__(self, job):
        if job.payload in self.started:
            self.started[job.payload].set()
            assert self.release[job.payload].wait(WAIT)
        self.ran.append(job.payload)
        self.jobs.append(job)
        if self.expect is not None and len(self.ran) >= self.expect:
            self.done.set()


def test_jobs_run_in_submission_order_with_times_set():
    run = Recorder()
    run.expect = 3
    queue = GradeQueue(run)
    try:
        for name in ("a", "b", "c"):
            assert queue.submit(GradeJob(name)) is True
        assert run.done.wait(WAIT)
    finally:
        assert queue.close(WAIT)
    assert run.ran == ["a", "b", "c"]
    for job in run.jobs:
        assert job.queued_at > 0.0
        assert job.started_at >= job.queued_at


def test_backlog_runs_before_submitted_jobs():
    run = Recorder(hold=["old-1"])
    backlog = [GradeJob("old-1"), GradeJob("old-2"), GradeJob("old-3"), GradeJob("old-4")]
    # Four backlog jobs with depth 1: the backlog is not bounded by depth.
    queue = GradeQueue(run, depth=1, backlog=backlog)
    try:
        assert run.started["old-1"].wait(WAIT)
        assert queue.submit(GradeJob("new")) is True
        run.release["old-1"].set()
    finally:
        assert queue.close(WAIT)
    assert run.ran == ["old-1", "old-2", "old-3", "old-4", "new"]
    assert all(job.started_at >= job.queued_at > 0.0 for job in run.jobs)


def test_submit_blocks_on_a_full_queue_until_a_job_is_taken():
    run = Recorder(hold=["held", "w1"])
    queue = GradeQueue(run, depth=2)
    accepted = []
    returned = threading.Event()

    def third():
        accepted.append(queue.submit(GradeJob("w3")))
        returned.set()

    try:
        assert queue.submit(GradeJob("held")) is True
        assert run.started["held"].wait(WAIT)  # running, so it no longer counts as waiting
        assert queue.submit(GradeJob("w1")) is True
        assert queue.submit(GradeJob("w2")) is True
        helper = threading.Thread(target=third, daemon=True)
        helper.start()
        assert not returned.wait(BLOCKED)
        run.release["held"].set()
        assert returned.wait(WAIT)
        assert accepted == [True]
        run.release["w1"].set()
        helper.join(WAIT)
    finally:
        for event in run.release.values():
            event.set()
        assert queue.close(WAIT)
    assert run.ran == ["held", "w1", "w2", "w3"]


def test_pending_counts_running_waiting_and_backlog():
    run = Recorder(hold=["old-1"])
    queue = GradeQueue(run, depth=3, backlog=[GradeJob("old-1"), GradeJob("old-2")])
    try:
        assert queue.pending == 2
        assert run.started["old-1"].wait(WAIT)
        assert queue.pending == 2  # one running, one backlog
        assert queue.submit(GradeJob("new-1"))
        assert queue.submit(GradeJob("new-2"))
        assert queue.pending == 4
        run.release["old-1"].set()
    finally:
        run.release["old-1"].set()
        assert queue.close(WAIT)
    assert queue.pending == 0
    assert len(run.ran) == 4


def test_a_failing_run_is_logged_counted_and_does_not_stop_the_worker(tmp_path):
    entry = write_entry(tmp_path, "120000_shot", {"original": "x.jpg"})
    after = threading.Event()
    ran = []
    lines = []

    def run(job):
        if job.payload == "bad":
            raise OSError("disk full")
        ran.append(job.payload)
        after.set()

    queue = GradeQueue(run, log=lines.append)
    try:
        assert queue.submit(GradeJob("bad", entry=entry))
        assert queue.submit(GradeJob("good"))
        # The worker notes the failure before it takes the next job.
        assert after.wait(WAIT)
        assert ran == ["good"]
        assert lines == ["grade failed: disk full"]
        assert json.loads(entry.read_text())["attempts"] == 1
        assert read_entries(tmp_path)[0][1]["original"] == "x.jpg"

        assert MAX_ATTEMPTS == 2
        assert queue.submit(GradeJob("bad", entry=entry))
    finally:
        assert queue.close(WAIT)
    assert not entry.exists()
    assert (tmp_path / "120000_shot.failed").exists()
    assert read_entries(tmp_path) == []
    assert len(lines) == 2


def test_a_failing_job_without_an_entry_and_a_failing_log_do_not_stop_the_worker():
    ran = []

    def run(job):
        if job.payload == "bad":
            raise RuntimeError("boom")
        ran.append(job.payload)

    def log(line):
        raise OSError("stdout is gone")

    queue = GradeQueue(run, log=log)
    assert queue.submit(GradeJob("bad"))
    assert queue.submit(GradeJob("good"))
    assert queue.close(WAIT)
    assert ran == ["good"]
    assert queue.pending == 0


def test_close_finishes_accepted_jobs_and_refuses_later_ones():
    run = Recorder(hold=["a"])
    queue = GradeQueue(run, backlog=[GradeJob("old")])
    assert queue.submit(GradeJob("a"))
    assert queue.submit(GradeJob("b"))
    assert queue.submit(GradeJob("c"))
    run.release["a"].set()
    assert queue.close(WAIT) is True
    assert run.ran == ["old", "a", "b", "c"]
    assert queue.submit(GradeJob("late")) is False
    assert queue.close() is True
    assert run.ran == ["old", "a", "b", "c"]
    assert queue.pending == 0


def test_close_wakes_a_blocked_submit_which_returns_false():
    run = Recorder(hold=["held"])
    queue = GradeQueue(run, depth=1)
    accepted = []
    returned = threading.Event()
    closed = []

    def blocked():
        accepted.append(queue.submit(GradeJob("dropped")))
        returned.set()

    try:
        assert queue.submit(GradeJob("held"))
        assert run.started["held"].wait(WAIT)
        assert queue.submit(GradeJob("waiting"))
        helper = threading.Thread(target=blocked, daemon=True)
        helper.start()
        assert not returned.wait(BLOCKED)

        closer = threading.Thread(target=lambda: closed.append(queue.close(WAIT)), daemon=True)
        closer.start()
        # The held job is still held: the submit is woken by close, not by a free slot.
        assert returned.wait(WAIT)
        assert accepted == [False]
        assert not run.release["held"].is_set()
    finally:
        run.release["held"].set()
    closer.join(WAIT)
    assert closed == [True]
    assert run.ran == ["held", "waiting"]  # the refused job was never run


def test_close_with_a_timeout_reports_an_unfinished_worker():
    run = Recorder(hold=["held"])
    queue = GradeQueue(run)
    try:
        assert queue.submit(GradeJob("held"))
        assert run.started["held"].wait(WAIT)
        assert queue.close(timeout=0.05) is False
        assert queue.pending == 1
    finally:
        run.release["held"].set()
    assert queue.close(WAIT) is True


def test_depth_below_one_is_refused():
    with pytest.raises(ValueError):
        GradeQueue(lambda job: None, depth=0)


def test_the_worker_is_a_named_daemon_thread():
    seen = []
    queue = GradeQueue(lambda job: seen.append(threading.current_thread()))
    assert queue.submit(GradeJob("a"))
    assert queue.close(WAIT)
    assert seen[0].name == "pifilm-grade"
    assert seen[0].daemon


def test_write_entry_round_trips_through_read_entries(tmp_path):
    state = tmp_path / "state" / "pending"
    path = write_entry(state, "120000_shot", {"original": "/p/a.jpg", "grain_seed": 7})

    assert path == state / "120000_shot.json"
    assert stat.S_IMODE(state.stat().st_mode) == 0o700
    assert [p.name for p in state.iterdir()] == ["120000_shot.json"]  # no temp file left
    assert read_entries(state) == [
        (
            path,
            {"original": "/p/a.jpg", "grain_seed": 7, "version": ENTRY_VERSION, "attempts": 0},
        )
    ]


def test_write_entry_keeps_a_version_and_attempts_it_was_given(tmp_path):
    payload = {"version": ENTRY_VERSION, "attempts": 1}
    path = write_entry(tmp_path, "a", payload)
    assert json.loads(path.read_text()) == {"version": ENTRY_VERSION, "attempts": 1}
    assert payload == {"version": ENTRY_VERSION, "attempts": 1}

    untouched = {"original": "a.jpg"}
    write_entry(tmp_path, "b", untouched)
    assert untouched == {"original": "a.jpg"}  # the caller's dict is not changed


def test_read_entries_sets_aside_what_it_cannot_use(tmp_path):
    assert read_entries(tmp_path / "missing") == []

    second = write_entry(tmp_path, "b_second", {"n": 2})
    first = write_entry(tmp_path, "a_first", {"n": 1})
    (tmp_path / "c_broken.json").write_text("{not json")
    (tmp_path / "d_array.json").write_text("[1, 2]")
    (tmp_path / "e_future.json").write_text(json.dumps({"version": 99, "attempts": 0}))
    (tmp_path / "f_binary.json").write_bytes(b"\xff\xfe\x00")

    entries = read_entries(tmp_path)

    assert [path for path, _ in entries] == [first, second]
    assert [payload["n"] for _, payload in entries] == [1, 2]
    for name in ("c_broken", "d_array", "e_future", "f_binary"):
        assert not (tmp_path / f"{name}.json").exists()
        assert (tmp_path / f"{name}.failed").exists()
    assert read_entries(tmp_path) == entries


def test_note_failure_counts_then_sets_the_entry_aside(tmp_path):
    path = write_entry(tmp_path, "shot", {"original": "a.jpg"})

    assert note_failure(path) is True
    assert json.loads(path.read_text()) == {
        "original": "a.jpg", "version": ENTRY_VERSION, "attempts": 1,
    }
    assert [p.name for p in tmp_path.iterdir()] == ["shot.json"]

    assert note_failure(path) is False
    assert [p.name for p in tmp_path.iterdir()] == ["shot.failed"]
    assert json.loads((tmp_path / "shot.failed").read_text())["attempts"] == MAX_ATTEMPTS


def test_note_failure_and_remove_entry_tolerate_missing_files(tmp_path):
    missing = tmp_path / "gone.json"
    assert note_failure(missing) is False
    remove_entry(None)
    remove_entry(missing)

    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert note_failure(broken) is False
    assert (tmp_path / "broken.failed").exists()

    path = write_entry(tmp_path, "shot", {})
    remove_entry(path)
    assert not path.exists()


def test_default_state_dir_honours_xdg_state_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    assert default_state_dir() == tmp_path / "xdg" / "pifilm" / "pending"

    monkeypatch.delenv("XDG_STATE_HOME")
    assert default_state_dir() == Path.home() / ".local" / "state" / "pifilm" / "pending"

    monkeypatch.setenv("XDG_STATE_HOME", "")
    assert default_state_dir() == Path.home() / ".local" / "state" / "pifilm" / "pending"


@pytest.mark.parametrize("name", ["", "../x", "a/b", ".hidden"])
def test_write_entry_refuses_a_name_that_is_not_a_plain_file_name(tmp_path, name):
    with pytest.raises(ValueError, match="plain file name"):
        gradequeue.write_entry(tmp_path / "state", name, {})
    assert not list(tmp_path.rglob("*.json"))


def test_write_entry_syncs_the_directory(tmp_path, monkeypatch):
    """The rename has to survive a power cut too, not only the file's bytes."""
    synced = []
    monkeypatch.setattr(gradequeue, "_fsync_dir", synced.append)
    gradequeue.write_entry(tmp_path / "state", "shot", {})
    assert synced == [tmp_path / "state"]


def test_a_failure_count_that_cannot_be_written_sets_the_entry_aside(tmp_path, monkeypatch):
    path = gradequeue.write_entry(tmp_path, "shot", {})

    def full(*_):
        raise OSError("disk full")

    monkeypatch.setattr(gradequeue, "_write_atomic", full)
    assert gradequeue.note_failure(path) is False
    assert not path.exists() and path.with_suffix(".failed").exists()
