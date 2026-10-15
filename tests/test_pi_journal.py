from pathlib import Path

import pytest

from scripts import pi_journal
from scripts.pi_journal import (
    DROPIN,
    dropin_text,
    format_report,
    main,
    plan,
    run,
    summarise,
    undo_plan,
)


def test_the_dropin_sorts_after_the_os_volatile_setting():
    # journald reads drop-ins in file-name order across /usr/lib and /etc; the
    # last one wins, so ours must sort after 40-rpi-volatile-storage.conf.
    assert DROPIN.parent == Path("/etc/systemd/journald.conf.d")
    assert sorted([DROPIN.name, "40-rpi-volatile-storage.conf"])[-1] == DROPIN.name


def test_dropin_text_is_persistent_and_capped():
    text = dropin_text("200M", "1G")
    assert "[Journal]\n" in text
    assert "Storage=persistent\n" in text
    assert "SystemMaxUse=200M\n" in text
    assert "SystemKeepFree=1G\n" in text


def test_plan_writes_the_setting_then_restarts_and_flushes_in_that_order():
    steps = plan("200M", "1G")
    assert steps[0].path == DROPIN and steps[0].text == dropin_text("200M", "1G")
    argvs = [s.argv for s in steps]
    restart = argvs.index(("systemctl", "restart", "systemd-journald"))
    flush = argvs.index(("journalctl", "--flush"))
    tmpfiles = next(i for i, a in enumerate(argvs) if a[:1] == ("systemd-tmpfiles",))
    assert 0 < tmpfiles < restart < flush
    # Nothing deletes logs.
    assert not any(a[:1] == ("rm",) for a in argvs)


def test_undo_plan_removes_only_our_dropin():
    steps = undo_plan()
    assert steps[0].path == DROPIN and steps[0].remove
    assert steps[1].argv == ("systemctl", "restart", "systemd-journald")


def test_run_writes_atomically_with_a_readable_mode_and_removes(tmp_path):
    target = tmp_path / "conf.d" / "90-x.conf"
    run(pi_journal.Step("write", path=target, text="[Journal]\n"))
    assert target.read_text() == "[Journal]\n"
    assert target.stat().st_mode & 0o777 == 0o644
    assert not list(target.parent.glob("*.tmp"))
    run(pi_journal.Step("remove", path=target, remove=True))
    assert not target.exists()
    run(pi_journal.Step("remove again", path=target, remove=True))   # no error


def test_run_stops_on_a_failing_command():
    with pytest.raises(SystemExit, match="failed"):
        run(pi_journal.Step("fail", ("false",)))


def test_main_dry_run_prints_the_plan_and_changes_nothing(capsys, monkeypatch):
    monkeypatch.setattr(pi_journal, "run", lambda _s: pytest.fail("ran a step in a dry run"))
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "Dry run" in out and "Storage=persistent" in out and "--flush" in out
    assert main(["--undo"]) == 0
    assert "remove /etc/systemd/journald.conf.d/" in capsys.readouterr().out


def test_main_apply_refuses_without_root(capsys, monkeypatch):
    monkeypatch.setattr(pi_journal.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(pi_journal, "run", lambda _s: pytest.fail("ran a step without root"))
    assert main(["--apply"]) == 1
    assert "root" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["200", "0M", "1.5G", "200MB", "-1G", ""])
def test_sizes_are_validated_before_anything_runs(bad):
    if bad == "200":
        assert main(["--max-use", bad]) == 0   # a plain byte count is valid
        return
    with pytest.raises(SystemExit):
        main(["--max-use", bad])


def test_report_refuses_apply_and_undo():
    with pytest.raises(SystemExit):
        main(["--report", "--apply"])
    with pytest.raises(SystemExit):
        main(["--report", "--undo"])


UNIT_LINES = [
    "2026-10-15T10:00:00+02:00 parr systemd[1]: Started pifilm-capture.service - pi-film.",
    "2026-10-15T10:05:00+02:00 parr pifilm-capture[1]: camera: no preview frame within 5.0 s",
    "2026-10-15T10:05:05+02:00 parr pifilm-capture[1]: error: camera stopped delivering "
    "frames: x; exiting for a restart",
    "2026-10-15T10:05:06+02:00 parr systemd[1]: pifilm-capture.service: Main process "
    "exited, code=exited, status=1/FAILURE",
    "2026-10-15T10:05:06+02:00 parr systemd[1]: pifilm-capture.service: Failed with "
    "result 'exit-code'.",
    "2026-10-15T10:05:11+02:00 parr systemd[1]: Started pifilm-capture.service - pi-film.",
    "2026-10-15T11:00:00+02:00 parr systemd[1]: pifilm-capture.service: Main process "
    "exited, code=killed, status=9/KILL",
    "2026-10-15T11:00:05+02:00 parr systemd[1]: Started pifilm-capture.service - pi-film.",
    "2026-10-15T12:00:00+02:00 parr systemd[1]: pifilm-capture.service: Main process "
    "exited, code=exited, status=0/SUCCESS",
]
KERNEL_LINES = [
    "2026-10-15T09:59:00+02:00 parr kernel: Booting Linux",
    "2026-10-15T10:30:00+02:00 parr kernel: hwmon hwmon1: Undervoltage detected!",
    "2026-10-15T10:40:00+02:00 parr kernel: Out of memory: Killed process 123 (python3)",
]


def test_summarise_counts_starts_stalls_and_each_error_exit_once():
    (boot,) = summarise([("boot 0", UNIT_LINES, KERNEL_LINES)])
    assert boot.starts == 3
    assert boot.stalls == 1
    # The stall's exit (logged twice by systemd) and the kill; a clean exit is not one.
    assert boot.failures == 2
    assert "status=1/FAILURE" in boot.exits[0][-1]
    assert any("no preview frame" in line for line in boot.exits[0])
    assert "status=9/KILL" in boot.exits[1][-1]
    assert [line.split("kernel: ")[1] for line in boot.kernel] == [
        "hwmon hwmon1: Undervoltage detected!",
        "Out of memory: Killed process 123 (python3)",
    ]


def test_format_report_lists_every_boot_and_the_lines_before_each_exit():
    report = format_report(summarise([
        ("boot -1 from Wed 2026-10-14 15:21:49", [], []),
        ("boot 0 from Thu 2026-10-15 16:46:44", UNIT_LINES, KERNEL_LINES),
    ]))
    assert "boot -1 from Wed 2026-10-14 15:21:49" in report
    assert "error exit 1, the lines before it" in report
    assert "error exit 2" in report
    assert "Undervoltage detected" in report
    assert format_report([]) == "The journal holds no boots.\n"


def test_read_boots_asks_journalctl_per_boot(monkeypatch):
    calls = []

    class Done:
        def __init__(self, out):
            self.stdout = out
            self.returncode = 0

    def fake_run(argv, capture_output, text):
        calls.append(argv)
        if "--list-boots" in argv:
            return Done("IDX BOOT ID                          FIRST ENTRY                  "
                        "LAST ENTRY\n"
                        " -1 aaaa Wed 2026-10-14 15:21:49 CEST Wed 2026-10-14 16:17:21 CEST\n"
                        "  0 bbbb Thu 2026-10-15 16:46:44 CEST Thu 2026-10-15 17:31:15 CEST\n")
        return Done("-- No entries --\n" + "\n".join(UNIT_LINES[:1]) + "\n")

    monkeypatch.setattr(pi_journal.subprocess, "run", fake_run)
    boots = pi_journal.read_boots()
    assert [b[0] for b in boots] == ["boot -1 from Wed 2026-10-14 15:21:49",
                                     "boot 0 from Thu 2026-10-15 16:46:44"]
    assert boots[0][1] == UNIT_LINES[:1]          # the "-- No entries --" line is dropped
    assert ["journalctl", "--no-pager", "-o", "short-iso", "-b", "-1", "-u",
            "pifilm-capture.service"] in calls
    assert ["journalctl", "--no-pager", "-o", "short-iso", "-b", "0", "-k"] in calls
