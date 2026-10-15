"""Keep the Pi's journal across reboots, and report the capture service's restarts.

Run this ON THE PI. Raspberry Pi OS keeps the systemd journal in memory only
(``/usr/lib/systemd/journald.conf.d/40-rpi-volatile-storage.conf`` sets
``Storage=volatile``), so every reboot or power cut loses what the capture
service logged before it: a camera stall, a crash, the battery running out. On
2026-10-15 several restarts during outdoor use left nothing to look at, because
the Pi had been switched off and on since. ``docs/known-issues.md`` (the Pi 4
preview stall) asks for exactly that history.

Without ``--apply`` nothing changes; the plan is printed. With ``--apply``, as
root, it:

1. writes ``/etc/systemd/journald.conf.d/90-pifilm-persistent.conf`` with
   ``Storage=persistent``. Drop-ins are read in file-name order across
   ``/usr/lib`` and ``/etc`` and a later one wins, so ``90-`` overrides the
   ``40-`` file without editing a file the OS owns (an update would put it back).
   ``SystemMaxUse`` caps the journal (200 MB by default, weeks of this service's
   logging) and ``SystemKeepFree`` leaves room for photographs (1 GB by default),
   so the journal can never fill the card;
2. makes sure ``/var/log/journal`` exists with journald's own ownership and ACLs
   (``systemd-tmpfiles``);
3. restarts journald, which reads the new setting;
4. ``journalctl --flush``: moves the log of the current boot from memory onto the
   card, so nothing since the last boot is lost by the change;
5. prints how much the journal takes on the card, to confirm it is there.

``--undo --apply`` removes the drop-in and restarts journald: back to memory only. The
files already on the card are left (``journalctl --directory /var/log/journal``
still reads them); delete them with ``sudo rm -rf /var/log/journal/*`` if wanted.

``--report`` (no root needed) summarises the capture service across every boot
the journal holds: per boot, how often it started, how often it ended on a
camera stall, and how often it exited with an error; the last lines before each
error exit; and any undervoltage or out-of-memory message from the kernel, the
two other usual reasons for a Pi to restart a service or reboot.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

DROPIN = Path("/etc/systemd/journald.conf.d/90-pifilm-persistent.conf")
JOURNAL_DIR = Path("/var/log/journal")
UNIT = "pifilm-capture.service"
DEFAULT_MAX_USE = "200M"
DEFAULT_KEEP_FREE = "1G"
# journald accepts a number with an optional K, M, G or T suffix (base 1024).
_SIZE = re.compile(r"^[1-9][0-9]*[KMGT]?$")
# Lines kept from before each error exit in the report.
CONTEXT_LINES = 12


@dataclass(frozen=True)
class Step:
    """One action: a command to run, or a file to write (``path`` + ``text``)."""

    description: str
    argv: tuple[str, ...] = ()
    path: Path | None = None
    text: str = ""
    remove: bool = False


def dropin_text(max_use: str, keep_free: str) -> str:
    return ("# Written by scripts/pi_journal.py: keep the journal across reboots.\n"
            "[Journal]\nStorage=persistent\n"
            f"SystemMaxUse={max_use}\nSystemKeepFree={keep_free}\n")


def plan(max_use: str, keep_free: str) -> list[Step]:
    return [
        Step("override the OS's memory-only journal", path=DROPIN,
             text=dropin_text(max_use, keep_free)),
        Step("create the journal directory with journald's ownership and ACLs",
             ("systemd-tmpfiles", "--create", "--prefix", str(JOURNAL_DIR))),
        Step("restart journald so it reads the setting",
             ("systemctl", "restart", "systemd-journald")),
        Step("move this boot's log from memory onto the card", ("journalctl", "--flush")),
        Step("confirm: the journal's size on the card", ("journalctl", "--disk-usage")),
    ]


def undo_plan() -> list[Step]:
    return [
        Step("remove the override", path=DROPIN, remove=True),
        Step("restart journald (memory only again)",
             ("systemctl", "restart", "systemd-journald")),
    ]


def describe(step: Step) -> str:
    if step.path is not None:
        if step.remove:
            return f"{step.description}\n    remove {step.path}\n"
        return f"{step.description}\n    write {step.path}:\n" + "".join(
            f"      {line}\n" for line in step.text.splitlines())
    return f"{step.description}\n    $ {' '.join(step.argv)}\n"


def run(step: Step) -> str:
    if step.path is not None:
        if step.remove:
            step.path.unlink(missing_ok=True)
            return ""
        step.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = step.path.with_name(step.path.name + ".tmp")
        tmp.write_text(step.text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, step.path)
        return ""
    result = subprocess.run(step.argv, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"failed: {' '.join(step.argv)}\n{result.stderr.strip()}")
    return result.stdout.strip()


# --- the report -----------------------------------------------------------------

_STALL = "camera stopped delivering frames"
_STARTED = re.compile(r"systemd\[1\]: Started pifilm-capture")
_FAILED = re.compile(r"pifilm-capture\.service: (Main process exited, code=\w+, "
                     r"status=(?!0/)|Failed with result)")
_KERNEL = re.compile(r"(?i)under-?voltage|out of memory|oom-kill|killed process")


@dataclass
class Boot:
    label: str
    starts: int = 0
    stalls: int = 0
    failures: int = 0
    exits: list[list[str]] = field(default_factory=list)
    kernel: list[str] = field(default_factory=list)


def summarise(boots: list[tuple[str, list[str], list[str]]]) -> list[Boot]:
    """``boots``: (label, the unit's lines, the kernel's lines) per boot, oldest first.

    A failure is counted once per exit: systemd logs ``Main process exited`` and then
    ``Failed with result`` for the same exit, so only the first after a start counts.
    """
    out = []
    for label, unit_lines, kernel_lines in boots:
        b = Boot(label)
        counted = False
        for i, line in enumerate(unit_lines):
            if _STARTED.search(line):
                b.starts += 1
                counted = False
            elif _STALL in line:
                b.stalls += 1
            elif _FAILED.search(line) and not counted:
                b.failures += 1
                counted = True
                b.exits.append(unit_lines[max(0, i - CONTEXT_LINES):i + 1])
        b.kernel = [line for line in kernel_lines if _KERNEL.search(line)]
        out.append(b)
    return out


def format_report(summary: list[Boot]) -> str:
    if not summary:
        return "The journal holds no boots.\n"
    lines = ["boot                                       starts  stalls  error exits"]
    for b in summary:
        lines.append(f"{b.label:<42} {b.starts:6} {b.stalls:7} {b.failures:12}")
    for b in summary:
        for n, exit_lines in enumerate(b.exits, 1):
            lines.append(f"\n--- {b.label}: error exit {n}, the lines before it")
            lines.extend(exit_lines)
        if b.kernel:
            lines.append(f"\n--- {b.label}: kernel messages (undervoltage, out of memory)")
            lines.extend(b.kernel)
    return "\n".join(lines) + "\n"


def _journal(*args: str) -> list[str]:
    result = subprocess.run(["journalctl", "--no-pager", "-o", "short-iso", *args],
                            capture_output=True, text=True)
    return [line for line in result.stdout.splitlines() if line and not line.startswith("--")]


def read_boots() -> list[tuple[str, list[str], list[str]]]:
    listing = subprocess.run(["journalctl", "--list-boots", "--no-pager"],
                             capture_output=True, text=True).stdout.splitlines()
    boots = []
    for row in listing[1:]:   # the first line is the header
        parts = row.split()
        if len(parts) < 3:
            continue
        idx, first = parts[0], " ".join(parts[2:5])
        boots.append((f"boot {idx} from {first}", _journal("-b", idx, "-u", UNIT),
                      _journal("-b", idx, "-k")))
    return boots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="make the change (as root)")
    parser.add_argument("--undo", action="store_true",
                        help="plan the way back to memory only (with --apply, do it)")
    parser.add_argument("--report", action="store_true",
                        help="summarise the capture service's restarts across boots")
    parser.add_argument("--max-use", default=DEFAULT_MAX_USE,
                        help=f"cap on the journal's size on the card (default {DEFAULT_MAX_USE})")
    parser.add_argument("--keep-free", default=DEFAULT_KEEP_FREE,
                        help=f"space the journal always leaves free (default {DEFAULT_KEEP_FREE})")
    args = parser.parse_args(argv)
    for name in ("max_use", "keep_free"):
        if not _SIZE.match(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must look like 200M or 1G, "
                         f"got {getattr(args, name)!r}")

    if args.report and (args.apply or args.undo):
        parser.error("--report reads only; it takes neither --apply nor --undo")
    if args.report:
        sys.stdout.write(format_report(summarise(read_boots())))
        return 0
    steps = undo_plan() if args.undo else plan(args.max_use, args.keep_free)
    if not args.apply:
        print("Dry run; nothing changed. Steps with --apply:\n")
        for i, step in enumerate(steps, 1):
            print(f"{i}. {describe(step)}")
        return 0
    if os.geteuid() != 0:
        print("--apply must run as root (sudo)", file=sys.stderr)
        return 1
    for i, step in enumerate(steps, 1):
        print(f"{i}/{len(steps)} {step.description}", flush=True)
        output = run(step)
        if output:
            print(f"    {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
