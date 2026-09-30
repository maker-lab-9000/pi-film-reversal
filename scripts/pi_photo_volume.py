"""Move the Pi's photo folder onto its own fixed-size filesystem.

Run this ON THE PI. The Pi boots from one SD card whose root partition fills
the card, so photos and the operating system share one filesystem: a burst of
12 MP captures plus DNGs (about 14 GB after the first month) could fill root
and take the OS down with it. Shrinking root needs the card offline, so this
creates a preallocated ext4 image file instead and mounts it over the photo
folder. The image's size is reserved up front, so photos can never take more
than that from root, and a full photo volume fails a capture instead of the Pi.

Without ``--apply`` nothing changes; the plan is printed. With ``--apply``,
as root, it:

1. stops the capture service and the Nextcloud sync so nothing writes;
2. creates and formats the image (``fallocate``, so the space is reserved);
3. copies the photos in with ``rsync -aHAX`` and verifies the copy with a
   checksum dry run, which must report no differences;
4. moves the old folder aside to ``<folder>.pre-volume`` (kept, not deleted);
5. makes the empty mount point immutable (``chattr +i``), so if the image ever
   fails to mount, writes into the bare folder fail loudly instead of quietly
   filling root again;
6. adds an fstab line (``nofail``: a broken image must not stop the Pi booting)
   and ``RequiresMountsFor`` drop-ins so capture and sync wait for the mount;
7. mounts it and restarts the services.

rsync keeps modification times and rclone compares size and time, so the
Nextcloud sync does not re-upload anything after the move. Delete
``<folder>.pre-volume`` once a capture has landed on the new volume.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_USER = "george"
DEFAULT_IMAGE = Path("/var/lib/pifilm/photos.img")
DEFAULT_SIZE_GIB = 30
STAGING = Path("/mnt/pifilm-photos-new")
SERVICES = ("pifilm-capture.service", "pifilm-nextcloud-sync.service")
STOP_UNITS = ("pifilm-nextcloud-sync.timer", *SERVICES)
START_UNITS = ("pifilm-nextcloud-sync.timer", "pifilm-capture.service")
# Room root keeps while the old copy and the new image both exist.
ROOT_MARGIN_GIB = 2
DROPIN_NAME = "photo-volume.conf"


@dataclass(frozen=True)
class Step:
    """One action: a command to run, or a file to write (``path`` + ``text``)."""

    description: str
    argv: tuple[str, ...] = ()
    path: Path | None = None
    text: str = ""
    append: bool = False


def fstab_line(image: Path, mount_point: Path) -> str:
    # Pass 0: systemd's fsck wants a block device, not a file.
    return f"{image} {mount_point} ext4 loop,noatime,nofail 0 0\n"


def dropin_text(mount_point: Path) -> str:
    return f"[Unit]\nRequiresMountsFor={mount_point}\n"


def preflight(mount_point: Path, image: Path, size_gib: int, *, fstab: str,
              root_free_bytes: int, is_mount: bool) -> list[str]:
    """Reasons to refuse; empty when the move can go ahead."""
    problems = []
    if is_mount:
        problems.append(f"{mount_point} is already a mount point; nothing to do")
    if not mount_point.is_dir():
        problems.append(f"{mount_point} does not exist")
    if image.exists():
        problems.append(f"{image} already exists; remove it or pass --image")
    if str(mount_point) in fstab:
        problems.append(f"/etc/fstab already mentions {mount_point}")
    need = (size_gib + ROOT_MARGIN_GIB) * 1024**3
    if root_free_bytes < need:
        problems.append(
            f"root has {root_free_bytes / 1024**3:.1f} GiB free; the image needs "
            f"{size_gib} GiB plus {ROOT_MARGIN_GIB} GiB headroom while the old copy exists"
        )
    return problems


def plan(user: str, mount_point: Path, image: Path, size_gib: int) -> list[Step]:
    old = mount_point.with_name(mount_point.name + ".pre-volume")
    steps = [Step("stop capture and sync", ("systemctl", "stop", *STOP_UNITS))]
    steps += [
        Step("create the image directory", ("mkdir", "-p", str(image.parent))),
        Step(f"reserve {size_gib} GiB", ("fallocate", "-l", f"{size_gib}G", str(image))),
        Step("restrict the image to root", ("chmod", "600", str(image))),
        Step("format it (no reserved blocks: nothing but photos lives here)",
             ("mkfs.ext4", "-q", "-m", "0", "-L", "pifilm-photos", str(image))),
        Step("create the staging mount point", ("mkdir", "-p", str(STAGING))),
        Step("mount the image for the copy",
             ("mount", "-o", "loop,noatime", str(image), str(STAGING))),
        Step("copy the photos", ("rsync", "-aHAX", f"{mount_point}/", f"{STAGING}/")),
        Step("give the volume's root to the user", ("chown", f"{user}:{user}", str(STAGING))),
        Step("verify: a checksum dry run must list no differences",
             ("rsync", "-aHAXn", "--checksum", "--itemize-changes",
              f"{mount_point}/", f"{STAGING}/")),
        Step("unmount the staging copy", ("umount", str(STAGING))),
        Step("keep the old folder aside", ("mv", str(mount_point), str(old))),
        Step("create the empty mount point", ("mkdir", str(mount_point))),
        Step("own it", ("chown", f"{user}:{user}", str(mount_point))),
        Step("make the bare mount point immutable", ("chattr", "+i", str(mount_point))),
        Step("add the fstab line", path=Path("/etc/fstab"),
             text=fstab_line(image, mount_point), append=True),
    ]
    for service in SERVICES:
        steps.append(Step(f"make {service} wait for the mount",
                          path=Path(f"/etc/systemd/system/{service}.d/{DROPIN_NAME}"),
                          text=dropin_text(mount_point)))
    steps += [
        Step("reload systemd", ("systemctl", "daemon-reload")),
        Step("mount the photo volume", ("mount", str(mount_point))),
        Step("confirm it is mounted", ("findmnt", str(mount_point))),
        Step("restart capture and sync", ("systemctl", "start", *START_UNITS)),
    ]
    return steps


def describe(step: Step) -> str:
    if step.path is not None:
        verb = "append to" if step.append else "write"
        return f"{step.description}\n    {verb} {step.path}:\n" + "".join(
            f"      {line}\n" for line in step.text.splitlines())
    return f"{step.description}\n    $ {' '.join(step.argv)}\n"


def run(step: Step) -> None:
    if step.path is not None:
        step.path.parent.mkdir(parents=True, exist_ok=True)
        with open(step.path, "a" if step.append else "w") as f:
            f.write(step.text)
        return
    result = subprocess.run(step.argv, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"failed: {' '.join(step.argv)}\n{result.stderr.strip()}")
    if "--itemize-changes" in step.argv and result.stdout.strip():
        raise SystemExit("copy verification found differences; stopping before the "
                         "old folder is touched:\n" + result.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--mount-point", type=Path, default=None,
                        help="photo folder (default: ~USER/Pictures/pifilm)")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--size-gib", type=int, default=DEFAULT_SIZE_GIB)
    parser.add_argument("--apply", action="store_true", help="make the change (as root)")
    args = parser.parse_args(argv)
    mount_point = args.mount_point or Path(f"/home/{args.user}/Pictures/pifilm")

    fstab = Path("/etc/fstab").read_text() if Path("/etc/fstab").exists() else ""
    problems = preflight(mount_point, args.image, args.size_gib, fstab=fstab,
                         root_free_bytes=shutil.disk_usage("/").free,
                         is_mount=os.path.ismount(mount_point))
    steps = plan(args.user, mount_point, args.image, args.size_gib)
    if problems:
        for p in problems:
            print(f"refusing: {p}", file=sys.stderr)
        return 1
    if not args.apply:
        print("Dry run; nothing changed. Steps with --apply:\n")
        for i, step in enumerate(steps, 1):
            print(f"{i:2}. {describe(step)}")
        return 0
    if os.geteuid() != 0:
        print("--apply must run as root (sudo)", file=sys.stderr)
        return 1
    for i, step in enumerate(steps, 1):
        print(f"{i:2}/{len(steps)} {step.description}", flush=True)
        run(step)
    old = mount_point.with_name(mount_point.name + ".pre-volume")
    print(f"\nDone. After a capture lands in {mount_point}, free root with:\n"
          f"  sudo rm -rf {old}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
