from pathlib import Path

from scripts.pi_photo_volume import (
    SERVICES,
    dropin_text,
    fstab_line,
    main,
    plan,
    preflight,
)

GIB = 1024**3


def _ok(tmp_path, **overrides):
    kwargs = dict(fstab="", root_free_bytes=40 * GIB, is_mount=False)
    kwargs.update(overrides)
    return preflight(tmp_path, tmp_path / "photos.img", 30, **kwargs)


def test_preflight_accepts_a_plain_folder_with_room(tmp_path):
    assert _ok(tmp_path) == []


def test_preflight_refuses_an_existing_mount_and_an_fstab_entry(tmp_path):
    assert any("already a mount point" in p for p in _ok(tmp_path, is_mount=True))
    assert any("fstab" in p for p in _ok(tmp_path, fstab=f"x {tmp_path} ext4\n"))


def test_preflight_refuses_when_root_cannot_hold_the_image_beside_the_old_copy(tmp_path):
    assert any("GiB free" in p for p in _ok(tmp_path, root_free_bytes=31 * GIB))


def test_preflight_refuses_an_existing_image(tmp_path):
    (tmp_path / "photos.img").write_bytes(b"")
    assert any("already exists" in p for p in _ok(tmp_path))


def test_fstab_line_uses_nofail_skips_fsck_and_opts_out_of_fstrim():
    line = fstab_line(Path("/var/lib/pifilm/photos.img"), Path("/home/g/Pictures/pifilm"))
    assert line == ("/var/lib/pifilm/photos.img /home/g/Pictures/pifilm "
                    "ext4 loop,noatime,nofail,X-fstrim.notrim 0 0\n")


def test_plan_verifies_the_copy_before_moving_the_old_folder_and_mounts_last():
    mp = Path("/home/g/Pictures/pifilm")
    steps = plan("g", mp, Path("/var/lib/pifilm/photos.img"), 30)
    argvs = [s.argv for s in steps]
    verify = next(i for i, a in enumerate(argvs) if "--itemize-changes" in a)
    move = next(i for i, a in enumerate(argvs) if a[:1] == ("mv",))
    immutable = next(i for i, a in enumerate(argvs) if a[:1] == ("chattr",))
    assert verify < move < immutable
    assert argvs[move] == ("mv", str(mp), "/home/g/Pictures/pifilm.pre-volume")
    assert argvs[0][:2] == ("systemctl", "stop")
    assert argvs[-1][:2] == ("systemctl", "start")
    dropins = {s.path for s in steps if s.path and s.path.name == "photo-volume.conf"}
    assert dropins == {Path(f"/etc/systemd/system/{s}.d/photo-volume.conf") for s in SERVICES}
    assert all(s.text == dropin_text(mp) for s in steps if s.path in dropins)
    # Nothing in the plan deletes the old copy; that is left to the user.
    assert not any(a[:1] == ("rm",) for a in argvs)


def test_plan_formats_without_discard_so_the_reservation_survives():
    steps = plan("g", Path("/home/g/Pictures/pifilm"), Path("/var/lib/pifilm/photos.img"), 30)
    mkfs = next(s.argv for s in steps if s.argv[:1] == ("mkfs.ext4",))
    assert "nodiscard" in mkfs


def test_main_dry_run_changes_nothing_and_prints_the_plan(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("shutil.disk_usage", lambda _p: type("U", (), {"free": 40 * GIB})())
    folder = tmp_path / "pifilm"
    folder.mkdir()
    code = main(["--mount-point", str(folder), "--image", str(tmp_path / "p.img")])
    assert code == 0
    out = capsys.readouterr().out
    assert "Dry run" in out and "fallocate" in out
    assert not (tmp_path / "p.img").exists()
