"""``pifilm-gpu-check``, the hardware acceptance tool, with a fake device."""

import dataclasses
import re
import subprocess
import sys

import numpy as np
import pytest

from pifilm.artifacts import write_artifact
from pifilm.gpu import check
from pifilm.halation import HalationParams
from pifilm.imageio import save_jpeg
from tests.gpu_fakes import TILE_GRAIN, FakeDevice, tile_artifacts


@pytest.fixture
def artifacts_dir(tmp_path):
    base = tile_artifacts()
    return write_artifact(tmp_path / "look", base.lut, base.normalize, TILE_GRAIN)


@pytest.fixture
def image(tmp_path):
    y, x = np.mgrid[0:120, 0:200].astype(np.float32)
    rgb = np.stack([40 + x, 30 + 1.6 * y, 220 - 0.5 * (x + y)], axis=-1)
    path = tmp_path / "frame.jpg"
    save_jpeg(np.clip(rgb, 0, 255).astype(np.uint8), path)
    return path


def _run(argv, **device_kwargs):
    lines: list[str] = []
    devices: list[FakeDevice] = []

    def factory():
        devices.append(FakeDevice(**device_kwargs))
        return devices[-1]

    code = check.main([str(a) for a in argv], out=lines.append, device_factory=factory)
    return code, "\n".join(lines), devices


def test_a_device_that_agrees_with_the_reference_passes(artifacts_dir, image):
    code, text, devices = _run(["--artifacts", artifacts_dir, "--image", image])
    assert code == 0
    assert "gpu grading on (Fake GPU, self-test passed)" in text
    assert "frame 200x120" in text
    runs = re.findall(r"run \d: total [\d.]+ ms \(upload 1.5, shader 2.5, read-back 3.5\)", text)
    assert len(runs) == 3
    assert "CPU pipeline" in text and "slow" in text
    assert re.search(r"against the reference: max 0 levels, mean 0.000; "
                     r"above 0: 0.000 %, above 1: 0.000 %, above 2: 0.000 %", text)
    assert re.search(r"against the CPU pipeline: max \d+ levels, mean \d\.\d{3}", text)
    assert text.rstrip().endswith("ok: the GPU is within 1 level of the reference")
    # The self-test, then the three runs, all with grain; and the device is released.
    assert [run["shape"] for run in devices[0].runs] == [(64, 96, 3)] + [(120, 200, 3)] * 3
    assert all(run["placement"] is not None for run in devices[0].runs)
    assert devices[0].closed == 1


def test_the_runs_option_sets_the_number_of_passes(artifacts_dir, image):
    code, text, devices = _run(["--artifacts", artifacts_dir, "--image", image, "--runs", "1"])
    assert code == 0
    assert len(devices[0].runs) == 2 and "run 1:" in text and "run 2:" not in text


def test_a_device_two_levels_off_fails(artifacts_dir, image):
    # Past the self-test (a driver can be right on the small frame and wrong on
    # a large one): this device goes wrong only after it.
    class LateFault(FakeDevice):
        def run(self, rgb_u8, **kwargs):
            self.off_by = 0 if rgb_u8.shape[:2] == (64, 96) else 2
            return super().run(rgb_u8, **kwargs)

    lines: list[str] = []
    code = check.main(["--artifacts", str(artifacts_dir), "--image", str(image)],
                      out=lines.append, device_factory=LateFault)
    text = "\n".join(lines)
    assert code == 1
    assert "against the reference: max 2 levels" in text
    assert "above 1: 100.000 %" in text
    assert text.rstrip().endswith("FAILED: the GPU is 2 levels from the reference (at most 1)")


def test_a_device_that_fails_its_self_test_is_no_gpu(artifacts_dir, image):
    code, text, devices = _run(["--artifacts", artifacts_dir, "--image", image], off_by=2)
    assert code == 1
    assert "gpu grading off: self-test:" in text
    assert "no GPU grading on this machine" in text
    assert devices[0].closed == 1


def test_no_gpu_says_so_and_fails(artifacts_dir, image):
    lines: list[str] = []
    code = check.main(["--artifacts", str(artifacts_dir), "--image", str(image)],
                      out=lines.append)
    text = "\n".join(lines)
    assert code == 1
    assert "gpu grading off: " in text and "no GPU grading on this machine" in text
    assert "run 1" not in text


def test_a_failed_pass_fails_the_check_and_releases_the_device(artifacts_dir, image):
    class SecondRunFails(FakeDevice):
        def run(self, rgb_u8, **kwargs):
            if len(self.runs) == 2:
                self.fail_next = 1
            return super().run(rgb_u8, **kwargs)

    made: list[FakeDevice] = []

    def factory():
        made.append(SecondRunFails())
        return made[-1]

    lines: list[str] = []
    code = check.main(["--artifacts", str(artifacts_dir), "--image", str(image)],
                      out=lines.append, device_factory=factory)
    assert code == 1
    assert any(line.startswith("FAILED: run 2: GL error after grading: 0x505") for line in lines)
    assert made[0].closed == 1


def test_a_frame_wider_than_the_largest_texture_is_reported(artifacts_dir, tmp_path):
    wide = tmp_path / "wide.jpg"
    save_jpeg(np.full((40, 300, 3), 90, np.uint8), wide)
    code, text, devices = _run(["--artifacts", artifacts_dir, "--image", wide], max_texture=256)
    assert code == 1
    assert "FAILED: the GPU cannot grade this frame" in text and "256" in text
    assert len(devices[0].runs) == 1 and devices[0].closed == 1


def test_an_artifact_without_grain_is_checked_without_it(tmp_path, image):
    base = tile_artifacts()
    grain = dataclasses.replace(TILE_GRAIN, enabled=False)
    directory = write_artifact(tmp_path / "plain", base.lut, base.normalize, grain)
    code, text, devices = _run(["--artifacts", directory, "--image", image])
    assert code == 0
    assert all(run["placement"] is None for run in devices[0].runs)


def test_a_halation_look_is_checked_with_its_glow(tmp_path, image):
    base = tile_artifacts()
    halation = HalationParams(enabled=True, strength=0.4)
    directory = write_artifact(tmp_path / "glow", base.lut, base.normalize, TILE_GRAIN,
                               halation=halation)
    code, text, devices = _run(["--artifacts", directory, "--image", image])
    assert code == 0
    assert re.search(r"halation on the CPU: [\d.]+ ms", text)
    assert re.search(r"against the reference: max 0 levels", text)
    assert all(run["halation"] is not None for run in devices[0].runs)

    # A device that leaves the halation out after its self-test is caught.
    class LateSkip(FakeDevice):
        def run(self, rgb_u8, **kwargs):
            self.skip_halation = rgb_u8.shape[:2] != (64, 96)
            return super().run(rgb_u8, **kwargs)

    lines: list[str] = []
    code = check.main(["--artifacts", str(directory), "--image", str(image)],
                      out=lines.append, device_factory=LateSkip)
    assert code == 1 and lines[-1].startswith("FAILED: the GPU is ")


def test_a_look_without_halation_prints_no_halation_line(artifacts_dir, image):
    code, text, _ = _run(["--artifacts", artifacts_dir, "--image", image])
    assert code == 0 and "halation" not in text


def test_the_synthetic_frame_is_the_imx477s_size():
    assert check.SYNTHETIC_SHAPE == (3040, 4056)
    frame = check._frame(None, shape=(30, 40))
    assert frame.shape == (30, 40, 3) and frame.dtype == np.uint8


def test_a_missing_artifact_or_image_is_an_error_line_not_a_traceback(tmp_path, artifacts_dir):
    lines: list[str] = []
    assert check.main(["--artifacts", str(tmp_path / "none")], out=lines.append,
                      device_factory=FakeDevice) == 1
    assert lines and lines[-1].startswith("error: ")
    lines.clear()
    assert check.main(["--artifacts", str(artifacts_dir), "--image", str(tmp_path / "no.jpg")],
                      out=lines.append, device_factory=FakeDevice) == 1
    assert lines[-1].startswith("error: ")


def test_runs_must_be_positive(artifacts_dir):
    with pytest.raises(SystemExit):
        check.main(["--artifacts", str(artifacts_dir), "--runs", "0"], out=lambda _line: None)


def test_the_command_is_installed(repo_root):
    import tomllib

    scripts = tomllib.loads((repo_root / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["pifilm-gpu-check"] == "pifilm.gpu.check:main"


def test_importing_the_command_loads_no_graphics_library():
    code = (
        "import ctypes\n"
        "def refuse(*a, **k): raise SystemExit('loaded ' + repr(a))\n"
        "ctypes.CDLL = refuse\n"
        "import pifilm.gpu.check\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
