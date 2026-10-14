"""``pifilm-tuning``: the fingerprint of the libcamera tuning, and stamping a look with it.

A trained look carries the fingerprint of the tuning it was trained under
(``tuning_sha1`` in ``params.json``), and the capture service compares it with
the tuning the camera runs with (``pifilm.tuning``). This prints that
fingerprint, resolved exactly as the camera resolves it, and with ``--stamp``
writes it into an artifact:

    pifilm-tuning --tuning-file imx477.json --stamp ~/pifilm/artifacts/imx477-v4

It does not open the camera, because the capture service usually holds it, so
the automatic case (no ``--tuning-file``) cannot ask the sensor for its model;
``--model`` names it instead (default ``imx477``). ``--sha1 HEX`` skips the
resolving altogether, so a look can be stamped on the Mac, where there is no
Picamera2. Picamera2 is imported only when a tuning is resolved.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Mapping

from ..artifacts import ArtifactsError, stamp_tuning, validate_tuning_sha1
from ..tuning import resolve


def _sha1_arg(text: str) -> str:
    try:
        return validate_tuning_sha1(text, "--sha1")
    except ArtifactsError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pifilm-tuning",
        description="Print the fingerprint of the libcamera tuning the camera would use, "
                    "and optionally stamp it into a look. The camera is not opened (the "
                    "capture service usually holds it), so without --tuning-file the "
                    "sensor model comes from --model.")
    parser.add_argument("--tuning-file", metavar="NAME",
                        help="the tuning file the service runs with (its --tuning-file), "
                             "a name Picamera2 finds or a path")
    parser.add_argument("--model", default="imx477",
                        help="sensor model for the automatic case, without --tuning-file "
                             "(default: imx477); the camera is not asked")
    parser.add_argument("--sha1", type=_sha1_arg, metavar="HEX",
                        help="use this fingerprint instead of resolving one (no Picamera2 "
                             "needed, so it works on the Mac)")
    parser.add_argument("--stamp", metavar="DIR",
                        help="also write the fingerprint into the artifact at DIR")
    return parser


def _picamera2_loader() -> Callable[[str], dict]:
    from picamera2 import Picamera2

    return Picamera2.load_tuning_file


def main(argv: list[str] | None = None, *, out: Callable[[str], None] = print,
         load: Callable[[str], dict] | None = None,
         environ: Mapping[str, str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sha1 is not None:
        sha1, source = args.sha1, "--sha1"
    else:
        if load is None:
            try:
                load = _picamera2_loader()
            except (ImportError, OSError) as exc:
                out(f"error: Picamera2 is unavailable ({exc}); pass --sha1 HEX to stamp "
                    "without it")
                return 1
        try:
            resolved = resolve(args.tuning_file, args.model, load=load,
                               environ=os.environ if environ is None else environ)
        except Exception as exc:  # a given tuning file that will not load
            out(f"error: cannot load tuning file {args.tuning_file!r}: {exc}")
            return 1
        if resolved.sha1 is None:
            out(f"error: cannot fingerprint the tuning in use ({resolved.reason})")
            return 1
        sha1, source = resolved.sha1, resolved.source
    out(f"tuning {source}: {sha1}")
    if args.stamp is not None:
        try:
            params = stamp_tuning(args.stamp, sha1)
        except (ArtifactsError, OSError) as exc:
            out(f"error: {exc}")
            return 1
        out(f"stamped {params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
