from pathlib import Path

import pifilm
from pifilm.artifacts import Artifacts

LOOKS = Path(pifilm.__file__).parent / "data" / "looks"


def test_kodachrome_look_loads_with_its_original_lut_hash():
    """Copied from kodachrome-film unchanged except for the cube's file name,
    so its hash must still be the one that repository trained and gated."""
    look = Artifacts.load(LOOKS / "kodachrome-k14")
    assert look.lut_sha1 == "b8ccf30cc719241c98ff3c4a9f73a1cdf8cb000b"
    assert (LOOKS / "kodachrome-k14" / "pifilm.cube").is_file()
