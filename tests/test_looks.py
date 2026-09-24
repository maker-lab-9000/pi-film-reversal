from pifilm.artifacts import Artifacts

KODACHROME_SHA1 = "b8ccf30cc719241c98ff3c4a9f73a1cdf8cb000b"


def test_the_bundled_look_is_the_kodachrome_experiment():
    """exp/kodachrome-look swaps kodachrome-film's trained look into pifilm/data,
    so a pull and a restart are enough on the Pi. Copied unchanged except for the
    cube's file name, so its hash must be the one that repository trained."""
    assert Artifacts.default().lut_sha1 == KODACHROME_SHA1


def test_the_starter_stays_beside_it_for_rollback():
    starter = Artifacts.load(Artifacts.default().path / "looks" / "starter")
    assert starter.lut_sha1 != KODACHROME_SHA1
