"""Quality checks for the pedagogical text and version-specific guidance."""
from src.data import VERSIONS
from src.explanations import LAB_EXPERIMENTS, VERSION_GUIDE


def test_every_review_version_has_a_guided_reading():
    assert set(VERSION_GUIDE) == set(VERSIONS)
    for version in VERSIONS:
        guide = VERSION_GUIDE[version]
        assert all(guide[key].strip() for key in ("question", "observation", "try_this"))


def test_correction_explained_as_replacing_data_not_new_study():
    v4 = VERSION_GUIDE["V4"]["observation"].lower()
    assert "keine neue studie" in v4
    assert "ersetzt" in v4
    assert "achtwochenwert" in v4


def test_laboratory_has_three_concrete_exercises():
    assert len(LAB_EXPERIMENTS) == 3
    assert all(title.strip() and body.strip() for title, body in LAB_EXPERIMENTS)
