"""Check release eligibility without uploading any artifacts."""

import pytest

from scripts.check_distribution import check_release_tag


def test_matching_stable_tag() -> None:
    check_release_tag("2.0.0", "v2.0.0")


@pytest.mark.parametrize(
    ("version", "tag"),
    [
        ("2.0.0.dev0", "v2.0.0.dev0"),
        ("2.0.0rc1", "v2.0.0rc1"),
        ("2.0.0+local", "v2.0.0+local"),
        ("2.0.0", "v2.0.1"),
        ("2.0.0", "2.0.0"),
    ],
)
def test_invalid_release_tag(version: str, tag: str) -> None:
    with pytest.raises(ValueError):
        check_release_tag(version, tag)
