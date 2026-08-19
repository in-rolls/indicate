"""What may and may not ship.

The exclusions here are a **license gate**, not a size optimization. Lookup
tables derive from ``data/hindi.csv.gz`` (which blends CC-BY-NC IIT Bombay
pairs) and ``data/punjabi.csv.gz`` (from the IRB-restricted electoral-roll
deposit); neither is redistributable in an MIT wheel. The exclusion once
rested on one line of ``.gitignore``, which the old hatchling backend happened
to honour; it now lives in ``[tool.uv.build-backend]`` excludes. These tests
assert it directly, so a build-config change that would ship restricted data
fails a test rather than reaching PyPI.
"""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import pytest

#: Nothing matching these may appear in a distribution. Matched on data
#: extensions rather than on the word "lookup", which would also catch
#: ``indicate/lookup.py`` -- the module ships, the table it reads does not.
FORBIDDEN = ("saved_weights/", ".safetensors", ".tsv.gz", ".csv.gz", ".parquet")

#: Directories that are development-only.
FORBIDDEN_DIRS = ("tests/", "gazetteer/", "training/", "notebooks/", "examples/")

#: These must ship, or the package cannot tokenize anything.
REQUIRED = (
    "indicate/data/hindi_to_english/hindi_tokens.json",
    "indicate/data/hindi_to_english/english_tokens.json",
    "indicate/data/punjabi_to_english/punjabi_tokens.json",
    "indicate/data/punjabi_to_english/english_tokens.json",
    "indicate/data/llm_examples.json",
    "indicate/py.typed",
)

#: A wheel this size means weights or tables leaked back in. Failing on size
#: catches it before anyone has to read a file listing.
MAX_WHEEL_BYTES = 2_000_000


def _wheel_names(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as archive:
        return archive.namelist()


class TestWheelContents:
    def test_no_restricted_data_ships(self, built_wheel: Path):
        offenders = [
            name
            for name in _wheel_names(built_wheel)
            if any(token in name for token in FORBIDDEN)
        ]
        assert offenders == [], (
            f"the wheel contains non-redistributable data: {offenders}. "
            "Lookup tables and weights must stay out; see the exclude lists in "
            "pyproject.toml."
        )

    def test_no_development_directories_ship(self, built_wheel: Path):
        offenders = [
            name
            for name in _wheel_names(built_wheel)
            if any(name.startswith(d) or f"/{d}" in name for d in FORBIDDEN_DIRS)
        ]
        assert offenders == []

    def test_the_tokenizers_and_examples_do_ship(self, built_wheel: Path):
        names = set(_wheel_names(built_wheel))
        for required in REQUIRED:
            assert required in names, f"{required} is missing from the wheel"

    def test_every_package_module_ships(self, built_wheel: Path):
        names = set(_wheel_names(built_wheel))
        for module in ("api", "engine", "languages", "lookup", "cli", "transliterator"):
            assert f"indicate/{module}.py" in names

    def test_the_console_script_is_registered(self, built_wheel: Path):
        with zipfile.ZipFile(built_wheel) as archive:
            entry = next(
                n for n in archive.namelist() if n.endswith("entry_points.txt")
            )
            text = archive.read(entry).decode()
        assert "indicate = indicate.cli:main" in text

    def test_the_wheel_stays_small(self, built_wheel: Path):
        size = built_wheel.stat().st_size
        assert size < MAX_WHEEL_BYTES, (
            f"wheel is {size / 1e6:.1f} MB; weights or lookup tables have "
            "probably leaked back in"
        )


class TestSdistContents:
    def test_no_restricted_data_ships(self, built_sdist: Path):
        with tarfile.open(built_sdist) as archive:
            names = archive.getnames()
        offenders = [
            name for name in names if any(token in name for token in FORBIDDEN)
        ]
        assert offenders == [], f"the sdist contains restricted data: {offenders}"

    def test_the_training_corpora_do_not_ship(self, built_sdist: Path):
        # data/ is excluded wholesale; these are the two files that matter.
        with tarfile.open(built_sdist) as archive:
            names = archive.getnames()
        assert not [n for n in names if n.endswith(("hindi.csv.gz", "punjabi.csv.gz"))]


@pytest.mark.usefixtures("built_wheel")
class TestTheGateIsReal:
    def test_pyproject_states_both_exclusions(self):
        # The rule must be in the build config, not inherited from .gitignore,
        # which is one edit away from silently including 6 MB of restricted data.
        text = Path("pyproject.toml").read_text(encoding="utf-8")
        backend_section = text.split("[tool.uv.build-backend]")[1]
        for key in ("source-exclude", "wheel-exclude"):
            line = next(ln for ln in backend_section.splitlines() if ln.startswith(key))
            assert "saved_weights" in line
            assert "lookup*.tsv.gz" in line
