"""The API reference must list every module, and only modules that exist.

``docs/api.rst`` names each module in an explicit ``automodule`` block, and
nothing else connects that list to the package. The two ways it can drift are
not equally visible:

* a **stale** entry is loud -- autodoc cannot import a module that was deleted,
  and ``sphinx-build -W`` turns that into a failed build;
* a **missing** entry is silent. Nothing references the new module, so nothing
  complains, and it is simply absent from the published reference.

That asymmetry is what this file exists for. It has already bitten twice in one
day: ``api.rst`` pointed at the deleted ``indicate.hindi2english`` and
``indicate.punjabi2english`` until someone noticed by eye, and
``indicate.logging`` was missing from it entirely.

Deliberately a string comparison over two directory listings rather than a
Sphinx build: it needs no artifacts, no network and no docs dependency group,
so it runs in the default suite on a fresh clone and fails in milliseconds
rather than at the end of a docs build.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "indicate"
API_RST = REPO_ROOT / "docs" / "api.rst"

#: Modules deliberately left out of the reference, each with a reason. Keep
#: this empty if you can: an entry here is a promise that the module is genuinely
#: not part of the documented surface, not a way to silence this test.
UNDOCUMENTED: dict[str, str] = {}

_AUTOMODULE = re.compile(r"^\.\.\s+automodule::\s+indicate\.(\w+)\s*$", re.M)


def _package_modules() -> set[str]:
    """Return every importable module name directly under ``indicate/``."""
    return {
        path.stem for path in PACKAGE.glob("*.py") if not path.stem.startswith("__")
    }


def _documented_modules() -> set[str]:
    """Return every ``indicate.X`` named by an automodule block in api.rst."""
    return set(_AUTOMODULE.findall(API_RST.read_text(encoding="utf-8")))


def test_every_module_appears_in_the_api_reference():
    missing = _package_modules() - _documented_modules() - set(UNDOCUMENTED)
    assert not missing, (
        f"{sorted(missing)} exist in indicate/ but have no `.. automodule::` "
        f"block in docs/api.rst, so they are absent from the published "
        f"reference and nothing else would have said so. Add a section, or "
        f"add the module to UNDOCUMENTED with a reason."
    )


def test_the_api_reference_names_no_module_that_was_deleted():
    stale = _documented_modules() - _package_modules()
    assert not stale, (
        f"docs/api.rst documents {sorted(stale)}, which no longer exist in "
        f"indicate/. sphinx-build -W also catches this, but as an autodoc "
        f"import traceback rather than by name."
    )


def test_exclusions_are_real_modules_with_stated_reasons():
    # An exclusion for a module that no longer exists is dead weight that
    # silently widens what the test above will tolerate.
    unknown = set(UNDOCUMENTED) - _package_modules()
    assert not unknown, f"UNDOCUMENTED names non-existent modules: {sorted(unknown)}"
    assert all(reason.strip() for reason in UNDOCUMENTED.values()), (
        "every UNDOCUMENTED entry needs a reason"
    )
