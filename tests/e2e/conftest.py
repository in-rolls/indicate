"""Build the wheel once, install it once, share both across the e2e tests."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def pytest_collection_modifyitems(config, items):
    """Mark everything below this directory as e2e.

    The path check is not decoration. A conftest in a subdirectory still
    receives the **whole** session's collected items, so the obvious loop marks
    every test in the repo as e2e -- which made ``-m e2e`` run 457 tests instead
    of the 20 that build and install a wheel.
    """
    here = Path(__file__).resolve().parent
    for item in items:
        if here in Path(str(item.fspath)).resolve().parents:
            item.add_marker(pytest.mark.e2e)


def _uv() -> str:
    """Return the uv executable, skipping the session if it is absent."""
    found = shutil.which("uv")
    if not found:
        pytest.skip("uv is not on PATH; e2e tests build and install with it")
    return found


@pytest.fixture(scope="session")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the wheel once per session and return its path.

    Args:
        tmp_path_factory: pytest's session-scoped temp directory factory.

    Returns:
        Path to the built ``.whl``.
    """
    out = tmp_path_factory.mktemp("dist")
    proc = subprocess.run(  # noqa: S603
        [_uv(), "build", "--wheel", "--out-dir", str(out)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        # fail, not skip: a wheel that will not build is the exact packaging
        # regression this suite exists to catch, and skipping turns the whole
        # dependent suite green. Only a missing `uv` is environmental.
        pytest.fail(f"uv build --wheel failed:\n{proc.stderr}")
    wheels = list(out.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, got {wheels}"
    return wheels[0]


@pytest.fixture(scope="session")
def built_sdist(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the sdist once per session and return its path.

    Args:
        tmp_path_factory: pytest's session-scoped temp directory factory.

    Returns:
        Path to the built ``.tar.gz``.
    """
    out = tmp_path_factory.mktemp("sdist")
    proc = subprocess.run(  # noqa: S603
        [_uv(), "build", "--sdist", "--out-dir", str(out)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.fail(f"uv build --sdist failed:\n{proc.stderr}")
    archives = list(out.glob("*.tar.gz"))
    assert len(archives) == 1, f"expected one sdist, got {archives}"
    return archives[0]


@pytest.fixture(scope="session")
def wheel_venv(built_wheel: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Install the wheel into a clean venv with only ``click``.

    Deliberately ``--no-deps`` plus click: the package's import path must not
    need torch, and installing the real dependency tree would pull ~2 GB and
    turn a packaging test into a download test.

    Args:
        built_wheel: The wheel to install.
        tmp_path_factory: pytest's session-scoped temp directory factory.

    Returns:
        The virtual environment root.
    """
    uv = _uv()
    venv = tmp_path_factory.mktemp("venv") / "env"
    subprocess.run(  # noqa: S603
        [uv, "venv", str(venv)], capture_output=True, text=True, check=True
    )
    proc = subprocess.run(  # noqa: S603
        [
            uv,
            "pip",
            "install",
            "--python",
            str(venv),
            "--no-deps",
            str(built_wheel),
            "click",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.fail(f"could not install the built wheel:\n{proc.stderr}")
    return venv


def venv_python(venv: Path) -> Path:
    """Return the interpreter inside a virtual environment.

    Args:
        venv: The environment root.

    Returns:
        Path to ``python``.
    """
    scripts = "Scripts" if sys.platform == "win32" else "bin"
    suffix = ".exe" if sys.platform == "win32" else ""
    return venv / scripts / f"python{suffix}"


def venv_script(venv: Path, name: str) -> Path:
    """Return a console script inside a virtual environment.

    Resolved from the environment rather than ``shutil.which(name)`` against the
    ambient ``PATH``, which would find whatever the developer has installed
    globally and test that instead.

    Args:
        venv: The environment root.
        name: Script name, without extension.

    Returns:
        Path to the script.
    """
    bindir = Path(sysconfig.get_path("scripts", vars={"base": str(venv)}))
    suffix = ".exe" if sys.platform == "win32" else ""
    return bindir / f"{name}{suffix}"


@pytest.fixture
def clean_env(tmp_path: Path) -> dict[str, str]:
    """An environment with no PYTHONPATH and an empty, offline HF cache.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        A full environment mapping for a subprocess.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["HF_HOME"] = str(tmp_path / "hf")
    env["HF_HUB_OFFLINE"] = "1"
    return env
