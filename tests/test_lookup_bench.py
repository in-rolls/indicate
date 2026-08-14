"""Performance floors for the lookup path.

The point of the table is speed, so a change that quietly gives the speed back
is a regression even when every accuracy test still passes. These assert the
numbers ``training/bench_lookup.py`` reports, at thresholds set well below the
measured values so ordinary machine-to-machine variation does not fail CI.

Measured on an M-series laptop, Punjabi table, warm filesystem cache:

===========================  =========  =====================
measurement                  measured   floor asserted here
===========================  =========  =====================
table load                   0.084 s    < 0.50 s
resident memory              52 MB      < 128 MB
lookups/s (zipf, cached)     16.9 M     > 3 M
lookups/s (all keys distinct) 833 k     > 200 k
cold start, all-hit input    0.10 s     < 1.0 s, no torch
cold start, model-only       0.44 s     (reference)
===========================  =========  =====================

The cold-start test is the load-bearing one: it runs a fresh interpreter and
asserts torch was never imported. Reintroducing a module-scope ``import torch``
anywhere on the import path of ``indicate.punjabi2english`` fails it, and that
single line is worth 4x on per-invocation latency.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

import pytest

from indicate.languages import PAIRS
from indicate.lookup import Lookup, clear_cache

SUBDIR = PAIRS[("punjabi", "english")].subdir
SINGH = "ਸਿੰਘ"

MAX_LOAD_S = 0.5
MAX_RSS_MB = 128.0
MIN_ZIPF_PER_S = 3_000_000
MIN_DISTINCT_PER_S = 200_000
MAX_COLD_START_S = 1.0


def _table_available() -> bool:
    return Lookup.load(SUBDIR) is not None


def _run(body: str) -> str:
    """Run a snippet in a clean interpreter and return its stdout.

    A subprocess, not this process: pytest runs under coverage, whose tracer
    costs roughly 6x on a tight loop. A throughput floor measured through it
    would be measuring coverage.

    Args:
        body: Python source to execute from the repository root.

    Returns:
        Whatever the snippet printed, stripped.
    """
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", body],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stderr)
    return proc.stdout.strip()


def _rss_mb() -> float:
    """Current resident set size in MB, or 0.0 when ``ps`` is unavailable."""
    try:
        out = subprocess.run(  # noqa: S603
            ["/bin/ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(out.stdout.strip()) / 1024
    except Exception:
        return 0.0


@unittest.skipUnless(_table_available(), "punjabi lookup table not built")
class TestLookupPerformance(unittest.TestCase):
    def test_table_loads_quickly_and_cheaply(self):
        clear_cache()
        before = _rss_mb()
        start = time.perf_counter()
        table = Lookup.load(SUBDIR)
        seconds = time.perf_counter() - start
        assert table is not None
        self.assertLess(seconds, MAX_LOAD_S, f"table load took {seconds:.2f}s")
        if before:
            grew = _rss_mb() - before
            self.assertLess(grew, MAX_RSS_MB, f"table cost {grew:.0f} MB")

    def test_throughput_under_repeated_access(self):
        rate = float(
            _run(
                "import time\n"
                "from indicate.lookup import Lookup\n"
                f"t = Lookup.load({SUBDIR!r})\n"
                f"words = [{SINGH!r}] * 200_000\n"
                "s = time.perf_counter()\n"
                "for w in words: t.get(w)\n"
                "print(len(words) / (time.perf_counter() - s))\n"
            )
        )
        self.assertGreater(rate, MIN_ZIPF_PER_S, f"{rate:,.0f} lookups/s")

    def test_throughput_with_no_repeats_at_all(self):
        # Worst case: the key cache can never help, so every call pays full
        # normalization. Guards against normalize.py growing an expensive step.
        rate = float(
            _run(
                "import time\n"
                "from indicate.lookup import Lookup\n"
                f"t = Lookup.load({SUBDIR!r})\n"
                "keys = list(t._table)[:50_000]\n"
                "s = time.perf_counter()\n"
                "for k in keys: t.get(k)\n"
                "print(len(keys) / (time.perf_counter() - s))\n"
            )
        )
        self.assertGreater(rate, MIN_DISTINCT_PER_S, f"{rate:,.0f} lookups/s")


class TestColdStart(unittest.TestCase):
    @pytest.mark.needs_lookup
    def test_an_all_hit_input_starts_fast_and_never_imports_torch(self):
        script = (
            "import sys, time\n"
            "t = time.perf_counter()\n"
            "import indicate\n"
            f"out = indicate.transliterate({SINGH!r}, source='punjabi')\n"
            "print(time.perf_counter() - t, 'torch' in sys.modules, out)\n"
        )
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        seconds, torch_imported, out = proc.stdout.split()
        if not _table_available():
            self.skipTest("punjabi lookup table not built")
        self.assertEqual(out, "singh")
        self.assertEqual(
            torch_imported,
            "False",
            "torch was imported for an input that never reaches the decoder; "
            "something on the import path of `import indicate` imports it at "
            "module scope",
        )
        self.assertLess(float(seconds), MAX_COLD_START_S)

    def test_a_chain_without_the_model_backend_never_imports_torch(self):
        # Even on an input that misses the table: with no model in the chain,
        # nothing should reach the decoder, so torch has no reason to load.
        # Deliberately artifact-free: with no table the chain raises, and the
        # property under test -- that a chain without the model backend never
        # loads torch -- must hold either way. That is what lets this one test
        # run on a fresh clone, where it is needed most.
        script = (
            "import sys\n"
            "import indicate\n"
            "try:\n"
            "    indicate.transliterate('ZZZQQ ' + "
            f"{SINGH!r}, source='punjabi', engine=['lookup'])\n"
            "except indicate.BackendsUnavailableError:\n"
            "    pass\n"
            "print('torch' in sys.modules, 'litellm' in sys.modules)\n"
        )
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.split(), ["False", "False"])


if __name__ == "__main__":
    unittest.main()
