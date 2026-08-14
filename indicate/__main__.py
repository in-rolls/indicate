"""Make ``python -m indicate`` equivalent to the ``indicate`` console script.

Worth the four lines: the console script only exists once the package is
installed with its entry points, so ``uv run python -m indicate`` and a checkout
without an active venv had no way to reach the CLI at all.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    main()
