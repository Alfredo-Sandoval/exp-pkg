"""Module entry point for ``python -m posetta``."""

from __future__ import annotations

import sys

from posetta.cli import main

if __name__ == "__main__":
    sys.exit(main())
