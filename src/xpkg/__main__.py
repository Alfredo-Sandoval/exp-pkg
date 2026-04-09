"""Module entry point for ``python -m xpkg``."""

from __future__ import annotations

import sys

from xpkg.cli import main

if __name__ == "__main__":
    sys.exit(main())
