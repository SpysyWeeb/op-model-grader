"""Allow `python -m opgrader` (used by the start scripts)."""
import sys

from .cli import main

sys.exit(main())
