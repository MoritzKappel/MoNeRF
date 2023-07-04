# -- coding: utf-8 --

"""projectpath.py: Helper class that sets up the project's directory structure for imports."""

import sys
from pathlib import Path


class context:
    """A context class adding the source codde location to the current python path."""

    def __enter__(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

    def __exit__(self, *_):
        sys.path.pop(0)
