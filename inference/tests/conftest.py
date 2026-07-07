"""Test bootstrap: force the demo plane and put the inference package on sys.path.

`config` reads ``STUDIO_DEMO`` at import time, so the env var must be set before
any test imports ``config``/``pipelines``. The path insert lets the flat inference
modules (``import config``) resolve when pytest is run from the repo root.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("STUDIO_DEMO", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
