import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402

needs_artifacts = pytest.mark.skipif(
    not (C.SNAPSHOTS_PATH.exists() and C.MODEL_PATH.exists() and C.PREDICTIONS_PATH.exists()),
    reason="processed data / trained model not available; run the pipeline first")
