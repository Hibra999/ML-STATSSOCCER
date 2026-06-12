from __future__ import annotations

import os


def configure_tensorflow_runtime() -> None:
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
