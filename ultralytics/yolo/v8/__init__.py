# Ultralytics YOLO 🚀, GPL-3.0 license

from ultralytics.yolo.configs import hydra_patch  # noqa (patch hydra CLI)
from ultralytics.yolo.v8 import detect

__all__ = ["detect"]
