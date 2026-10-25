# Ultralytics YOLO 🚀, GPL-3.0 license

from pathlib import Path

from ultralytics.yolo.configs import get_config
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, SETTINGS
from ultralytics.yolo.v8 import detect

CFG_DET = 'yolov8n.yaml'
CFG = get_config(DEFAULT_CONFIG)
MODEL = Path(SETTINGS['weights_dir']) / 'yolov8n'
SOURCE = ROOT / "assets"


def test_detect():
    overrides = {"data": "coco8.yaml", "model": CFG_DET, "imgsz": 32, "epochs": 1, "save": False}
    CFG.data = "coco8.yaml"

    # Trainer
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.train()

    # Validator
    val = detect.DetectionValidator(args=CFG)
    val(model=trainer.best)  # validate best.pt

    # Predictor
    pred = detect.DetectionPredictor(overrides={"imgsz": [64, 64]})
    result = pred(source=SOURCE, model=f"{MODEL}.pt")
    assert len(result), "predictor test failed"

    overrides["resume"] = trainer.last
    trainer = detect.DetectionTrainer(overrides=overrides)
    try:
        trainer.train()
    except Exception as e:
        print(f"Expected exception caught: {e}")
        return

    Exception("Resume test failed!")

