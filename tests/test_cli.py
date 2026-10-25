# Ultralytics YOLO 🚀, GPL-3.0 license

import subprocess
from pathlib import Path

from ultralytics.yolo.utils import ROOT, SETTINGS

MODEL = Path(SETTINGS['weights_dir']) / 'yolov8n'
CFG = 'yolov8n'


def run(cmd):
    # Run a subprocess command with check=True
    subprocess.run(cmd.split(), check=True)


def test_special_modes():
    run('yolo checks')
    run('yolo settings')
    run('yolo help')


# Train checks ---------------------------------------------------------------------------------------------------------
def test_train_det():
    run(f'yolo train detect model={CFG}.yaml data=coco8.yaml imgsz=32 epochs=1')


# Val checks -----------------------------------------------------------------------------------------------------------
def test_val_detect():
    run(f'yolo val detect model={MODEL}.pt data=coco8.yaml imgsz=32 epochs=1')


# Predict checks -------------------------------------------------------------------------------------------------------
def test_predict_detect():
    run(f"yolo predict detect model={MODEL}.pt source={ROOT / 'assets'}")


# Export checks --------------------------------------------------------------------------------------------------------
def test_export_detect_torchscript():
    run(f'yolo export model={MODEL}.pt format=torchscript')
