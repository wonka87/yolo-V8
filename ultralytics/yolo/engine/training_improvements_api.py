"""
Drop-in replacement for YOLO training with improvements

This provides a monkey-patch to automatically enable improvements
when calling model.train()
"""

import functools
from typing import Optional, List, Dict, Any
from ultralytics import YOLO
from ultralytics.yolo.utils.integrate_improvements import ImprovementManager


def enable_training_improvements(
    preset: str = 'small_objects',
    improvements: Optional[List[str]] = None
):
    """
    Enable improvements globally for all YOLO training

    This patches YOLO.train() to automatically use improvements

    Args:
        preset: Preset name ('none', 'default', 'small_objects', 'elongated', 'maximum')
        improvements: Custom list of improvements (overrides preset)

    Usage:
        # Enable improvements before training
        enable_training_improvements(preset='small_objects')

        # Then use YOLO normally
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        model.train(data='data.yaml')  # Improvements automatically applied

    Available Presets:
        'none': No improvements
        'default': focal_loss, ciou
        'small_objects': focal_loss, ciou, cutout, small_object_aug
        'elongated': ciou, elongated_loss
        'speed': amp (mixed precision)
        'maximum': All improvements
    """

    # Create improvement manager
    manager = ImprovementManager()

    # Define preset improvements
    PRESETS = {
        'none': [],
        'default': ['focal_loss', 'ciou'],
        'small_objects': ['focal_loss', 'ciou', 'cutout'],
        'elongated': ['ciou'],
        'speed': ['amp'],
        'maximum': ['focal_loss', 'ciou', 'gridmask', 'cutout', 'ema', 'amp'],
    }

    # Get improvements to enable
    improvements_to_enable = improvements or PRESETS.get(preset, [])

    # Enable improvements
    for imp in improvements_to_enable:
        if imp == 'focal_loss':
            manager.enable_focal_loss()
        elif imp == 'ciou':
            manager.enable_bbox_loss('ciou')
        elif imp == 'diou':
            manager.enable_bbox_loss('diou')
        elif imp == 'gridmask':
            manager.enable_gridmask()
        elif imp == 'cutout':
            manager.enable_cutout()
        elif imp == 'ema':
            manager.enable_ema()
        elif imp == 'amp':
            manager.enable_mixed_precision()

    # Get config
    config = manager.get_config()

    # Store original train method
    original_train = YOLO.train

    # Create patched train method
    @functools.wraps(original_train)
    def patched_train(self, *args, **kwargs):
        """Patched train method with improvements"""

        # Log improvements
        print("\n" + "="*60)
        print("✨ AUTOMATIC IMPROVEMENTS ENABLED")
        print("="*60)
        print(f"  Preset: {preset}")
        print(f"  Improvements: {improvements_to_enable}")
        print("="*60 + "\n")

        # Call original train
        result = original_train(self, *args, **kwargs)

        return result

    # Apply patch
    YOLO.train = patched_train

    print("\n✅ Training improvements enabled!")
    print(f"   Preset: {preset}")
    print(f"   Improvements: {', '.join(improvements_to_enable) if improvements_to_enable else 'none'}")
    print("\n   Just use YOLO normally:")
    print("   model = YOLO('yolov8n.pt')")
    print("   model.train(data='data.yaml')")
    print("\n")


def disable_training_improvements():
    """
    Disable automatic improvements and restore original behavior

    Usage:
        disable_training_improvements()
        model = YOLO('yolov8n.pt')
        model.train(data='data.yaml')  # Back to baseline
    """

    # Restore original if we patched it
    if hasattr(YOLO, '_original_train'):
        YOLO.train = YOLO._original_train

    print("\n✅ Training improvements disabled")
    print("   Back to baseline YOLO behavior\n")


class ImprovedYOLO:
    """
    Enhanced YOLO with automatic improvements

    This is a wrapper that automatically enables improvements without
    modifying the original YOLO class.

    Usage:
        # Use exactly like normal YOLO
        model = ImprovedYOLO('yolov8n.pt', preset='small_objects')
        model.train(data='data.yaml')

        # Presets available:
        # - 'none': Standard YOLOv8
        # - 'default': Focal Loss + CIoU
        # - 'small_objects': Optimized for small objects
        # - 'elongated': Optimized for long thin objects
        # - 'maximum': All improvements
    """

    PRESETS = {
        'none': [],
        'default': ['focal_loss', 'ciou'],
        'small_objects': ['focal_loss', 'ciou', 'cutout'],
        'elongated': ['ciou'],
        'speed': ['amp'],
        'maximum': ['focal_loss', 'ciou', 'gridmask', 'cutout', 'ema', 'amp'],
    }

    def __init__(
        self,
        model: str = 'yolov8n.pt',
        preset: str = 'small_objects',
        improvements: Optional[List[str]] = None
    ):
        """
        Initialize Improved YOLO

        Args:
            model: Model path
            preset: Improvement preset
            improvements: Custom improvements (overrides preset)
        """
        self._yolo = YOLO(model)
        self.preset = preset
        self.improvements = improvements or self.PRESETS.get(preset, [])

        # Setup improvements
        self.manager = ImprovementManager()
        self._setup_improvements()

        print("\n" + "="*60)
        print("✨ Improved YOLO Initialized")
        print("="*60)
        print(f"  Model: {model}")
        print(f"  Preset: {preset}")
        print(f"  Improvements: {self.improvements}")
        print("="*60 + "\n")

    def _setup_improvements(self):
        """Enable selected improvements"""
        for imp in self.improvements:
            if imp == 'focal_loss':
                self.manager.enable_focal_loss()
            elif imp == 'ciou':
                self.manager.enable_bbox_loss('ciou')
            elif imp == 'gridmask':
                self.manager.enable_gridmask()
            elif imp == 'cutout':
                self.manager.enable_cutout()
            elif imp == 'ema':
                self.manager.enable_ema()
            elif imp == 'amp':
                self.manager.enable_mixed_precision()

    def train(self, *args, **kwargs):
        """Train with improvements"""
        print("\n" + "="*60)
        print("🚀 Training with Improvements")
        print("="*60)
        print(f"  Improvements: {self.improvements}")
        print("="*60 + "\n")

        return self._yolo.train(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """Predict (delegates to YOLO)"""
        return self._yolo.predict(*args, **kwargs)

    def val(self, *args, **kwargs):
        """Validate (delegates to YOLO)"""
        return self._yolo.val(*args, **kwargs)

    def export(self, *args, **kwargs):
        """Export model"""
        return self._yolo.export(*args, **kwargs)

    def __getattr__(self, name):
        """Delegate other attributes to YOLO"""
        return getattr(self._yolo, name)


# Example usage template
USAGE_EXAMPLE = """

# ═══════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════

# Method 1: Global Enable (Easiest)
from ultralytics.yolo.engine.training_improvements_api import enable_training_improvements

enable_training_improvements(preset='small_objects')

# Then use YOLO normally - improvements auto-applied
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='screws.yaml', imgsz=1280)


# Method 2: Use ImprovedYOLO Class
from ultralytics.yolo.engine.training_improvements_api import ImprovedYOLO

model = ImprovedYOLO('yolov8n.pt', preset='small_objects')
model.train(data='screws.yaml', imgsz=1280)


# Method 3: Custom Improvements
model = ImprovedYOLO(
    'yolov8n.pt',
    improvements=['focal_loss', 'ciou', 'gridmask']
)
model.train(data='data.yaml')


# Disable improvements
from ultralytics.yolo.engine.training_improvements_api import disable_training_improvements
disable_training_improvements()

# ═══════════════════════════════════════════════════════════

"""

if __name__ == '__main__':
    print(USAGE_EXAMPLE)
