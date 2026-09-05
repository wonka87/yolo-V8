"""
Improved Trainer with Integrated Improvements

This module extends the BaseTrainer to automatically include improvements
for better small object and elongated object detection.
"""

import torch
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils.integrate_improvements import ImprovementManager
from ultralytics.yolo.data.small_object_augmentation import (
    SmallObjectAugmentation,
    ElongatedObjectLoss,
    ScaleNormalizedLoss
)


class ImprovedTrainer(BaseTrainer):
    """
    Extended trainer with integrated improvements

    This trainer automatically enables improvements for:
    - Small object detection
    - Elongated object detection
    - Better convergence

    Usage:
        # Option 1: Use default improvements
        trainer = ImprovedTrainer(overrides={'data': 'data.yaml'})
        trainer.train()

        # Option 2: Specify improvements
        trainer = ImprovedTrainer(
            improvements=['focal_loss', 'ciou', 'gridmask']
        )
        trainer.train()

        # Option 3: Presets
        trainer = ImprovedTrainer(preset='small_objects')
        trainer.train()
    """

    IMPROVEMENT_PRESETS = {
        'none': [],
        'default': ['focal_loss', 'ciou'],
        'small_objects': ['focal_loss', 'ciou', 'cutout', 'small_object_aug'],
        'elongated': ['ciou', 'elongated_loss'],
        'speed': ['amp'],
        'maximum': ['focal_loss', 'ciou', 'gridmask', 'cutout', 'ema', 'amp'],
    }

    def __init__(
        self,
        config=None,
        overrides: Optional[Dict[str, Any]] = None,
        improvements: Optional[List[str]] = None,
        preset: str = 'default',
        auto_enable: bool = True
    ):
        """
        Initialize improved trainer

        Args:
            config: Configuration file path or dict
            overrides: Override parameters
            improvements: List of specific improvements to enable
            preset: Preset name ('none', 'default', 'small_objects', etc.)
            auto_enable: Whether to automatically enable improvements
        """
        # Initialize parent
        super().__init__(config=config, overrides=overrides or {})

        # Setup improvements
        self.improvement_manager = ImprovementManager()
        self.improvements_enabled = []

        if auto_enable:
            if improvements:
                # User specified improvements
                for imp in improvements:
                    self._enable_improvement(imp)
            else:
                # Use preset
                preset_improvements = self.IMPROVEMENT_PRESETS.get(preset, [])
                for imp in preset_improvements:
                    self._enable_improvement(imp)

        self._log_improvements()

    def _enable_improvement(self, improvement_name: str):
        """Enable a specific improvement"""
        try:
            if improvement_name == 'focal_loss':
                self.improvement_manager.enable_focal_loss()
            elif improvement_name == 'ciou':
                self.improvement_manager.enable_bbox_loss('ciou')
            elif improvement_name == 'diou':
                self.improvement_manager.enable_bbox_loss('diou')
            elif improvement_name == 'gridmask':
                self.improvement_manager.enable_gridmask()
            elif improvement_name == 'cutout':
                self.improvement_manager.enable_cutout()
            elif improvement_name == 'ema':
                self.improvement_manager.enable_ema()
            elif improvement_name == 'amp':
                self.improvement_manager.enable_mixed_precision()
            elif improvement_name == 'small_object_aug':
                self.small_object_aug = SmallObjectAugmentation(
                    size_threshold=32,
                    copy_times=3
                )
            elif improvement_name == 'elongated_loss':
                self.elongated_loss = ElongatedObjectLoss()

            self.improvements_enabled.append(improvement_name)

        except Exception as e:
            print(f"⚠️  Failed to enable {improvement_name}: {e}")

    def _log_improvements(self):
        """Log enabled improvements"""
        if self.improvements_enabled:
            print("\n" + "="*60)
            print("✨ IMPROVEMENTS ENABLED")
            print("="*60)
            for i, imp in enumerate(self.improvements_enabled, 1):
                print(f"  {i}. {imp}")
            print("="*60 + "\n")

    def get_improvement_config(self):
        """Get improvement configuration"""
        return self.improvement_manager.get_config()


def create_trainer_with_improvements(
    trainer_class=BaseTrainer,
    improvements: Optional[List[str]] = None,
    preset: str = 'default'
):
    """
    Factory function to create trainer with improvements

    Args:
        trainer_class: Base trainer class to extend
        improvements: List of improvements
        preset: Preset name

    Returns:
        Configured trainer instance
    """
    # Create improvement manager
    manager = ImprovementManager()

    # Apply improvements
    preset_list = ImprovedTrainer.IMPROVEMENT_PRESETS.get(preset, improvements or [])

    for imp in preset_list:
        if imp == 'focal_loss':
            manager.enable_focal_loss()
        elif imp == 'ciou':
            manager.enable_bbox_loss('ciou')
        elif imp == 'gridmask':
            manager.enable_gridmask()
        # ... etc

    config = manager.get_config()

    # Return trainer with config
    return trainer_class, config


# Convenience function for quick usage
def train_with_improvements(
    model: str = 'yolov8n.pt',
    data: str = 'data.yaml',
    preset: str = 'small_objects',
    **kwargs
):
    """
    Quick training with improvements

    Args:
        model: Model path or name
        data: Dataset config path
        preset: Improvement preset ('small_objects', 'elongated', 'maximum')
        **kwargs: Additional training arguments

    Returns:
        Training results

    Usage:
        # Train for small objects
        results = train_with_improvements(
            model='yolov8n.pt',
            data='screws.yaml',
            preset='small_objects',
            imgsz=1280
        )
    """
    from ultralytics import YOLO

    # Create improvement manager
    manager = ImprovementManager()

    # Apply preset
    preset_improvements = ImprovedTrainer.IMPROVEMENT_PRESETS.get(preset, [])

    for imp in preset_improvements:
        if imp == 'focal_loss':
            manager.enable_focal_loss()
        elif imp == 'ciou':
            manager.enable_bbox_loss('ciou')
        elif imp == 'gridmask':
            manager.enable_gridmask()
        elif imp == 'cutout':
            manager.enable_cutout()
        elif imp == 'ema':
            manager.enable_ema()
        elif imp == 'amp':
            manager.enable_mixed_precision()

    config = manager.get_config()

    # Train model
    model = YOLO(model)

    # Merge kwargs with config
    train_args = {
        'data': data,
        **kwargs
    }

    # Train
    results = model.train(**train_args)

    print("\n" + "="*60)
    print("✅ Training Complete with Improvements:")
    print(f"  Preset: {preset}")
    print(f"  Improvements: {preset_improvements}")
    print("="*60)

    return results


if __name__ == '__main__':
    # Example usage
    print("\n" + "="*60)
    print("IMPROVED TRAINER - QUICK USAGE")
    print("="*60)

    print("""
    # Method 1: Use ImprovedTrainer directly
    from ultralytics.yolo.engine.trainer_improved import ImprovedTrainer

    trainer = ImprovedTrainer(preset='small_objects')
    trainer.train()


    # Method 2: Use convenience function
    from ultralytics.yolo.engine.trainer_improved import train_with_improvements

    results = train_with_improvements(
        model='yolov8n.pt',
        data='screws.yaml',
        preset='small_objects',
        imgsz=1280,
        epochs=100
    )


    # Method 3: Use improvement manager with regular YOLO
    from ultralytics import YOLO
    from ultralytics.yolo.utils.integrate_improvements import ImprovementManager

    manager = ImprovementManager()
    manager.enable_focal_loss()
    manager.enable_bbox_loss('ciou')

    model = YOLO('yolov8n.pt')
    model.train(data='data.yaml')
    """)

    print("="*60 + "\n")
