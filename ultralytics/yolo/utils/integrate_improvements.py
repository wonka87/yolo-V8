"""
Integration module for YOLO v8 improvements
This module provides easy API to enable/disable specific improvements
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional

from ..data.augment_improved import (
    GridMask, CutOut, RandomErasing, ImprovedMixUp,
    create_augmentation_pipeline
)
from ..utils.loss_improved import (
    FocalLoss, GIoULoss, DIoULoss, CIoULoss, 
    AlphaIoULoss, SIoULoss, create_bbox_loss
)


class ImprovementManager:
    """
    Manager class for enabling/disabling specific improvements
    """
    
    def __init__(self, model=None, config: Dict[str, Any] = None):
        """
        Initialize improvement manager
        
        Args:
            model: YOLO model instance
            config: Configuration dictionary with improvement settings
        """
        self.model = model
        self.config = config or {}
        self.original_modules = {}
        self.enabled_improvements = []
        
    def enable_focal_loss(self, alpha=0.25, gamma=2.0):
        """
        Enable Focal Loss for classification
        
        Args:
            alpha: Weighting factor in range (0, 1)
            gamma: Focusing parameter
        """
        print(f"✓ Enabling Focal Loss (alpha={alpha}, gamma={gamma})")
        self.enabled_improvements.append('focal_loss')
        self.config['use_focal_loss'] = True
        self.config['focal_alpha'] = alpha
        self.config['focal_gamma'] = gamma
        
    def enable_bbox_loss(self, loss_type='ciou', **kwargs):
        """
        Enable improved bounding box loss
        
        Args:
            loss_type: 'giou' | 'diou' | 'ciou' | 'alphaiou' | 'siou'
            **kwargs: Additional arguments for the loss function
        """
        print(f"✓ Enabling {loss_type.upper()} Loss for bounding box regression")
        self.enabled_improvements.append(f'bbox_loss_{loss_type}')
        self.config['bbox_loss'] = loss_type
        self.config['bbox_loss_params'] = kwargs
        
    def enable_gridmask(self, p=0.3, ratio=0.5, d_range=(96, 224)):
        """Enable GridMask augmentation"""
        print(f"✓ Enabling GridMask (p={p})")
        self.enabled_improvements.append('gridmask')
        self.config['gridmask'] = True
        self.config['gridmask_p'] = p
        self.config['gridmask_ratio'] = ratio
        self.config['gridmask_d_range'] = d_range
        
    def enable_cutout(self, p=0.3, n_holes=1, length=64):
        """Enable CutOut augmentation"""
        print(f"✓ Enabling CutOut (p={p}, holes={n_holes}, length={length})")
        self.enabled_improvements.append('cutout')
        self.config['cutout'] = True
        self.config['cutout_p'] = p
        self.config['cutout_n_holes'] = n_holes
        self.config['cutout_length'] = length
        
    def enable_random_erasing(self, p=0.3, sl=0.02, sh=0.4):
        """Enable Random Erasing augmentation"""
        print(f"✓ Enabling Random Erasing (p={p})")
        self.enabled_improvements.append('random_erasing')
        self.config['random_erasing'] = True
        self.config['random_erasing_p'] = p
        self.config['random_erasing_sl'] = sl
        self.config['random_erasing_sh'] = sh
        
    def enable_improved_mixup(self, p=0.15, max_ratio=0.5):
        """Enable improved MixUp augmentation"""
        print(f"✓ Enabling Improved MixUp (p={p})")
        self.enabled_improvements.append('mixup_improved')
        self.config['mixup_improved'] = True
        self.config['mixup_p'] = p
        self.config['mixup_max_ratio'] = max_ratio
        
    def enable_ema(self, decay=0.9998):
        """Enable Exponential Moving Average"""
        print(f"✓ Enabling EMA (decay={decay})")
        self.enabled_improvements.append('ema')
        self.config['ema'] = True
        self.config['ema_decay'] = decay
        
    def enable_mixed_precision(self):
        """Enable Automatic Mixed Precision training"""
        print("✓ Enabling Mixed Precision Training")
        self.enabled_improvements.append('amp')
        self.config['amp'] = True
        
    def enable_adamw(self, lr=0.001, weight_decay=0.01):
        """Enable AdamW optimizer"""
        print(f"✓ Enabling AdamW (lr={lr}, wd={weight_decay})")
        self.enabled_improvements.append('adamw')
        self.config['optimizer'] = 'AdamW'
        self.config['lr0'] = lr
        self.config['weight_decay'] = weight_decay
        
    def enable_cosine_scheduler(self, warmup_epochs=3):
        """Enable Cosine Annealing learning rate scheduler"""
        print(f"✓ Enabling Cosine Scheduler (warmup={warmup_epochs})")
        self.enabled_improvements.append('cosine_scheduler')
        self.config['lr_scheduler'] = 'cosine'
        self.config['warmup_epochs'] = warmup_epochs
        
    def enable_all_augmentations(self):
        """Enable all augmentation improvements"""
        self.enable_gridmask()
        self.enable_cutout()
        self.enable_random_erasing()
        self.enable_improved_mixup()
        
    def enable_all_losses(self, bbox_loss='ciou'):
        """Enable all loss function improvements"""
        self.enable_focal_loss()
        self.enable_bbox_loss(loss_type=bbox_loss)
        
    def enable_all_training(self):
        """Enable all training strategy improvements"""
        self.enable_ema()
        self.enable_mixed_precision()
        self.enable_adamw()
        self.enable_cosine_scheduler()
        
    def enable_all(self):
        """Enable all improvements"""
        print("\n" + "="*60)
        print("Enabling ALL improvements")
        print("="*60)
        
        self.enable_all_augmentations()
        self.enable_all_losses()
        self.enable_all_training()
        
        print("="*60)
        print(f"Total improvements enabled: {len(self.enabled_improvements)}")
        print("="*60 + "\n")
        
    def get_config(self):
        """Get current configuration"""
        return self.config.copy()
    
    def get_enabled_improvements(self):
        """Get list of enabled improvements"""
        return self.enabled_improvements.copy()
    
    def summary(self):
        """Print summary of enabled improvements"""
        print("\n" + "="*60)
        print("IMPROVEMENTS SUMMARY")
        print("="*60)
        
        if not self.enabled_improvements:
            print("No improvements enabled")
        else:
            for i, imp in enumerate(self.enabled_improvements, 1):
                print(f"{i}. {imp}")
        
        print("="*60 + "\n")


def create_improved_loss_fn(config: Dict[str, Any]):
    """
    Create improved loss function based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (bbox_loss_fn, cls_loss_fn)
    """
    # Bounding box loss
    bbox_loss_type = config.get('bbox_loss', 'ciou')
    bbox_loss_params = config.get('bbox_loss_params', {})
    
    if bbox_loss_type != 'original':
        bbox_loss_fn = create_bbox_loss(bbox_loss_type, **bbox_loss_params)
    else:
        bbox_loss_fn = None  # Use original
        
    # Classification loss
    if config.get('use_focal_loss', False):
        cls_loss_fn = FocalLoss(
            alpha=config.get('focal_alpha', 0.25),
            gamma=config.get('focal_gamma', 2.0)
        )
    else:
        cls_loss_fn = None  # Use original BCE
        
    return bbox_loss_fn, cls_loss_fn


def create_improved_augment_pipeline(config: Dict[str, Any], dataset=None):
    """
    Create improved augmentation pipeline based on configuration
    
    Args:
        config: Configuration dictionary
        dataset: Dataset object for mixup
        
    Returns:
        Augmentation pipeline
    """
    augmentations = []
    
    # GridMask
    if config.get('gridmask', False):
        augmentations.append(GridMask(
            p=config.get('gridmask_p', 0.3),
            ratio=config.get('gridmask_ratio', 0.5),
            d_range=tuple(config.get('gridmask_d_range', (96, 224)))
        ))
    
    # CutOut
    if config.get('cutout', False):
        augmentations.append(CutOut(
            p=config.get('cutout_p', 0.3),
            n_holes=config.get('cutout_n_holes', 1),
            length=config.get('cutout_length', 64)
        ))
    
    # Random Erasing
    if config.get('random_erasing', False):
        augmentations.append(RandomErasing(
            probability=config.get('random_erasing_p', 0.3),
            sl=config.get('random_erasing_sl', 0.02),
            sh=config.get('random_erasing_sh', 0.4)
        ))
    
    # Improved MixUp
    if config.get('mixup_improved', False) and dataset:
        augmentations.append(ImprovedMixUp(
            dataset=dataset,
            p=config.get('mixup_p', 0.15),
            max_mixup_ratio=config.get('mixup_max_ratio', 0.5)
        ))
    
    return augmentations


def quick_test_improvement(improvement_name: str = 'all'):
    """
    Quick test to verify improvement modules work
    
    Args:
        improvement_name: Name of improvement to test or 'all'
    """
    print("\n" + "="*60)
    print(f"Testing improvement: {improvement_name}")
    print("="*60)
    
    try:
        # Test loss functions
        print("\n[Testing Loss Functions]")
        
        dummy_boxes = torch.rand(10, 4) * 100
        dummy_boxes[:, 2:] += dummy_boxes[:, :2]  # Convert to xyxy
        
        # Test each loss type
        for loss_type in ['giou', 'diou', 'ciou', 'alphaiou', 'siou']:
            loss_fn = create_bbox_loss(loss_type)
            loss_value = loss_fn(dummy_boxes, dummy_boxes)
            print(f"  ✓ {loss_type.upper()} Loss: {loss_value.item():.4f}")
            
        # Test Focal Loss
        print("\n[Test Focal Loss]")
        focal_loss = FocalLoss()
        pred = torch.randn(10, 80)
        target = torch.randint(0, 80, (10,))
        loss = focal_loss(pred, target)
        print(f"  ✓ Focal Loss: {loss.item():.4f}")
        
        # Test augmentations
        print("\n[Test Augmentations]")
        
        # Create dummy labels dict
        dummy_labels = {
            'img': torch.randint(0, 255, (640, 640, 3), dtype=torch.uint8).numpy()
        }
        
        # Test GridMask
        gridmask = GridMask(p=1.0)
        result = gridmask(dummy_labels.copy())
        print(f"  ✓ GridMask: {result['img'].shape}")
        
        # Test CutOut
        cutout = CutOut(p=1.0)
        result = cutout(dummy_labels.copy())
        print(f"  ✓ CutOut: {result['img'].shape}")
        
        # Test Random Erasing
        erasing = RandomErasing(probability=1.0)
        result = erasing(dummy_labels.copy())
        print(f"  ✓ Random Erasing: {result['img'].shape}")
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# Convenience function
def setup_improvements_from_config(config_file: str):
    """
    Setup improvements from YAML config file
    
    Args:
        config_file: Path to config file
        
    Returns:
        ImprovementManager instance
    """
    import yaml
    
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    manager = ImprovementManager(config=config)
    
    # Helper to apply config
    def apply_config_section(section_name, target_section=None):
        """Apply configuration from a section"""
        section = config.get(section_name, {})
        target = target_section or manager.config
        
        for key, value in section.items():
            if key.startswith('enable_'):
                # Call enable method
                method_name = key
                if hasattr(manager, method_name):
                    if isinstance(value, dict):
                        getattr(manager, method_name)(**value)
                    else:
                        getattr(manager, method_name)()
            else:
                target[key] = value
    
    # Apply baseline first
    apply_config_section('baseline')
    
    # Then apply experimental config if specified
    experiment = config.get('experiment', 'baseline')
    if experiment != 'baseline' and experiment in config:
        apply_config_section(experiment)
    
    return manager


if __name__ == '__main__':
    # Quick test
    quick_test_improvement()
