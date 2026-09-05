#!/usr/bin/env python3
"""
Quick Start Demo for YOLO v8 Improvements
Shows how to use the improvements in just a few lines of code
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics.yolo.utils.integrate_improvements import (
    ImprovementManager,
    quick_test_improvement
)


def demo_quick_start():
    """
    Quick demonstration of YOLO v8 improvements
    """
    print("\n" + "="*80)
    print("🚀 YOLO v8 IMPROVEMENTS - QUICK START DEMO")
    print("="*80)
    
    # Step 1: Quick verification
    print("\n📦 Step 1: Verifying all modules...")
    print("-"*80)
    
    success = quick_test_improvement('all')
    
    if not success:
        print("\n❌ Some modules failed verification")
        return
    
    # Step 2: Create improvement manager
    print("\n🎯 Step 2: Creating improvement configuration...")
    print("-"*80)
    
    manager = ImprovementManager()
    
    # Demo scenarios
    scenarios = [
        {
            'name': 'Scenario 1: Focal Loss for Class Imbalance',
            'setup': lambda: manager.enable_focal_loss(alpha=0.25, gamma=2.0),
            'use_case': 'Use when you have imbalanced classes'
        },
        {
            'name': 'Scenario 2: CIoU Loss for Better BBox Regression',
            'setup': lambda: manager.enable_bbox_loss('ciou'),
            'use_case': 'Use for better bounding box accuracy'
        },
        {
            'name': 'Scenario 3: GridMask for Occluded Objects',
            'setup': lambda: manager.enable_gridmask(p=0.3),
            'use_case': 'Use when objects are often partially hidden'
        },
        {
            'name': 'Scenario 4: CutOut for Regularization',
            'setup': lambda: manager.enable_cutout(p=0.3, n_holes=1, length=64),
            'use_case': 'Simple and effective augmentation'
        },
        {
            'name': 'Scenario 5: Mixed Precision Training',
            'setup': lambda: manager.enable_mixed_precision(),
            'use_case': 'Speed up training by 30-50%'
        },
        {
            'name': 'Scenario 6: EMA for Better Generalization',
            'setup': lambda: manager.enable_ema(decay=0.9998),
            'use_case': 'More stable and accurate model'
        },
        {
            'name': 'Scenario 7: All Improvements Combined',
            'setup': lambda: manager.enable_all(),
            'use_case': 'Maximum accuracy improvement'
        }
    ]
    
    print("\n📋 Available Scenarios:")
    print("-"*80)
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Use case: {scenario['use_case']}")
    
    # Interactive selection
    print("\n" + "="*80)
    print("Choose a scenario (1-7) or 'all' to test everything")
    print("="*80)
    
    choice = input("\nEnter your choice (1-7/all): ").strip().lower()
    
    if choice == 'all':
        print("\n🧪 Testing all scenarios...")
        print("="*80)
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n[{i}/{len(scenarios)}] {scenario['name']}")
            print("-"*80)
            
            # Reset manager
            manager = ImprovementManager()
            
            # Setup scenario
            scenario['setup']()
            
            # Show configuration
            manager.summary()
            
            # Get configuration
            config = manager.get_config()
            print(f"\nConfiguration:")
            for key, value in config.items():
                print(f"  - {key}: {value}")
            
            print("\n✓ Scenario complete!")
        
        print("\n" + "="*80)
        print("✅ All scenarios tested successfully!")
        print("="*80)
        
    elif choice.isdigit() and 1 <= int(choice) <= 7:
        idx = int(choice) - 1
        scenario = scenarios[idx]
        
        print(f"\n🎯 Setting up: {scenario['name']}")
        print("="*80)
        
        # Reset manager
        manager = ImprovementManager()
        
        # Setup
        scenario['setup']()
        
        # Show summary
        manager.summary()
        
        # Show configuration
        config = manager.get_config()
        print("\n" + "="*80)
        print("CONFIGURATION:")
        print("="*80)
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        print("\n" + "="*80)
        print(f"✅ {scenario['name']} is ready!")
        print("="*80)
        
        print("\n🎓 Next steps:")
        print("  1. Integrate this configuration into your training script")
        print("  2. Run baseline benchmark: python scripts/run_benchmarks.py --mode baseline")
        print("  3. Train with improvements and compare")
        print("  4. Run ablation study: python scripts/test_improvements_standalone.py --mode ablation")
        
    else:
        print("\n❌ Invalid choice")
        return
    
    print("\n" + "="*80)
    print("📚 For more information, see:")
    print("  - docs/IMPROVEMENTS_GUIDE.md")
    print("  - scripts/validation_checklist.md")
    print("="*80 + "\n")


def demo_api_usage():
    """
    Show API usage examples
    """
    print("\n" + "="*80)
    print("💡 API USAGE EXAMPLES")
    print("="*80)
    
    examples = """
# Example 1: Enable specific improvements
from ultralytics.yolo.utils.integrate_improvements import ImprovementManager

manager = ImprovementManager()
manager.enable_focal_loss(alpha=0.25, gamma=2.0)
manager.enable_bbox_loss('ciou')
manager.enable_gridmask(p=0.3)
config = manager.get_config()


# Example 2: Quick test all modules
from ultralytics.yolo.utils.integrate_improvements import quick_test_improvement

quick_test_improvement('all')


# Example 3: Use augmentation directly
from ultralytics.yolo.data.augment_improved import GridMask, CutOut

labels = {'img': image_array}
gridmask = GridMask(p=0.3)
result = gridmask(labels)


# Example 4: Use loss functions directly
from ultralytics.yolo.utils.loss_improved import FocalLoss, CIoULoss
import torch

# Focal Loss
focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
pred = torch.randn(10, 80)
target = torch.randint(0, 80, (10,))
loss = focal_loss(pred, target)

# CIoU Loss
ciou_loss = CIoULoss()
pred_boxes = torch.rand(10, 4)
target_boxes = torch.rand(10, 4)
loss = ciou_loss(pred_boxes, target_boxes)


# Example 5: Run benchmark
from tests.test_benchmark import BenchmarkRunner

runner = BenchmarkRunner('yolov8n.pt', 'coco128.yaml')
results = runner.run_full_validation()


# Example 6: Ablation study
from tests.test_benchmark import AblationStudy

study = AblationStudy('yolov8n.pt', 'coco128.yaml')
study.add_experiment('baseline', 'Original model')
study.add_experiment('focal_loss', 'With Focal Loss')
study.add_experiment('ciou', 'With CIoU Loss')
results = study.run_all()
"""
    
    print(examples)
    print("="*80 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO v8 Improvements Quick Start')
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['demo', 'api', 'test'],
                       help='Mode to run')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        demo_quick_start()
    elif args.mode == 'api':
        demo_api_usage()
    elif args.mode == 'test':
        quick_test_improvement('all')


# Usage:
"""
# Interactive demo
python quick_start.py

# Show API examples
python quick_start.py --mode api

# Quick test
python quick_start.py --mode test
"""
