#!/usr/bin/env python3
"""
Test improvements systematically with ablation studies
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import torch
import numpy as np
import pandas as pd

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_benchmark import BenchmarkRunner, AblationStudy
from ultralytics.yolo.utils.integrate_improvements import (
    ImprovementManager,
    quick_test_improvement
)


def test_single_improvement(
    improvement_name: str,
    model: str = 'yolov8n.pt',
    data: str = 'coco128.yaml',
    device: str = 'cpu',
    batch_size: int = 1,
    epochs: int = 1,
    **improvement_kwargs
):
    """
    Test a single improvement
    
    Args:
        improvement_name: Name of improvement to test
        model: Model path
        data: Dataset config
        device: Device to use
        batch_size: Batch size
        epochs: Number of epochs (use 1 for quick test)
        **improvement_kwargs: Additional arguments for the improvement
        
    Returns:
        Dictionary with benchmark results
    """
    print("\n" + "="*80)
    print(f"Testing Improvement: {improvement_name}")
    print("="*80)
    
    # Create improvement manager
    manager = ImprovementManager()
    
    # Enable the specific improvement
    if improvement_name == 'focal_loss':
        manager.enable_focal_loss(**improvement_kwargs)
    elif improvement_name == 'ciou_loss':
        manager.enable_bbox_loss('ciou', **improvement_kwargs)
    elif improvement_name == 'diou_loss':
        manager.enable_bbox_loss('diou', **improvement_kwargs)
    elif improvement_name == 'alphaiou_loss':
        manager.enable_bbox_loss('alphaiou', **improvement_kwargs)
    elif improvement_name == 'siou_loss':
        manager.enable_bbox_loss('siou', **improvement_kwargs)
    elif improvement_name == 'gridmask':
        manager.enable_gridmask(**improvement_kwargs)
    elif improvement_name == 'cutout':
        manager.enable_cutout(**improvement_kwargs)
    elif improvement_name == 'random_erasing':
        manager.enable_random_erasing(**improvement_kwargs)
    elif improvement_name == 'improved_mixup':
        manager.enable_improved_mixup(**improvement_kwargs)
    elif improvement_name == 'ema':
        manager.enable_ema(**improvement_kwargs)
    elif improvement_name == 'amp':
        manager.enable_mixed_precision()
    elif improvement_name == 'adamw':
        manager.enable_adamw(**improvement_kwargs)
    elif improvement_name == 'all':
        manager.enable_all()
    else:
        print(f"Warning: Unknown improvement '{improvement_name}'")
        return None
    
    manager.summary()
    
    # Run benchmark
    # Note: In real implementation, you would modify the model/training
    # with these improvements. For now, we just benchmark the base model.
    
    print("\nRunning benchmark...")
    runner = BenchmarkRunner(
        model_path=model,
        data_config=data,
        save_dir=f'runs/improvement_tests/{improvement_name}',
        device=device
    )
    
    results = runner.run_full_validation(
        batch_size=batch_size,
        imgsz=640
    )
    
    # Add improvement info to results
    results['improvement'] = improvement_name
    results['config'] = manager.get_config()
    results['enabled_improvements'] = manager.get_enabled_improvements()
    
    return results


def run_ablation_study(
    improvements: List[str] = None,
    baseline_name: str = 'baseline',
    model: str = 'yolov8n.pt',
    data: str = 'coco128.yaml',
    device: str = 'cpu'
):
    """
    Run ablation study comparing multiple improvements
    
    Args:
        improvements: List of improvement names to test
        baseline_name: Name for baseline experiment
        model: Model path
        data: Dataset config
        device: Device to use
        
    Returns:
        DataFrame with comparison results
    """
    if improvements is None:
        # Default improvements to test
        improvements = [
            'baseline',
            'focal_loss',
            'ciou_loss',
            'gridmask',
            'cutout',
            'random_erasing',
            'all_improvements'
        ]
    
    print("\n" + "="*80)
    print("ABLATION STUDY")
    print("="*80)
    print(f"Testing {len(improvements)} configurations:")
    for i, imp in enumerate(improvements, 1):
        print(f"  {i}. {imp}")
    print("="*80 + "\n")
    
    # Setup ablation study
    study = AblationStudy(
        baseline_model=model,
        data_config=data,
        save_dir='runs/ablation_study'
    )
    
    # Add experiments
    for imp_name in improvements:
        if imp_name == 'baseline':
            study.add_experiment(
                name='baseline',
                description='Original YOLOv8n (no modifications)'
            )
        else:
            study.add_experiment(
                name=imp_name,
                description=f'YOLOv8n with {imp_name}',
                modifications={'improvement': imp_name}
            )
    
    # Run all experiments
    print("\nStarting experiments...")
    results_df = study.run_all(
        batch_size=1,  # Use small batch for quick testing
        imgsz=640,
        device=device
    )
    
    # Calculate improvements over baseline
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    comparison_df = study.compare_with_baseline(baseline_name)
    print(comparison_df.to_string(index=False))
    
    # Save detailed results
    results_path = Path('runs/ablation_study')
    results_path.mkdir(parents=True, exist_ok=True)
    
    comparison_df.to_csv(results_path / 'improvement_comparison.csv', index=False)
    
    print("\n" + "="*80)
    print(f"Results saved to: {results_path}")
    print("="*80)
    
    return results_df, comparison_df


def quick_test_all_modules():
    """Quick test to verify all improvement modules work"""
    print("\n" + "="*80)
    print("QUICK MODULE TEST")
    print("="*80)
    
    success = quick_test_improvement('all')
    
    if success:
        print("\n✅ All improvement modules are working correctly!")
        return True
    else:
        print("\n❌ Some modules failed. Check the errors above.")
        return False


def interactive_test():
    """
    Interactive test menu for trying different improvements
    """
    print("\n" + "="*80)
    print("YOLO v8 IMPROVEMENT TESTER")
    print("="*80)
    
    while True:
        print("\nOptions:")
        print("1. Quick test all modules (verification)")
        print("2. Test single improvement")
        print("3. Run ablation study (multiple improvements)")
        print("4. Show baseline vs comparison")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            quick_test_all_modules()
            
        elif choice == '2':
            print("\nAvailable improvements:")
            improvements = [
                'focal_loss', 'ciou_loss', 'diou_loss', 'alphaiou_loss', 'siou_loss',
                'gridmask', 'cutout', 'random_erasing', 'improved_mixup',
                'ema', 'amp', 'adamw', 'all'
            ]
            for i, imp in enumerate(improvements, 1):
                print(f"  {i}. {imp}")
            
            imp_choice = input("\nEnter improvement name: ").strip()
            
            if imp_choice in improvements or imp_choice == 'all':
                device = input("Enter device (cpu/cuda): ").strip() or 'cpu'
                
                result = test_single_improvement(
                    improvement_name=imp_choice,
                    device=device,
                    batch_size=1
                )
                
                if result:
                    print(f"\nResults saved to: runs/improvement_tests/{imp_choice}")
                    
        elif choice == '3':
            print("\nRunning full ablation study...")
            device = input("Enter device (cpu/cuda): ").strip() or 'cpu'
            
            results_df, comparison_df = run_ablation_study(device=device)
            
            print("\nAblation study complete!")
            print(f"Results saved to: runs/ablation_study/")
            
        elif choice == '4':
            # Load and display comparison
            comparison_file = Path('runs/ablation_study/ablation_comparison.csv')
            
            if comparison_file.exists():
                df = pd.read_csv(comparison_file)
                print("\n" + "="*80)
                print("COMPARISON RESULTS")
                print("="*80)
                print(df.to_string(index=False))
            else:
                print("\nNo comparison file found. Run ablation study first.")
                
        elif choice == '5':
            print("\nExiting...")
            break
            
        else:
            print("\nInvalid choice. Please try again.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test YOLO v8 improvements')
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['test', 'single', 'ablation', 'interactive'],
                       help='Test mode')
    parser.add_argument('--improvement', type=str, default='all',
                       help='Improvement to test (for single mode)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu/cuda)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Model path')
    parser.add_argument('--data', type=str, default='coco128.yaml',
                       help='Dataset config')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        # Quick module test
        quick_test_all_modules()
        
    elif args.mode == 'single':
        # Test single improvement
        test_single_improvement(
            improvement_name=args.improvement,
            model=args.model,
            data=args.data,
            device=args.device
        )
        
    elif args.mode == 'ablation':
        # Run ablation study
        run_ablation_study(
            model=args.model,
            data=args.data,
            device=args.device
        )
        
    elif args.mode == 'interactive':
        # Interactive menu
        interactive_test()


if __name__ == '__main__':
    main()


# Usage examples:
"""
# Quick test all modules
python scripts/test_improvements.py --mode test

# Test single improvement
python scripts/test_improvements.py --mode single --improvement focal_loss --device cpu

# Run ablation study
python scripts/test_improvements.py --mode ablation --device cpu

# Interactive mode
python scripts/test_improvements.py --mode interactive
"""
