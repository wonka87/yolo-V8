#!/usr/bin/env python3
"""
Script to run comprehensive benchmarks before and after improvements
This creates baseline metrics and compares improvements
"""

import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_benchmark import BenchmarkRunner, AblationStudy


def create_baseline(model: str = 'yolov8n.pt', 
                    data: str = 'coco128.yaml',
                    device: str = ''):
    """Create baseline benchmark results"""
    print("="*80)
    print("CREATING BASELINE BENCHMARK")
    print("="*80)
    print(f"Model: {model}")
    print(f"Dataset: {data}")
    print(f"Device: {device or 'auto'}")
    print()
    
    runner = BenchmarkRunner(
        model_path=model,
        data_config=data,
        save_dir='runs/baseline',
        device=device
    )
    
    results = runner.run_full_validation()
    
    # Save as baseline
    baseline_file = Path('runs/baseline_metrics.json')
    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(baseline_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Baseline saved to {baseline_file}")
    return results


def compare_with_baseline(improved_results: dict, 
                          baseline_file: str = 'runs/baseline_metrics.json'):
    """Compare improved results with baseline"""
    
    if not Path(baseline_file).exists():
        print("Warning: No baseline found. Run create_baseline() first.")
        return None
    
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
    
    # Calculate improvements
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'improvements': {}
    }
    
    # Accuracy metrics
    metrics = ['mAP50', 'mAP50-95', 'precision', 'recall']
    for metric in metrics:
        baseline_val = baseline['metrics'][metric]
        improved_val = improved_results['metrics'][metric]
        delta = improved_val - baseline_val
        percent = (delta / baseline_val) * 100 if baseline_val != 0 else 0
        
        comparison['improvements'][metric] = {
            'baseline': baseline_val,
            'improved': improved_val,
            'delta': delta,
            'percent': f"{percent:+.2f}%"
        }
    
    # Speed metrics
    speed_metrics = ['fps', 'mean_latency_ms']
    for metric in speed_metrics:
        baseline_val = baseline['speed'][metric]
        improved_val = improved_results['speed'][metric]
        delta = improved_val - baseline_val
        percent = (delta / baseline_val) * 100 if baseline_val != 0 else 0
        
        comparison['improvements'][f'speed_{metric}'] = {
            'baseline': baseline_val,
            'improved': improved_val,
            'delta': delta,
            'percent': f"{percent:+.2f}%"
        }
    
    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON WITH BASELINE")
    print("="*80)
    print(f"{'Metric':<20} {'Baseline':<12} {'Improved':<12} {'Delta':<12} {'%':<10}")
    print("-"*80)
    
    for metric, values in comparison['improvements'].items():
        print(f"{metric:<20} {values['baseline']:<12.4f} {values['improved']:<12.4f} "
              f"{values['delta']:<12.4f} {values['percent']:<10}")
    
    print("="*80)
    
    return comparison


def run_improvement_test(improvement_name: str,
                         model_path: str,
                         data: str = 'coco128.yaml',
                         device: str = ''):
    """Test a specific improvement and compare with baseline"""
    
    print(f"\nTesting improvement: {improvement_name}")
    print("="*80)
    
    runner = BenchmarkRunner(
        model_path=model_path,
        data_config=data,
        save_dir=f'runs/improvements/{improvement_name}',
        device=device
    )
    
    results = runner.run_full_validation()
    
    # Compare with baseline
    comparison = compare_with_baseline(results)
    
    # Save comparison
    comp_file = Path(f'runs/improvements/{improvement_name}/comparison.json')
    comp_file.parent.mkdir(parents=True, exist_ok=True)
    with open(comp_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    return results, comparison


def main():
    """Main function to demonstrate benchmark workflow"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run YOLO v8 benchmarks')
    parser.add_argument('--mode', type=str, default='baseline',
                       choices=['baseline', 'compare', 'full'],
                       help='Running mode')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Model path or name')
    parser.add_argument('--data', type=str, default='coco128.yaml',
                       help='Dataset config file')
    parser.add_argument('--device', type=str, default='',
                       help='Device (e.g., 0 or cpu)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    
    args = parser.parse_args()
    
    if args.mode == 'baseline':
        # Create baseline only
        create_baseline(model=args.model, data=args.data, device=args.device)
    
    elif args.mode == 'compare':
        # Test current model and compare with baseline
        runner = BenchmarkRunner(
            model_path=args.model,
            data_config=args.data,
            device=args.device
        )
        results = runner.run_full_validation(batch_size=args.batch)
        compare_with_baseline(results)
    
    elif args.mode == 'full':
        # Full workflow: baseline + test
        print("Step 1: Creating baseline...")
        create_baseline(model=args.model, data=args.data, device=args.device)
        
        print("\nStep 2: Running current model benchmark...")
        runner = BenchmarkRunner(
            model_path=args.model,
            data_config=args.data,
            save_dir='runs/current',
            device=args.device
        )
        results = runner.run_full_validation(batch_size=args.batch)
        
        print("\nStep 3: Comparing with baseline...")
        compare_with_baseline(results)


if __name__ == '__main__':
    main()


# Example usage in Python:
"""
# Example 1: Create baseline
python scripts/run_benchmarks.py --mode baseline --model yolov8n.pt --data coco128.yaml

# Example 2: Compare current model with baseline
python scripts/run_benchmarks.py --mode compare --model path/to/improved_model.pt

# Example 3: Full workflow
python scripts/run_benchmarks.py --mode full

# Example 4: Custom device
python scripts/run_benchmarks.py --mode baseline --device cuda:0
python scripts/run_benchmarks.py --mode baseline --device cpu
"""
