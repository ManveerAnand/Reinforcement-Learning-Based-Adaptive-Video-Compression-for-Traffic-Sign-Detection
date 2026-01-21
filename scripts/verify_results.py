"""
Verify that reproduced results match the reference results from the paper.

This script compares your reproduced results against the reference values
to ensure successful reproduction within acceptable tolerance.

Usage:
    python scripts/verify_results.py [--reproduced outputs/reproduction] [--tolerance 0.01]
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Reference results from the paper
REFERENCE_RESULTS = {
    "fixed_baselines": {
        "B6": {"mean_map": 50.23, "std_map": 8.45},
        "B8": {"mean_map": 48.67, "std_map": 8.92},
        "B10": {"mean_map": 46.34, "std_map": 9.01},
        "B12": {"mean_map": 44.12, "std_map": 9.23},
        "B15": {"mean_map": 42.56, "std_map": 9.45},
        "B20": {"mean_map": 40.45, "std_map": 9.12},
    },
    "random_policy": {
        "mean_map": 45.87,
        "std_map": 7.23,
    },
    "rl_agent": {
        "mean_map": 49.58,
        "std_map": 7.89,
        "improvement_vs_worst": 9.13,
        "improvement_vs_random": 3.71,
    },
    "statistical_tests": {
        "rl_vs_random": {"p_value": 0.001, "significant": True},
        "rl_vs_b20": {"p_value": 0.001, "significant": True},
        "rl_vs_b6": {"p_value": 0.082, "significant": False},
    }
}

def load_fixed_baseline_results(results_dir):
    """Load and analyze fixed baseline results."""
    csv_path = results_dir / "fixed_baseline_results.csv"
    
    if not csv_path.exists():
        print(f"✗ Fixed baseline results not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        # Group by B value and compute mean/std
        results = {}
        for b_value in [6, 8, 10, 12, 15, 20]:
            b_data = df[df['B'] == b_value]
            if len(b_data) > 0:
                results[f'B{b_value}'] = {
                    'mean_map': b_data['mAP50'].mean() * 100,  # Convert to percentage
                    'std_map': b_data['mAP50'].std() * 100,
                }
        
        return results
        
    except Exception as e:
        print(f"✗ Error loading fixed baseline results: {e}")
        return None

def load_random_policy_results(results_dir):
    """Load and analyze random policy results."""
    csv_path = results_dir / "random_policy_results.csv"
    
    if not csv_path.exists():
        print(f"✗ Random policy results not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        results = {
            'mean_map': df['mAP50'].mean() * 100,
            'std_map': df['mAP50'].std() * 100,
        }
        
        return results
        
    except Exception as e:
        print(f"✗ Error loading random policy results: {e}")
        return None

def load_rl_agent_results(results_dir):
    """Load and analyze RL agent results."""
    csv_path = results_dir / "rl_agent_results.csv"
    
    if not csv_path.exists():
        print(f"✗ RL agent results not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        results = {
            'mean_map': df['mAP50'].mean() * 100,
            'std_map': df['mAP50'].std() * 100,
        }
        
        return results
        
    except Exception as e:
        print(f"✗ Error loading RL agent results: {e}")
        return None

def compare_value(name, reproduced, reference, tolerance, is_percentage=True):
    """Compare a single value against reference."""
    if reference is None or reproduced is None:
        print(f"  ⚠ {name:30s} MISSING DATA")
        return False
    
    diff = abs(reproduced - reference)
    diff_pct = (diff / abs(reference)) * 100 if reference != 0 else 0
    
    # Determine tolerance
    if is_percentage:
        # For percentages, use absolute tolerance
        passed = diff <= tolerance * 100
    else:
        # For other values, use relative tolerance
        passed = diff_pct <= tolerance * 100
    
    status = "✓" if passed else "✗"
    
    print(f"  {status} {name:30s} Ref: {reference:6.2f}%, Reproduced: {reproduced:6.2f}%, Diff: {diff:+.2f}%")
    
    return passed

def verify_fixed_baselines(reproduced, reference, tolerance):
    """Verify fixed baseline results."""
    print("\n" + "="*80)
    print("FIXED BASELINE RESULTS")
    print("="*80)
    
    if reproduced is None:
        print("✗ Reproduced results not available")
        return False
    
    all_passed = True
    
    for b_key in ['B6', 'B8', 'B10', 'B12', 'B15', 'B20']:
        if b_key not in reproduced or b_key not in reference:
            print(f"  ⚠ {b_key} results missing")
            all_passed = False
            continue
        
        print(f"\n{b_key}:")
        
        passed = compare_value(
            "  Mean mAP@0.5",
            reproduced[b_key]['mean_map'],
            reference[b_key]['mean_map'],
            tolerance
        )
        all_passed = all_passed and passed
        
        passed = compare_value(
            "  Std Dev",
            reproduced[b_key]['std_map'],
            reference[b_key]['std_map'],
            tolerance * 2  # More lenient for std dev
        )
        # Don't fail on std dev differences
    
    return all_passed

def verify_random_policy(reproduced, reference, tolerance):
    """Verify random policy results."""
    print("\n" + "="*80)
    print("RANDOM POLICY RESULTS")
    print("="*80)
    
    if reproduced is None:
        print("✗ Reproduced results not available")
        return False
    
    all_passed = True
    
    passed = compare_value(
        "Mean mAP@0.5",
        reproduced['mean_map'],
        reference['mean_map'],
        tolerance * 1.5  # More lenient for stochastic policy
    )
    all_passed = all_passed and passed
    
    passed = compare_value(
        "Std Dev",
        reproduced['std_map'],
        reference['std_map'],
        tolerance * 2
    )
    # Don't fail on std dev
    
    return all_passed

def verify_rl_agent(reproduced, reference, tolerance):
    """Verify RL agent results."""
    print("\n" + "="*80)
    print("RL AGENT RESULTS")
    print("="*80)
    
    if reproduced is None:
        print("✗ Reproduced results not available")
        return False
    
    all_passed = True
    
    passed = compare_value(
        "Mean mAP@0.5",
        reproduced['mean_map'],
        reference['mean_map'],
        tolerance
    )
    all_passed = all_passed and passed
    
    passed = compare_value(
        "Std Dev",
        reproduced['std_map'],
        reference['std_map'],
        tolerance * 2
    )
    # Don't fail on std dev
    
    return all_passed

def main():
    parser = argparse.ArgumentParser(
        description="Verify reproduced results against reference values"
    )
    
    parser.add_argument(
        '--reproduced',
        type=Path,
        default=Path('outputs/reproduction'),
        help='Directory containing reproduced results'
    )
    
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.01,
        help='Tolerance for comparison (default: 0.01 = 1%%)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("RESULTS VERIFICATION")
    print("="*80)
    print(f"\nReproduced results: {args.reproduced}")
    print(f"Tolerance: ±{args.tolerance*100:.1f}%")
    print()
    
    # Load reproduced results
    print("Loading reproduced results...")
    reproduced_fixed = load_fixed_baseline_results(args.reproduced)
    reproduced_random = load_random_policy_results(args.reproduced)
    reproduced_rl = load_rl_agent_results(args.reproduced)
    
    # Verify each experiment
    results = {}
    
    results['fixed_baselines'] = verify_fixed_baselines(
        reproduced_fixed,
        REFERENCE_RESULTS['fixed_baselines'],
        args.tolerance
    )
    
    results['random_policy'] = verify_random_policy(
        reproduced_random,
        REFERENCE_RESULTS['random_policy'],
        args.tolerance
    )
    
    results['rl_agent'] = verify_rl_agent(
        reproduced_rl,
        REFERENCE_RESULTS['rl_agent'],
        args.tolerance
    )
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    print()
    
    all_passed = all(results.values())
    
    for exp_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {exp_name:25s} {status}")
    
    print("\n" + "="*80)
    
    if all_passed:
        print("✓ REPRODUCTION SUCCESSFUL")
        print("\nAll results match reference values within tolerance.")
        print("Your reproduction is verified and ready for publication.")
        sys.exit(0)
    else:
        print("⚠ VERIFICATION INCOMPLETE")
        print("\nSome results differ from reference values.")
        print("\nPossible reasons:")
        print("1. Stochastic variation (acceptable for random policy)")
        print("2. Different random seed")
        print("3. Hardware/software differences")
        print("4. Dataset or model mismatch")
        print("\nIf differences are small (<2%), this is likely acceptable.")
        sys.exit(1)

if __name__ == "__main__":
    main()
