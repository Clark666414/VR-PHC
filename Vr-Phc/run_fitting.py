#!/usr/bin/env python3
"""
Pico-Body-Fitter: Main Entry Point
Fits SMPL model to Pico VR full-body tracking data using pure geometric optimization.
"""

import os
import sys
import argparse
from pathlib import Path

# Import from src module
from src import (
    load_tracking_data,
    prepare_target_joints,
    fit_smpl_batch,
    export_to_phc_format
)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Fit SMPL model to Pico VR full-body tracking data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python run_fitting.py data/input_pico/sample.txt

  # Specify SMPL model path and output location
  python run_fitting.py data/input_pico/sample.txt \\
      --smpl_model_path data/smpl_models \\
      --output data/output/result.pkl

  # Adjust optimization parameters for better quality
  python run_fitting.py data/input_pico/sample.txt \\
      --iterations 1000 \\
      --weight_spine_lock 30.0 \\
      --weight_knee_hinge 25.0
        """
    )

    parser.add_argument('input_file', type=str,
                        help='Path to trackingData_*.txt file')
    parser.add_argument('--smpl_model_path', type=str, default='data/smpl_models',
                        help='Path to SMPL model directory (default: data/smpl_models)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for result.pkl (default: auto-generated in data/output/)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Mocap framerate (default: 30)')
    parser.add_argument('--iterations', type=int, default=500,
                        help='Number of optimization iterations (default: 500)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for optimization (default: cuda)')

    # Coordinate system fixes
    parser.add_argument('--no_mirror_fix', action='store_true',
                        help='Disable X-axis mirroring fix')
    parser.add_argument('--rotate_180', action='store_true',
                        help='Enable 180° Y-axis rotation fix')
    parser.add_argument('--no_root_zero', action='store_true',
                        help='Disable root position zeroing')

    # Loss weights
    parser.add_argument('--weight_joints', type=float, default=0.5,
                        help='Weight for joint position loss (default: 0.5)')
    parser.add_argument('--weight_pose_prior', type=float, default=2.0,
                        help='Weight for pose prior loss (default: 2.0)')
    parser.add_argument('--weight_spine_lock', type=float, default=20.0,
                        help='Weight for spine lock loss (default: 20.0)')
    parser.add_argument('--weight_foot_flat', type=float, default=5.0,
                        help='Weight for foot flat loss (default: 5.0)')
    parser.add_argument('--weight_knee_hinge', type=float, default=20.0,
                        help='Weight for knee hinge constraint (default: 20.0)')
    parser.add_argument('--weight_balance', type=float, default=2.0,
                        help='Weight for balance loss (default: 2.0)')
    parser.add_argument('--weight_temporal', type=float, default=50.0,
                        help='Weight for temporal smoothness (default: 50.0)')
    parser.add_argument('--smooth_sigma', type=float, default=3.0,
                        help='Gaussian smoothing sigma (default: 3.0)')

    args = parser.parse_args()

    # Check input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    # Check SMPL model path
    if not os.path.exists(args.smpl_model_path):
        print(f"Error: SMPL model path not found: {args.smpl_model_path}")
        print("\nPlease download SMPL model from https://smpl.is.tue.mpg.de/")
        print("and place the files in data/smpl_models/")
        sys.exit(1)

    # Generate output path if not specified
    if args.output is None:
        input_name = Path(args.input_file).stem
        args.output = f"data/output/{input_name}_fitted.pkl"
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("=" * 80)
    print("Pico-Body-Fitter: Pure Geometric SMPL Fitting")
    print("=" * 80)

    # Step 1: Load tracking data
    frames = load_tracking_data(args.input_file)

    if len(frames) == 0:
        print("Error: No valid frames found!")
        sys.exit(1)

    # Step 2: Prepare target joints
    target_joints, valid_frames = prepare_target_joints(
        frames,
        fix_mirror_x=not args.no_mirror_fix,
        fix_rotate_y_180=args.rotate_180,
        fix_root_zero=not args.no_root_zero
    )

    if len(target_joints) == 0:
        print("Error: No valid target joints prepared!")
        sys.exit(1)

    # Step 3: Fit SMPL model with pure optimization
    results = fit_smpl_batch(
        target_joints,
        args.smpl_model_path,
        device=args.device,
        num_iterations=args.iterations,
        weight_joints=args.weight_joints,
        weight_pose_prior=args.weight_pose_prior,
        weight_spine_lock=args.weight_spine_lock,
        weight_foot_flat=args.weight_foot_flat,
        weight_knee_hinge=args.weight_knee_hinge,
        weight_balance=args.weight_balance,
        weight_temporal=args.weight_temporal,
        smooth_sigma=args.smooth_sigma
    )

    # Step 4: Export to PHC format
    data = export_to_phc_format(results, target_joints, args.output, fps=args.fps)

    print("\n" + "=" * 80)
    print("✓ Fitting complete!")
    print(f"✓ Output saved to: {args.output}")
    print("=" * 80)

    # Print next steps
    print("\nNext steps:")
    print(f"1. Visualize the result:")
    print(f"   python visualize.py {args.input_file} {args.output}")
    print(f"\n2. If you need to adjust parameters, re-run with different values:")
    print(f"   python run_fitting.py {args.input_file} \\")
    print(f"       --weight_spine_lock 25.0 \\")
    print(f"       --weight_knee_hinge 25.0 \\")
    print(f"       --iterations 1000")


if __name__ == "__main__":
    main()
