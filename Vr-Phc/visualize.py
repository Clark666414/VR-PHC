#!/usr/bin/env python3
"""
Pico-Body-Fitter: Visualization Tool
Displays original tracking data (red spheres) and fitted SMPL mesh (gray) side-by-side.
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import pickle
import json

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Error: Open3D not installed. Install with: pip install open3d")
    sys.exit(1)

import torch
from src.smpl_engine import SMPL_layer


def parse_pose_string(pose_str):
    """Parse pose string 'x,y,z,qx,qy,qz,qw' into position."""
    values = [float(v) for v in pose_str.split(',')]
    position = np.array(values[:3])
    return position


def load_tracking_data_raw(file_path):
    """Load raw tracking data from txt file."""
    print(f"Loading raw tracking data from: {file_path}")

    frames = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                if 'notice' in data:
                    continue

                if 'Body' in data and 'joints' in data['Body']:
                    joints = []
                    for joint in data['Body']['joints']:
                        if 'p' in joint:
                            joint_pos = parse_pose_string(joint['p'])
                            joints.append(joint_pos)

                    if len(joints) >= 24:
                        frames.append(np.array(joints[:24]))

            except json.JSONDecodeError:
                continue

    joints_array = np.array(frames)
    print(f"✓ Loaded {len(joints_array)} frames")
    print(f"  Shape: {joints_array.shape}")

    return joints_array


def load_fitted_result(pkl_path, smpl_model_path, device='cpu'):
    """Load fitted SMPL result and generate meshes."""
    print(f"\nLoading fitted result from: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    poses = data['poses']
    trans = data['trans']
    betas = data['betas']
    target_joints = data.get('target_joints', None)

    print(f"✓ Loaded fitted data")
    print(f"  Frames: {poses.shape[0]}")
    print(f"  Poses: {poses.shape}")
    print(f"  Trans: {trans.shape}")

    # Initialize SMPL
    h36m_jreg_path = Path('data/smpl_models/J_regressor_h36m.npy')
    h36m_jregressor = np.load(str(h36m_jreg_path)) if h36m_jreg_path.exists() else None

    smpl = SMPL_layer(
        str(smpl_model_path),
        h36m_jregressor=h36m_jregressor,
        dtype=torch.float32
    ).to(device)

    # Generate meshes
    print(f"\nGenerating SMPL meshes...")
    num_frames = poses.shape[0]

    poses_torch = torch.from_numpy(poses).float().to(device)
    trans_torch = torch.from_numpy(trans).float().to(device)
    betas_torch = torch.from_numpy(betas).float().to(device)
    betas_batch = betas_torch.unsqueeze(0).expand(num_frames, -1)

    with torch.no_grad():
        output = smpl(
            pose_axis_angle=poses_torch[:, 3:],
            betas=betas_batch,
            global_orient=poses_torch[:, :3],
            transl=trans_torch,
            return_verts=True
        )

    vertices = output.vertices.cpu().numpy()
    joints = output.joints.cpu().numpy()
    faces = smpl.faces.astype(np.int32)

    print(f"✓ Generated meshes")
    print(f"  Vertices: {vertices.shape}")
    print(f"  Joints: {joints.shape}")
    print(f"  Faces: {faces.shape}")

    return vertices, faces, joints, target_joints


def create_coordinate_frame(size=0.5):
    """Create world coordinate frame (X=Red, Y=Green, Z=Blue)."""
    points = [
        [0, 0, 0],
        [size, 0, 0],  # X-axis (Red)
        [0, size, 0],  # Y-axis (Green)
        [0, 0, size],  # Z-axis (Blue)
    ]
    lines = [[0, 1], [0, 2], [0, 3]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def create_ground_plane(size=4.0, divisions=20):
    """Create ground plane grid."""
    points = []
    lines = []
    colors = []

    step = size / divisions
    half_size = size / 2

    for i in range(divisions + 1):
        offset = -half_size + i * step

        # Lines parallel to X-axis
        points.append([offset, 0, -half_size])
        points.append([offset, 0, half_size])
        lines.append([len(points) - 2, len(points) - 1])
        colors.append([0.3, 0.3, 0.3])

        # Lines parallel to Z-axis
        points.append([-half_size, 0, offset])
        points.append([half_size, 0, offset])
        lines.append([len(points) - 2, len(points) - 1])
        colors.append([0.3, 0.3, 0.3])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def visualize_comparison(amass_joints, target_joints, fitted_vertices, fitted_faces, framerate=30.0):
    """Visualize comparison between target joints, AMASS joints, and fitted mesh."""
    if not HAS_OPEN3D:
        print("Error: Open3D is required for visualization")
        return

    num_frames = min(len(amass_joints), len(fitted_vertices))
    if target_joints is not None:
        num_frames = min(num_frames, len(target_joints))

    print(f"\n{'='*80}")
    print("PICO-BODY-FITTER VISUALIZATION")
    print(f"{'='*80}")
    print("  RED SPHERES   = Target joints from tracking data")
    print("  BLUE SPHERES  = SMPL joint positions (from fitted parameters)")
    print("  GRAY MESH     = Fitted SMPL body mesh")
    print(f"{'='*80}")
    print("\nCONTROLS:")
    print("  SPACE     - Play/Pause")
    print("  RIGHT     - Next frame")
    print("  LEFT      - Previous frame")
    print("  R         - Reset view")
    print("  Q/ESC     - Quit")
    print(f"{'='*80}\n")

    # Create visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Pico-Body-Fitter Visualization", width=1280, height=720)

    # Create fitted mesh (GRAY)
    fitted_mesh = o3d.geometry.TriangleMesh()
    fitted_mesh.vertices = o3d.utility.Vector3dVector(fitted_vertices[0])
    fitted_mesh.triangles = o3d.utility.Vector3iVector(fitted_faces)
    fitted_mesh.compute_vertex_normals()
    fitted_mesh.paint_uniform_color([0.7, 0.7, 0.7])

    # Create AMASS joint spheres (BLUE)
    amass_spheres = []
    for j in range(amass_joints.shape[1]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.paint_uniform_color([0.0, 0.5, 1.0])
        sphere.translate(amass_joints[0, j])
        amass_spheres.append(sphere)

    # Create target joint spheres (RED)
    target_spheres = []
    if target_joints is not None:
        for j in range(target_joints.shape[1]):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            sphere.paint_uniform_color([1.0, 0.0, 0.0])
            sphere.translate(target_joints[0, j])
            target_spheres.append(sphere)

    # Create coordinate frame and ground plane
    coord_frame = create_coordinate_frame(size=0.5)
    ground = create_ground_plane(size=4.0, divisions=20)

    # Add geometries
    vis.add_geometry(fitted_mesh)
    for sphere in amass_spheres:
        vis.add_geometry(sphere)
    for sphere in target_spheres:
        vis.add_geometry(sphere)
    vis.add_geometry(coord_frame)
    vis.add_geometry(ground)

    # Animation state
    state = {
        'frame': 0,
        'playing': True,
        'last_update': 0
    }

    def update_frame(frame_idx):
        """Update visualization to show specific frame"""
        frame_idx = max(0, min(frame_idx, num_frames - 1))

        # Update fitted mesh
        fitted_mesh.vertices = o3d.utility.Vector3dVector(fitted_vertices[frame_idx])
        fitted_mesh.compute_vertex_normals()
        vis.update_geometry(fitted_mesh)

        # Update AMASS joint spheres
        for j, sphere in enumerate(amass_spheres):
            sphere.translate(-sphere.get_center())
            sphere.translate(amass_joints[frame_idx, j])
            vis.update_geometry(sphere)

        # Update target joint spheres
        if target_joints is not None:
            for j, sphere in enumerate(target_spheres):
                sphere.translate(-sphere.get_center())
                sphere.translate(target_joints[frame_idx, j])
                vis.update_geometry(sphere)

        state['frame'] = frame_idx

    def toggle_play(vis):
        state['playing'] = not state['playing']
        status = "Playing" if state['playing'] else "Paused"
        print(f"\r{status} - Frame {state['frame']}/{num_frames}", end='', flush=True)
        return False

    def next_frame(vis):
        state['playing'] = False
        state['frame'] = (state['frame'] + 1) % num_frames
        update_frame(state['frame'])
        print(f"\rPaused - Frame {state['frame']}/{num_frames}", end='', flush=True)
        return False

    def prev_frame(vis):
        state['playing'] = False
        state['frame'] = (state['frame'] - 1) % num_frames
        update_frame(state['frame'])
        print(f"\rPaused - Frame {state['frame']}/{num_frames}", end='', flush=True)
        return False

    # Register key callbacks
    vis.register_key_callback(32, toggle_play)   # SPACE
    vis.register_key_callback(262, next_frame)   # RIGHT
    vis.register_key_callback(263, prev_frame)   # LEFT

    # Set camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, 1, 0])

    # Animation loop
    import time
    frame_time = 1.0 / framerate

    print(f"Playing animation ({num_frames} frames @ {framerate} fps)...")
    print(f"Press SPACE to pause, Q to quit\n")

    while vis.poll_events():
        current_time = time.time()

        if state['playing'] and (current_time - state['last_update']) >= frame_time:
            state['frame'] = (state['frame'] + 1) % num_frames
            update_frame(state['frame'])
            state['last_update'] = current_time

            print(f"\rPlaying - Frame {state['frame']}/{num_frames}", end='', flush=True)

        vis.update_renderer()

    vis.destroy_window()
    print("\n\nVisualization closed.")


def main():
    parser = argparse.ArgumentParser(
        description='Visualization tool for Pico-Body-Fitter results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize fitted result
  python visualize.py data/input_pico/sample.txt data/output/sample_fitted.pkl

  # Use CPU for visualization (recommended)
  python visualize.py data/input_pico/sample.txt data/output/sample_fitted.pkl --device cpu
        """
    )

    parser.add_argument('tracking_file', type=str,
                       help='Path to trackingData_*.txt file (raw data)')
    parser.add_argument('fitted_pkl', type=str,
                       help='Path to fitted .pkl file (result)')
    parser.add_argument('--smpl_model', type=str,
                       default='data/smpl_models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
                       help='Path to SMPL model file')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cuda', 'cpu'],
                       help='Compute device (use cpu for visualization)')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='Playback framerate (default: 30)')

    args = parser.parse_args()

    # Check files exist
    if not os.path.exists(args.tracking_file):
        print(f"Error: Tracking file not found: {args.tracking_file}")
        sys.exit(1)

    if not os.path.exists(args.fitted_pkl):
        print(f"Error: Fitted pkl not found: {args.fitted_pkl}")
        sys.exit(1)

    if not os.path.exists(args.smpl_model):
        print(f"Error: SMPL model not found: {args.smpl_model}")
        sys.exit(1)

    print("=" * 80)
    print("Pico-Body-Fitter: Visualization Tool")
    print("=" * 80)

    # Load fitted result
    fitted_vertices, fitted_faces, amass_joints, target_joints = load_fitted_result(
        args.fitted_pkl,
        args.smpl_model,
        args.device
    )

    # Visualize comparison
    visualize_comparison(amass_joints[:, :24, :], target_joints, fitted_vertices, fitted_faces, framerate=args.fps)


if __name__ == '__main__':
    main()
