"""
Utility functions for VR tracking data processing.
Handles parsing, coordinate transformations, and data preparation for SMPL fitting.
"""

import json
import numpy as np
import pickle
from pathlib import Path
from .filters import apply_lowpass_filter


# SMPL joint names (24 joints, indices 0-23)
SMPL_JOINT_NAMES = [
    'pelvis',           # 0
    'left_hip',         # 1
    'right_hip',        # 2
    'spine1',           # 3
    'left_knee',        # 4
    'right_knee',       # 5
    'spine2',           # 6
    'left_ankle',       # 7
    'right_ankle',      # 8
    'spine3',           # 9
    'left_foot',        # 10
    'right_foot',       # 11
    'neck',             # 12
    'left_collar',      # 13
    'right_collar',     # 14
    'head',             # 15
    'left_shoulder',    # 16
    'right_shoulder',   # 17
    'left_elbow',       # 18
    'right_elbow',      # 19
    'left_wrist',       # 20
    'right_wrist',      # 21
    'left_hand',        # 22
    'right_hand',       # 23
]

# Pico Body joint mapping to SMPL joints
PICO_TO_SMPL_MAPPING = {
    0: (0, 'pelvis'),
    1: (1, 'left_hip'),
    2: (2, 'right_hip'),
    3: (3, 'spine1'),
    4: (4, 'left_knee'),
    5: (5, 'right_knee'),
    6: (6, 'spine2'),
    7: (7, 'left_ankle'),
    8: (8, 'right_ankle'),
    9: (9, 'spine3'),
    10: (10, 'left_foot'),
    11: (11, 'right_foot'),
    12: (12, 'neck'),
    13: (13, 'left_collar'),
    14: (14, 'right_collar'),
    15: (15, 'head'),
    16: (16, 'left_shoulder'),
    17: (17, 'right_shoulder'),
    18: (18, 'left_elbow'),
    19: (19, 'right_elbow'),
    20: (20, 'left_wrist'),
    21: (21, 'right_wrist'),
    22: (22, 'left_hand'),
    23: (23, 'right_hand'),
}


def parse_pose_string(pose_str):
    """Parse pose string 'x,y,z,qx,qy,qz,qw' into position and quaternion."""
    values = [float(v) for v in pose_str.split(',')]
    position = np.array(values[:3])
    quaternion = np.array(values[3:7]) if len(values) >= 7 else None
    return position, quaternion


def load_tracking_data(file_path):
    """
    Load and parse tracking data from JSON file.

    Args:
        file_path: Path to trackingData_*.txt file

    Returns:
        List of frame dictionaries with 'frame_num', 'head', and 'body_joints'
    """
    print(f"Loading tracking data from: {file_path}")

    frames = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Skip the first line if it's just a notice
                if 'notice' in data:
                    continue

                frame_data = {
                    'frame_num': line_num,
                    'head': None,
                    'body_joints': []
                }

                # Parse Head data (for alignment)
                if 'Head' in data and 'pose' in data['Head']:
                    head_pos, head_quat = parse_pose_string(data['Head']['pose'])
                    frame_data['head'] = head_pos

                # Parse Body joint data
                if 'Body' in data and 'joints' in data['Body']:
                    for joint in data['Body']['joints']:
                        if 'p' in joint:
                            joint_pos, joint_quat = parse_pose_string(joint['p'])
                            frame_data['body_joints'].append(joint_pos)

                frames.append(frame_data)

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue

    print(f"✓ Loaded {len(frames)} frames")
    return frames


def convert_yup_to_zup(positions):
    """
    Convert from Y-up (Unity/Pico) to Z-up (SMPL/PHC).
    Y-up: X=right, Y=up, Z=forward
    Z-up: X=right, Y=forward, Z=up
    Transformation: [x, y, z] -> [x, z, y]
    """
    if len(positions.shape) == 1:
        # Single position
        return np.array([positions[0], positions[2], positions[1]])
    else:
        # Multiple positions
        converted = positions.copy()
        converted[:, [0, 1, 2]] = positions[:, [0, 2, 1]]
        return converted


def rotate_y(positions, degrees):
    """
    Rotate positions around Y-axis by specified degrees.

    Args:
        positions: (N, 3) or (3,) array of positions
        degrees: Rotation angle in degrees

    Returns:
        Rotated positions
    """
    angle_rad = np.radians(degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    if len(positions.shape) == 1:
        # Single position
        x, y, z = positions
        return np.array([
            cos_a * x + sin_a * z,
            y,
            -sin_a * x + cos_a * z
        ])
    else:
        # Multiple positions
        rotated = positions.copy()
        x = positions[:, 0]
        z = positions[:, 2]
        rotated[:, 0] = cos_a * x + sin_a * z
        rotated[:, 2] = -sin_a * x + cos_a * z
        return rotated


def align_body_to_head(body_joints, head_pos):
    """
    Align body root to tracked head position.
    This prevents body-head separation issues.
    """
    if len(body_joints) < 16:
        return body_joints

    # Body joint 15 should be the head
    body_head = body_joints[15]

    # Calculate offset
    offset = head_pos - body_head

    # Apply offset to all body joints
    aligned_joints = body_joints + offset

    return aligned_joints


def prepare_target_joints(frames, fix_mirror_x=True, fix_rotate_y_180=False, fix_root_zero=True):
    """
    Prepare target joint positions for SMPL fitting.

    Args:
        frames: List of frame dictionaries from load_tracking_data()
        fix_mirror_x: Negate X coordinates to fix left-right mirroring
        fix_rotate_y_180: Rotate 180° around Y axis to fix front-back orientation
        fix_root_zero: Zero out root position to prevent drift

    Returns:
        target_joints: (N_frames, 24, 3) array of target positions
        valid_frames: List of valid frame indices
    """
    print("\n=== Preparing Target Joints ===")
    print(f"\nCoordinate system fixes:")
    print(f"  FIX_MIRROR_X:     {fix_mirror_x}")
    print(f"  FIX_ROTATE_Y_180: {fix_rotate_y_180}")
    print(f"  FIX_ROOT_ZERO:    {fix_root_zero}")

    target_joints_list = []
    valid_frames = []
    first_frame_root_offset = None

    for frame_idx, frame in enumerate(frames):
        if len(frame['body_joints']) < 24:
            print(f"Warning: Frame {frame_idx} has only {len(frame['body_joints'])} joints, skipping")
            continue

        body_joints = np.array(frame['body_joints'])

        # Align body to head if head tracking is available
        if frame['head'] is not None:
            body_joints = align_body_to_head(body_joints, frame['head'])

        # Convert coordinate system: Y-up -> Z-up
        body_joints = convert_yup_to_zup(body_joints)

        # Apply coordinate fixes
        if frame_idx == 0:
            print("\nApplying coordinate fixes:")

            if fix_mirror_x:
                body_joints[:, 0] *= -1
                print("  Applied FIX_MIRROR_X: negated X coordinates")

            if fix_rotate_y_180:
                body_joints = rotate_y(body_joints, 180)
                print("  Applied FIX_ROTATE_Y_180: rotated 180° around Y axis")

            if fix_root_zero:
                first_frame_root_offset = body_joints[0].copy()
                body_joints -= first_frame_root_offset
                print(f"  Applied FIX_ROOT_ZERO: zeroed root (offset: {first_frame_root_offset})")
        else:
            # Apply same transforms to subsequent frames
            if fix_mirror_x:
                body_joints[:, 0] *= -1

            if fix_rotate_y_180:
                body_joints = rotate_y(body_joints, 180)

            if fix_root_zero and first_frame_root_offset is not None:
                body_joints -= first_frame_root_offset

        # Map Pico joints to SMPL joints
        smpl_joints = np.zeros((24, 3))
        for pico_idx, (smpl_idx, joint_name) in PICO_TO_SMPL_MAPPING.items():
            if pico_idx < len(body_joints):
                smpl_joints[smpl_idx] = body_joints[pico_idx]

        target_joints_list.append(smpl_joints)
        valid_frames.append(frame_idx)

    target_joints = np.array(target_joints_list)

    print(f"\n✓ Prepared {len(target_joints)} valid frames")
    print(f"  Target shape: {target_joints.shape}")

    # Apply input filtering to remove Pico tracking noise
    print(f"\n=== Applying Input Filtering ===")
    print(f"  Filtering target joints to remove Pico tracking noise")
    print(f"  Cutoff frequency: 5.0 Hz (removes high-frequency jitter)")

    target_joints_filtered = np.zeros_like(target_joints)
    for joint_idx in range(24):
        for dim in range(3):  # X, Y, Z
            target_joints_filtered[:, joint_idx, dim] = apply_lowpass_filter(
                target_joints[:, joint_idx, dim],
                cutoff=5.0,
                fs=30.0,
                order=4
            )

    target_joints = target_joints_filtered
    print(f"  ✓ Input filtering complete - smooth target trajectories")

    # Analyze joint positions
    joint_ranges = target_joints.max(axis=0) - target_joints.min(axis=0)
    print(f"\nJoint position ranges (across all frames):")
    print(f"  X-axis: {joint_ranges[:, 0].mean():.3f}m (avg)")
    print(f"  Y-axis: {joint_ranges[:, 1].mean():.3f}m (avg)")
    print(f"  Z-axis: {joint_ranges[:, 2].mean():.3f}m (avg)")

    return target_joints, valid_frames


def export_to_phc_format(results, target_joints, output_path, fps=30):
    """
    Export fitted SMPL parameters to PHC-compatible format.

    Format: amass_data.pkl with keys:
        - poses: (N, 72) - global_orient (3) + body_pose (69)
        - trans: (N, 3) - translation
        - betas: (10,) - body shape
        - mocap_framerate: float
        - target_joints: (N, 24, 3) - original target positions for debugging
    """
    print(f"\n=== Exporting to PHC Format ===")

    # Combine global_orient and body_pose
    poses = np.concatenate([
        results['global_orient'],  # (N, 3)
        results['body_pose']       # (N, 69)
    ], axis=1)  # (N, 72)

    # Prepare data dictionary
    data = {
        'poses': poses,
        'trans': results['transl'],
        'betas': results['betas'],
        'mocap_framerate': fps,
        'gender': 'neutral',
        'target_joints': target_joints  # Save for visualization debugging
    }

    # Save to pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"✓ Saved to: {output_path}")
    print(f"  Poses shape: {poses.shape}")
    print(f"  Trans shape: {data['trans'].shape}")
    print(f"  Betas shape: {data['betas'].shape}")
    print(f"  Target joints shape: {target_joints.shape}")
    print(f"  FPS: {fps}")

    return data
