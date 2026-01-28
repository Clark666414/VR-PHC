"""
SMPL optimization solver with geometric constraints.
Implements pure geometric fitting with biological constraints and anti-jitter filtering.
"""

import numpy as np
import torch
try:
    from smplx import SMPL
except ImportError:
    print("Error: smplx not installed. Install with: pip install smplx")
    import sys
    sys.exit(1)

from .filters import apply_double_filter


def fit_smpl_batch(target_joints, smpl_model_path, device='cuda', num_iterations=500,
                   weight_joints=0.5, weight_pose_prior=2.0, weight_spine_lock=20.0,
                   weight_foot_flat=5.0, weight_knee_hinge=20.0, weight_balance=2.0,
                   weight_temporal=50.0, smooth_sigma=3.0):
    """
    Fit SMPL model to target joint positions using pure optimization with geometric constraints.

    This approach prioritizes extreme motion smoothness over positional accuracy, implementing
    a "spring-damper" strategy where joints are loosely tethered to targets rather than rigidly locked.

    Args:
        target_joints: (N, 24, 3) array of target joint positions
        smpl_model_path: Path to SMPL model files
        device: 'cuda' or 'cpu'
        num_iterations: Number of optimization iterations
        weight_joints: Weight for joint position loss (default: 0.5, very low for tolerance)
        weight_pose_prior: Weight for pose prior loss (default: 2.0, moderate regularization)
        weight_spine_lock: Weight for spine lock loss - prevents 180° flip (default: 20.0)
        weight_foot_flat: Weight for foot flat loss (default: 5.0)
        weight_knee_hinge: Weight for knee hinge constraint - prevents lateral bending (default: 20.0)
        weight_balance: Weight for balance loss - keeps CoM over feet (default: 2.0)
        weight_temporal: Weight for temporal smoothness - prevents frame-to-frame jitter (default: 50.0, very high)
        smooth_sigma: Gaussian smoothing sigma for post-processing (default: 3.0, aggressive)

    Returns:
        Dictionary with fitted SMPL parameters
    """
    print(f"\n=== Pure Optimization SMPL Fitting ===")
    print(f"Device: {device}")
    print(f"Frames: {len(target_joints)}")
    print(f"Iterations: {num_iterations}")
    print(f"\nLoss weights (Extreme Smoothing Mode):")
    print(f"  Joints: {weight_joints} (very low - loose tethering)")
    print(f"  Pose Prior: {weight_pose_prior} (moderate regularization)")
    print(f"  Spine Lock: {weight_spine_lock} (prevents 180° flip)")
    print(f"  Foot Flat: {weight_foot_flat}")
    print(f"  Knee Hinge: {weight_knee_hinge} (prevents lateral knee bending)")
    print(f"  Balance: {weight_balance} (keeps CoM over feet)")
    print(f"  Temporal: {weight_temporal} (very high - extreme smoothness)")
    print(f"  Gaussian Sigma: {smooth_sigma} (aggressive post-processing)")

    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'

    # Initialize SMPL model
    smpl = SMPL(
        model_path=smpl_model_path,
        gender='neutral',
        batch_size=len(target_joints)
    ).to(device)

    # Convert target joints to torch tensor
    target_joints_torch = torch.tensor(target_joints, dtype=torch.float32, device=device)

    # Initialize SMPL parameters
    batch_size = len(target_joints)

    # Smart initialization: Calculate initial global orientation from first frame head direction
    print("\n=== Smart Initialization ===")
    first_frame_head = target_joints[0, 15, :]  # Head joint (index 15)
    first_frame_pelvis = target_joints[0, 0, :]  # Pelvis joint (index 0)

    # Calculate head-to-pelvis vector (body forward direction)
    body_forward = first_frame_head - first_frame_pelvis
    body_forward = body_forward / (np.linalg.norm(body_forward) + 1e-8)

    # Calculate rotation to align with this direction
    # Default SMPL forward is +Y axis in local space
    default_forward = np.array([0.0, 1.0, 0.0])

    # Calculate rotation axis and angle
    rotation_axis = np.cross(default_forward, body_forward)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    if rotation_axis_norm > 1e-6:
        rotation_axis = rotation_axis / rotation_axis_norm
        rotation_angle = np.arccos(np.clip(np.dot(default_forward, body_forward), -1.0, 1.0))
        initial_global_orient = rotation_axis * rotation_angle
    else:
        initial_global_orient = np.zeros(3)

    print(f"  First frame head direction: {body_forward}")
    print(f"  Initial global_orient: {initial_global_orient}")

    # Global orientation (3,) - axis-angle, initialized with smart value
    global_orient = torch.tensor(
        np.tile(initial_global_orient, (batch_size, 1)),
        dtype=torch.float32, device=device, requires_grad=True
    )

    # Body pose (69,) - 23 joints * 3 (axis-angle)
    body_pose = torch.zeros((batch_size, 69), dtype=torch.float32, device=device, requires_grad=True)

    # Translation (3,)
    transl = torch.zeros((batch_size, 3), dtype=torch.float32, device=device, requires_grad=True)

    # Body shape (10,) - shared across all frames
    betas = torch.zeros((1, 10), dtype=torch.float32, device=device, requires_grad=True)

    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': [global_orient], 'lr': 0.01},
        {'params': [body_pose], 'lr': 0.01},
        {'params': [transl], 'lr': 0.01},
        {'params': [betas], 'lr': 0.001},
    ])

    # Optimization loop
    print("\n=== Starting Optimization ===")

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Forward pass through SMPL
        smpl_output = smpl(
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            betas=betas.expand(batch_size, -1)
        )

        # Get SMPL joint positions (first 24 joints)
        smpl_joints = smpl_output.joints[:, :24, :]

        # ============================================================
        # Loss 1: Joint Position Loss (with Selective Root Constraint)
        # ============================================================
        joint_errors = (smpl_joints - target_joints_torch) ** 2  # (N, 24, 3)

        # Create weight mask: relax root (pelvis) Y-axis to allow backward movement
        joint_weights = torch.ones(24, 3, device=device) * 10.0
        joint_weights[0, 1] = 0.1  # Root Y-axis (depth): very low weight

        # Increase weights for arm joints to improve arm fitting quality
        arm_joint_indices = [16, 17, 18, 19]  # Shoulders and elbows
        for idx in arm_joint_indices:
            joint_weights[idx, :] = 30.0

        # Extra emphasis on wrists (end effectors)
        wrist_joint_indices = [20, 21]
        for idx in wrist_joint_indices:
            joint_weights[idx, :] = 35.0

        # Apply selective weights
        weighted_errors = joint_errors * joint_weights.unsqueeze(0)
        loss_joints = torch.mean(weighted_errors)

        # ============================================================
        # Loss 2: Pose Prior Loss
        # ============================================================
        loss_pose_prior = torch.mean(body_pose ** 2)

        # ============================================================
        # Loss 3: Spine Lock Loss - CRITICAL for preventing 180° flip
        # ============================================================
        spine1_rot = body_pose[:, 6:9]
        spine2_rot = body_pose[:, 15:18]
        spine3_rot = body_pose[:, 24:27]
        neck_rot = body_pose[:, 33:36]
        head_rot = body_pose[:, 42:45]

        # Method 1: Direct constraint - penalize large rotations in spine chain
        loss_spine_direct = (torch.mean(spine1_rot ** 2) +
                            torch.mean(spine2_rot ** 2) +
                            torch.mean(spine3_rot ** 2) +
                            torch.mean(neck_rot ** 2) +
                            torch.mean(head_rot ** 2))

        # Method 2: Geometric constraint - ensure head and pelvis face same direction
        head_pos = smpl_joints[:, 15, :]  # Head joint
        pelvis_pos = smpl_joints[:, 0, :]  # Pelvis joint
        neck_pos = smpl_joints[:, 12, :]  # Neck joint

        # Calculate vectors
        head_to_neck = head_pos - neck_pos
        pelvis_to_spine = smpl_joints[:, 9, :] - pelvis_pos

        # Normalize vectors
        head_to_neck_norm = head_to_neck / (torch.norm(head_to_neck, dim=-1, keepdim=True) + 1e-8)
        pelvis_to_spine_norm = pelvis_to_spine / (torch.norm(pelvis_to_spine, dim=-1, keepdim=True) + 1e-8)

        # Compute dot product - should be close to 1.0 (same direction)
        dot_product = torch.sum(head_to_neck_norm * pelvis_to_spine_norm, dim=-1)
        loss_spine_geometric = torch.mean((1.0 - dot_product) ** 2)

        # Combine both methods for strong spine lock
        loss_spine_lock = loss_spine_direct + loss_spine_geometric

        # ============================================================
        # Loss 4: Foot Flat Loss
        # ============================================================
        left_foot_rot = body_pose[:, 27:30]  # Joint 10
        right_foot_rot = body_pose[:, 30:33]  # Joint 11

        # Penalize X and Z rotations (only allow Y-axis rotation)
        loss_foot_flat = (torch.mean(left_foot_rot[:, [0, 2]] ** 2) +
                         torch.mean(right_foot_rot[:, [0, 2]] ** 2))

        # ============================================================
        # Loss 5: Knee Hinge Constraint
        # ============================================================
        left_knee_rot = body_pose[:, 9:12]   # Joint 4
        right_knee_rot = body_pose[:, 12:15]  # Joint 5

        # Stack knee rotations: (N, 2, 3)
        knee_pose = torch.stack([left_knee_rot, right_knee_rot], dim=1)

        # Heavily penalize any rotation in non-primary axes
        loss_knee_hinge = torch.mean(knee_pose[:, :, 1:] ** 2)

        # ============================================================
        # Loss 6: Balance Loss - Center of Mass over Feet
        # ============================================================
        com_upper = (head_pos + pelvis_pos) / 2.0

        # Compute feet center (support polygon center)
        left_foot = smpl_joints[:, 10, :]
        right_foot = smpl_joints[:, 11, :]
        center_feet = (left_foot + right_foot) / 2.0

        # Compute horizontal plane distance
        com_horizontal = com_upper[:, [0, 1]]
        feet_horizontal = center_feet[:, [0, 1]]

        # Penalize horizontal offset between CoM and support center
        loss_balance = torch.mean((com_horizontal - feet_horizontal) ** 2)

        # ============================================================
        # Loss 7: Temporal Smoothness Loss
        # ============================================================
        if batch_size > 1:
            global_orient_diff = global_orient[1:] - global_orient[:-1]
            body_pose_diff = body_pose[1:] - body_pose[:-1]
            transl_diff = transl[1:] - transl[:-1]

            loss_temporal = (torch.mean(global_orient_diff ** 2) +
                           torch.mean(body_pose_diff ** 2) +
                           torch.mean(transl_diff ** 2))
        else:
            loss_temporal = torch.tensor(0.0, device=device)

        # ============================================================
        # Total Loss
        # ============================================================
        loss = (weight_joints * loss_joints +
                weight_pose_prior * loss_pose_prior +
                weight_spine_lock * loss_spine_lock +
                weight_foot_flat * loss_foot_flat +
                weight_knee_hinge * loss_knee_hinge +
                weight_balance * loss_balance +
                weight_temporal * loss_temporal)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print progress
        if (iteration + 1) % 50 == 0 or iteration == 0:
            print(f"  Iter {iteration+1}/{num_iterations} | "
                  f"Total: {loss.item():.6f} | "
                  f"Joint: {loss_joints.item():.6f} | "
                  f"Prior: {loss_pose_prior.item():.6f} | "
                  f"Spine: {loss_spine_lock.item():.6f} | "
                  f"Foot: {loss_foot_flat.item():.6f} | "
                  f"Knee: {loss_knee_hinge.item():.6f} | "
                  f"Balance: {loss_balance.item():.6f} | "
                  f"Temporal: {loss_temporal.item():.6f}")

    print(f"\n✓ Optimization complete! Final loss: {loss.item():.6f}")

    # ============================================================
    # Apply Double Filtering (Butterworth + Gaussian)
    # ============================================================
    print(f"\n=== Applying Output Filtering ===")
    print(f"  Double filtering: Butterworth (10Hz) + Gaussian (sigma={smooth_sigma})")

    # Convert to numpy for filtering
    global_orient_np = global_orient.detach().cpu().numpy()
    body_pose_np = body_pose.detach().cpu().numpy()
    transl_np = transl.detach().cpu().numpy()

    # Apply double filter to each dimension
    global_orient_smooth = np.zeros_like(global_orient_np)
    for dim in range(3):
        global_orient_smooth[:, dim] = apply_double_filter(
            global_orient_np[:, dim],
            butterworth_cutoff=10.0,
            gaussian_sigma=smooth_sigma,
            fs=30.0
        )

    body_pose_smooth = np.zeros_like(body_pose_np)
    for dim in range(69):
        body_pose_smooth[:, dim] = apply_double_filter(
            body_pose_np[:, dim],
            butterworth_cutoff=10.0,
            gaussian_sigma=smooth_sigma,
            fs=30.0
        )

    transl_smooth = np.zeros_like(transl_np)
    for dim in range(3):
        transl_smooth[:, dim] = apply_double_filter(
            transl_np[:, dim],
            butterworth_cutoff=10.0,
            gaussian_sigma=smooth_sigma,
            fs=30.0
        )

    print(f"  ✓ Double filtering complete - motion is now extremely smooth")

    # Extract results (using double-filtered outputs)
    results = {
        'global_orient': global_orient_smooth,
        'body_pose': body_pose_smooth,
        'transl': transl_smooth,
        'betas': betas.detach().cpu().numpy()[0],
        'joints': smpl_joints.detach().cpu().numpy()
    }

    return results
