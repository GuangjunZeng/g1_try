#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""
Quick start example for G1 robot retargeting.

This script demonstrates basic usage of the retargeting system.
"""

import torch
import numpy as np
from g1_interaction.retargeting.smplx_retarget import SMPLXRetargeter
from g1_interaction import G1_INTERACTION_ROOT_DIR


def main():
    print("=" * 80)
    print("G1 Robot Human Motion Retargeting - Quick Start")
    print("=" * 80)
    
    # Setup paths
    smplx_model_path = f"{G1_INTERACTION_ROOT_DIR}/resources/smplx_models"
    robot_urdf_path = f"{G1_INTERACTION_ROOT_DIR}/resources/robots/g1/urdf/g1.urdf"
    
    print("\n[Step 1] Initializing SMPLX Retargeter...")
    retargeter = SMPLXRetargeter(
        smplx_model_path=smplx_model_path,
        robot_urdf_path=robot_urdf_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print("  ✓ Retargeter initialized")
    
    # Load or create motion data
    print("\n[Step 2] Loading human motion data...")
    motion_file = f"{G1_INTERACTION_ROOT_DIR}/resources/human_motions/walk_forward.npz"
    retargeter.load_motion_sequence(motion_file)
    print("  ✓ Motion loaded")
    
    # Get target poses at different times
    print("\n[Step 3] Retargeting human motion to robot...")
    num_samples = 10
    times = torch.linspace(0, 3, num_samples).to(retargeter.device)
    
    target_data = retargeter.get_target_pose(times, num_samples)
    
    print(f"  ✓ Generated {num_samples} target poses")
    print(f"    - Joint positions shape: {target_data['joint_positions'].shape}")
    print(f"    - Root position shape: {target_data['root_position'].shape}")
    print(f"    - Contact states shape: {target_data['contacts'].shape}")
    
    # Display sample retargeted pose
    print("\n[Step 4] Sample retargeted pose at t=1.5s:")
    sample_idx = num_samples // 2
    print(f"  Joint positions (first 6 DOFs):")
    for i, angle in enumerate(target_data['joint_positions'][sample_idx, :6]):
        print(f"    DOF {i}: {angle.item():.3f} rad ({np.degrees(angle.item()):.1f}°)")
    
    print(f"\n  Root position: {target_data['root_position'][sample_idx].cpu().numpy()}")
    print(f"  Contact states: {target_data['contacts'][sample_idx].cpu().numpy()}")
    
    # Demonstrate retargeting loss computation
    print("\n[Step 5] Computing retargeting loss...")
    robot_joint_pos = target_data['joint_positions'] + torch.randn_like(
        target_data['joint_positions']
    ) * 0.1
    robot_contacts = target_data['contacts']
    
    loss_dict = retargeter.compute_retargeting_loss(
        robot_joint_pos,
        target_data['joint_positions'],
        robot_contacts,
        target_data['contacts']
    )
    
    print(f"  Loss components:")
    for key, value in loss_dict.items():
        print(f"    {key}: {value.item():.4f}")
    
    print("\n" + "=" * 80)
    print("Quick start completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Train a policy: python -m g1_interaction.scripts.train --task=g1_retarget")
    print("  2. Test trained policy: python -m g1_interaction.scripts.play --task=g1_retarget")
    print("  3. Customize config: Edit g1_interaction/envs/g1/g1_retarget_config.py")
    

if __name__ == '__main__':
    main()

