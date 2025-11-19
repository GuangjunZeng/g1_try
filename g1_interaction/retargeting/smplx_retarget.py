# SPDX-License-Identifier: MIT

import torch
import numpy as np
from typing import Dict, Optional
import os


class SMPLXRetargeter:
    """
    Contact-aware retargeting from SMPLX human motion to robot motion.
    
    This module handles:
    1. Loading SMPLX human motion data
    2. Computing inverse kinematics mapping from human to robot
    3. Contact state detection and matching
    4. Real-time motion retargeting
    """
    
    def __init__(
        self,
        smplx_model_path: str,
        robot_urdf_path: str,
        device: str = 'cuda:0'
    ):
        """
        Initialize the SMPLX retargeter.
        
        Args:
            smplx_model_path: Path to SMPLX model files
            robot_urdf_path: Path to robot URDF file
            device: Device for computation
        """
        self.device = device
        self.smplx_model_path = smplx_model_path
        self.robot_urdf_path = robot_urdf_path
        
        # Load SMPLX model
        self._load_smplx_model()
        
        # Load robot model
        self._load_robot_model()
        
        # Build retargeting mapping
        self._build_retarget_mapping()
        
        # Motion data cache
        self.motion_data = None
        self.motion_fps = 30
        
    def _load_smplx_model(self):
        """Load SMPLX body model."""
        try:
            import smplx
            self.smplx_model = smplx.create(
                self.smplx_model_path,
                model_type='smplx',
                gender='neutral',
                use_face_contour=False,
                num_betas=10,
                num_expression_coeffs=10,
                ext='npz'
            ).to(self.device)
            print(f"SMPLX model loaded from {self.smplx_model_path}")
        except Exception as e:
            print(f"Warning: Could not load SMPLX model: {e}")
            print("Using dummy SMPLX model for testing")
            self.smplx_model = None
            
    def _load_robot_model(self):
        """Load robot kinematic model from URDF."""
        # For now, store robot DOF information
        # In production, would use a proper URDF parser
        self.robot_joints = {
            'left_hip_yaw': 0,
            'left_hip_roll': 1, 
            'left_hip_pitch': 2,
            'left_knee': 3,
            'left_ankle_pitch': 4,
            'left_ankle_roll': 5,
            'right_hip_yaw': 6,
            'right_hip_roll': 7,
            'right_hip_pitch': 8,
            'right_knee': 9,
            'right_ankle_pitch': 10,
            'right_ankle_roll': 11,
            'left_shoulder_pitch': 12,
            'left_shoulder_roll': 13,
            'left_shoulder_yaw': 14,
            'left_elbow': 15,
            'right_shoulder_pitch': 16,
            'right_shoulder_roll': 17,
            'right_shoulder_yaw': 18,
            'right_elbow': 19,
        }
        
        # SMPLX to robot joint mapping
        self.smplx_to_robot = {
            'left_hip': ['left_hip_yaw', 'left_hip_roll', 'left_hip_pitch'],
            'left_knee': ['left_knee'],
            'left_ankle': ['left_ankle_pitch', 'left_ankle_roll'],
            'right_hip': ['right_hip_yaw', 'right_hip_roll', 'right_hip_pitch'],
            'right_knee': ['right_knee'],
            'right_ankle': ['right_ankle_pitch', 'right_ankle_roll'],
            'left_shoulder': ['left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw'],
            'left_elbow': ['left_elbow'],
            'right_shoulder': ['right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw'],
            'right_elbow': ['right_elbow'],
        }
        
    def _build_retarget_mapping(self):
        """Build mapping from SMPLX joints to robot joints."""
        # This is a simplified version
        # In production, would compute proper kinematic retargeting
        
        # Key joint correspondences for lower body
        self.joint_mapping = {
            'pelvis': 'base',
            'left_hip': 'left_hip',
            'left_knee': 'left_knee', 
            'left_ankle': 'left_ankle',
            'left_foot': 'left_foot',
            'right_hip': 'right_hip',
            'right_knee': 'right_knee',
            'right_ankle': 'right_ankle',
            'right_foot': 'right_foot',
        }
        
        # Scaling factors (human to robot)
        self.scale_factors = {
            'height': 1.0,  # Adjust based on robot height
            'leg_length': 1.0,
            'arm_length': 1.0,
        }
        
    def load_motion_sequence(self, motion_file: str):
        """
        Load a motion sequence from file.
        
        Args:
            motion_file: Path to motion data file (npz format)
        """
        if not os.path.exists(motion_file):
            print(f"Warning: Motion file {motion_file} not found, using default pose")
            self._create_default_motion()
            return
            
        data = np.load(motion_file, allow_pickle=True)
        self.motion_data = {
            'body_pose': torch.tensor(data['body_pose'], device=self.device),
            'global_orient': torch.tensor(data['global_orient'], device=self.device),
            'transl': torch.tensor(data['transl'], device=self.device),
            'betas': torch.tensor(data.get('betas', np.zeros(10)), device=self.device),
        }
        self.motion_length = len(self.motion_data['body_pose'])
        self.motion_fps = data.get('fps', 30)
        
        print(f"Loaded motion sequence: {self.motion_length} frames at {self.motion_fps} fps")
        
    def _create_default_motion(self):
        """Create a default standing motion for testing."""
        num_frames = 100
        self.motion_data = {
            'body_pose': torch.zeros(num_frames, 63, device=self.device),
            'global_orient': torch.zeros(num_frames, 3, device=self.device),
            'transl': torch.zeros(num_frames, 3, device=self.device),
            'betas': torch.zeros(10, device=self.device),
        }
        self.motion_length = num_frames
        self.motion_fps = 30
        
    def get_target_pose(
        self,
        time: torch.Tensor,
        num_envs: int
    ) -> Dict[str, torch.Tensor]:
        """
        Get target robot pose at given time.
        
        Args:
            time: Current time for each environment (num_envs,)
            num_envs: Number of parallel environments
            
        Returns:
            Dictionary containing:
                - joint_positions: Target joint positions (num_envs, num_dof)
                - root_position: Target root position (num_envs, 3)
                - contacts: Target contact states (num_envs, 4) for 4 end-effectors
        """
        if self.motion_data is None:
            self._create_default_motion()
            
        # Compute frame indices
        frame_indices = (time * self.motion_fps).long() % self.motion_length
        
        # Get SMPLX poses for current frames
        body_pose = self.motion_data['body_pose'][frame_indices]
        global_orient = self.motion_data['global_orient'][frame_indices]
        transl = self.motion_data['transl'][frame_indices]
        
        # Retarget to robot joint positions
        joint_positions = self._retarget_pose(
            body_pose, global_orient, transl
        )
        
        # Estimate contact states from motion
        contacts = self._estimate_contacts(
            body_pose, global_orient, transl
        )
        
        # Extract root position
        root_position = transl
        
        return {
            'joint_positions': joint_positions,
            'root_position': root_position,
            'contacts': contacts
        }
        
    def _retarget_pose(
        self,
        body_pose: torch.Tensor,
        global_orient: torch.Tensor,
        transl: torch.Tensor
    ) -> torch.Tensor:
        """
        Retarget SMPLX pose to robot joint angles.
        
        This is a simplified implementation. In production, would use:
        1. Forward kinematics on SMPLX
        2. Inverse kinematics on robot
        3. Contact-aware optimization
        
        Args:
            body_pose: SMPLX body pose parameters (num_envs, 63)
            global_orient: Global orientation (num_envs, 3)
            transl: Translation (num_envs, 3)
            
        Returns:
            Robot joint positions (num_envs, num_dof)
        """
        num_envs = body_pose.shape[0]
        
        if self.smplx_model is not None:
            # Use SMPLX forward kinematics
            smplx_output = self.smplx_model(
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
                return_verts=True
            )
            
            # Extract joint positions
            joints = smplx_output.joints
            
            # Perform IK retargeting (simplified)
            joint_angles = self._compute_ik(joints)
        else:
            # Dummy retargeting for testing
            joint_angles = self._dummy_retarget(body_pose)
            
        return joint_angles
        
    def _compute_ik(self, smplx_joints: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse kinematics from SMPLX joint positions to robot joint angles.
        
        Args:
            smplx_joints: SMPLX joint positions (num_envs, num_joints, 3)
            
        Returns:
            Robot joint angles (num_envs, num_dof)
        """
        num_envs = smplx_joints.shape[0]
        
        # Simplified IK - in production would use proper IK solver
        # For now, extract relevant joints and compute approximate angles
        
        # Extract key joints (indices based on SMPLX joint ordering)
        pelvis = smplx_joints[:, 0]  # Pelvis
        left_hip = smplx_joints[:, 1]
        right_hip = smplx_joints[:, 2]
        left_knee = smplx_joints[:, 4]
        right_knee = smplx_joints[:, 5]
        left_ankle = smplx_joints[:, 7]
        right_ankle = smplx_joints[:, 8]
        
        # Compute joint angles from positions
        # This is a placeholder - real implementation would use analytical or numerical IK
        joint_angles = torch.zeros(num_envs, 20, device=self.device)
        
        # Left leg
        joint_angles[:, 0] = 0.0  # left_hip_yaw
        joint_angles[:, 1] = 0.0  # left_hip_roll
        joint_angles[:, 2] = self._compute_hip_pitch(pelvis, left_hip, left_knee)
        joint_angles[:, 3] = self._compute_knee_angle(left_hip, left_knee, left_ankle)
        joint_angles[:, 4] = 0.0  # left_ankle_pitch
        joint_angles[:, 5] = 0.0  # left_ankle_roll
        
        # Right leg
        joint_angles[:, 6] = 0.0  # right_hip_yaw
        joint_angles[:, 7] = 0.0  # right_hip_roll
        joint_angles[:, 8] = self._compute_hip_pitch(pelvis, right_hip, right_knee)
        joint_angles[:, 9] = self._compute_knee_angle(right_hip, right_knee, right_ankle)
        joint_angles[:, 10] = 0.0  # right_ankle_pitch
        joint_angles[:, 11] = 0.0  # right_ankle_roll
        
        return joint_angles
        
    def _compute_hip_pitch(
        self,
        pelvis: torch.Tensor,
        hip: torch.Tensor,
        knee: torch.Tensor
    ) -> torch.Tensor:
        """Compute hip pitch angle from positions."""
        vec = knee - hip
        angle = torch.atan2(vec[:, 2], torch.norm(vec[:, :2], dim=1))
        return angle
        
    def _compute_knee_angle(
        self,
        hip: torch.Tensor,
        knee: torch.Tensor,
        ankle: torch.Tensor
    ) -> torch.Tensor:
        """Compute knee angle from positions."""
        upper = knee - hip
        lower = ankle - knee
        
        # Compute angle between vectors
        dot = torch.sum(upper * lower, dim=1)
        upper_norm = torch.norm(upper, dim=1)
        lower_norm = torch.norm(lower, dim=1)
        
        cos_angle = dot / (upper_norm * lower_norm + 1e-8)
        angle = torch.acos(torch.clip(cos_angle, -1.0, 1.0))
        
        return np.pi - angle
        
    def _dummy_retarget(self, body_pose: torch.Tensor) -> torch.Tensor:
        """Dummy retargeting for testing without SMPLX."""
        num_envs = body_pose.shape[0]
        
        # Generate simple walking motion
        joint_angles = torch.zeros(num_envs, 20, device=self.device)
        
        # Simple sinusoidal motion for testing
        phase = torch.linspace(0, 2*np.pi, num_envs, device=self.device)
        
        # Hip pitch
        joint_angles[:, 2] = 0.3 * torch.sin(phase)  # left
        joint_angles[:, 8] = -0.3 * torch.sin(phase)  # right
        
        # Knee
        joint_angles[:, 3] = 0.6 * torch.abs(torch.sin(phase))  # left
        joint_angles[:, 9] = 0.6 * torch.abs(torch.cos(phase))  # right
        
        return joint_angles
        
    def _estimate_contacts(
        self,
        body_pose: torch.Tensor,
        global_orient: torch.Tensor,
        transl: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate contact states from motion.
        
        Args:
            body_pose: SMPLX body pose
            global_orient: Global orientation
            transl: Translation
            
        Returns:
            Contact states (num_envs, 4) for [left_foot, right_foot, left_hand, right_hand]
        """
        num_envs = body_pose.shape[0]
        
        if self.smplx_model is not None:
            # Use SMPLX to get foot positions
            smplx_output = self.smplx_model(
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
                return_verts=True
            )
            
            joints = smplx_output.joints
            
            # Get foot heights
            left_foot_height = joints[:, 7, 2]  # Left ankle Z
            right_foot_height = joints[:, 8, 2]  # Right ankle Z
            
            # Contact if below threshold
            contact_threshold = 0.1
            left_contact = left_foot_height < contact_threshold
            right_contact = right_foot_height < contact_threshold
            
            # For now, no hand contacts
            contacts = torch.stack([
                left_contact,
                right_contact,
                torch.zeros(num_envs, dtype=torch.bool, device=self.device),
                torch.zeros(num_envs, dtype=torch.bool, device=self.device)
            ], dim=1)
        else:
            # Dummy contacts - alternate feet
            contacts = torch.zeros(num_envs, 4, dtype=torch.bool, device=self.device)
            contacts[:num_envs//2, 0] = True  # Left foot
            contacts[num_envs//2:, 1] = True  # Right foot
            
        return contacts
        
    def compute_retargeting_loss(
        self,
        robot_joint_pos: torch.Tensor,
        target_joint_pos: torch.Tensor,
        robot_contacts: torch.Tensor,
        target_contacts: torch.Tensor,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute retargeting loss for training.
        
        Args:
            robot_joint_pos: Current robot joint positions (num_envs, num_dof)
            target_joint_pos: Target joint positions (num_envs, num_dof)
            robot_contacts: Current robot contacts (num_envs, 4)
            target_contacts: Target contacts (num_envs, 4)
            weights: Loss term weights
            
        Returns:
            Dictionary of loss terms
        """
        if weights is None:
            weights = {
                'pose': 1.0,
                'contact': 0.5,
            }
            
        # Pose loss
        pose_loss = torch.mean(
            torch.sum(torch.square(robot_joint_pos - target_joint_pos), dim=1)
        )
        
        # Contact loss
        contact_loss = torch.mean(
            torch.sum((robot_contacts.float() - target_contacts.float())**2, dim=1)
        )
        
        # Total loss
        total_loss = (
            weights['pose'] * pose_loss +
            weights['contact'] * contact_loss
        )
        
        return {
            'total': total_loss,
            'pose': pose_loss,
            'contact': contact_loss,
        }

