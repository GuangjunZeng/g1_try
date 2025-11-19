# SPDX-License-Identifier: MIT

import numpy as np
import os
import torch
from torch import Tensor
from typing import Tuple, Dict
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from g1_interaction import G1_INTERACTION_ROOT_DIR
from g1_interaction.envs.base.base_config import BaseConfig
from g1_interaction.retargeting.smplx_retarget import SMPLXRetargeter
from g1_interaction.utils.math_utils import quat_apply_yaw, wrap_to_pi, quat_rotate_inverse


class RetargetRobot:
    """
    End-to-end contact-aware retargeting environment for SMPLX human motion to robot motion.
    """
    
    def __init__(self, cfg: BaseConfig, sim_params, physics_engine, sim_device, headless):
        """
        Initialize the retargeting environment.
        
        Args:
            cfg: Environment configuration
            sim_params: Simulation parameters
            physics_engine: Physics engine type
            sim_device: Device to run simulation on
            headless: Whether to run without rendering
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        self.headless = headless
        self.device = sim_device
        
        # Parse configuration
        self._parse_cfg(cfg)
        
        # Create Isaac Gym simulation
        self.gym = gymapi.acquire_gym()
        self.create_sim()
        
        # Initialize SMPLX retargeter
        self.smplx_retargeter = SMPLXRetargeter(
            smplx_model_path=cfg.retarget.smplx_model_path,
            robot_urdf_path=cfg.asset.file.format(G1_INTERACTION_ROOT_DIR=G1_INTERACTION_ROOT_DIR),
            device=self.device
        )
        
        # Set up viewer if not headless
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        else:
            self.viewer = None
            
        self.enable_viewer_sync = True
        self.debug_viz = False
        self.init_done = False
        
        # Initialize buffers
        self._init_buffers()
        self._prepare_reward_function()
        
        self.init_done = True

    def step(self, actions):
        """
        Step the simulation with given actions.
        
        Args:
            actions: Action tensor of shape (num_envs, num_actions)
            
        Returns:
            observations, privileged_observations, rewards, dones, extras
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        
        # Step physics and render
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            
        self.post_physics_step()
        
        # Return clipped observations
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
            
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """Update states and compute observations, rewards, resets."""
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.episode_length_buf += 1
        self.common_step_counter += 1
        
        # Update base quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        # Update human motion reference
        self._update_human_motion_reference()
        
        # Compute observations, rewards, resets
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()
        
        # Update history
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def compute_observations(self):
        """
        Compute observations including:
        - Robot state (base velocity, orientation, joint positions/velocities)
        - Target human pose from SMPLX
        - Contact information
        """
        # Robot proprioceptive observations
        robot_obs = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)
        
        # Target human pose observations (retargeted)
        target_joint_pos = self.target_joint_positions * self.obs_scales.dof_pos
        target_root_pos = self.target_root_positions * self.obs_scales.lin_vel
        
        # Contact observations
        contact_obs = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_obs = contact_obs.float()
        
        self.obs_buf = torch.cat((
            robot_obs,
            target_joint_pos,
            target_root_pos,
            contact_obs
        ), dim=-1)
        
        # Add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def compute_reward(self):
        """
        Compute rewards including:
        - Pose tracking reward
        - Contact matching reward  
        - Smoothness reward
        - Regularization terms
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
            
        # Add termination reward
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def check_termination(self):
        """Check if environments need to be reset."""
        # Terminate on body contact
        self.reset_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
            dim=1
        )
        # Terminate on timeout
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """Reset specified environments."""
        if len(env_ids) == 0:
            return
            
        # Reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        
        # Reset human motion playback
        self._reset_human_motion(env_ids)
        
        # Reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        
        # Fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]
            ) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
            
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def create_sim(self):
        """Create simulation, terrain, and environments."""
        self.up_axis_idx = 2  # z-up
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params
        )
        
        # Create ground plane
        self._create_ground_plane()
        
        # Create environments
        self._create_envs()

    def set_camera(self, position, lookat):
        """Set camera position and look-at target."""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def render(self):
        """Render the simulation."""
        if self.viewer:
            if self.gym.query_viewer_has_closed(self.viewer):
                exit()
            
            if self.enable_viewer_sync:
                self.gym.sync_frame_time(self.sim)
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
            else:
                self.gym.poll_viewer_events(self.viewer)

    # Private methods
    def _parse_cfg(self, cfg):
        """Parse configuration parameters."""
        self.dt = cfg.control.decimation * self.sim_params.dt
        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions
        
        self.obs_scales = cfg.normalization.obs_scales
        self.reward_scales = self._class_to_dict(cfg.rewards.scales)
        
        self.max_episode_length_s = cfg.env.episode_length_s
        self.max_episode_length = int(self.max_episode_length_s / self.dt)
        
        # Get device info
        self.graphics_device_id = self.sim_params.graphics_device_id
        self.sim_device_id = self.sim_params.sim_device_id

    def _init_buffers(self):
        """Initialize torch tensors for simulation states."""
        # Get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # Create wrapper tensors
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, -1, 13
        )
        
        # Initialize data buffers
        self.common_step_counter = 0
        self.extras = {}
        
        self.gravity_vec = to_torch(
            [0., 0., -1.], device=self.device
        ).repeat((self.num_envs, 1))
        
        self.torques = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, 
            device=self.device, requires_grad=False
        )
        self.p_gains = torch.zeros(
            self.num_actions, dtype=torch.float, 
            device=self.device, requires_grad=False
        )
        self.d_gains = torch.zeros(
            self.num_actions, dtype=torch.float,
            device=self.device, requires_grad=False
        )
        
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float,
            device=self.device, requires_grad=False
        )
        self.last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        # SMPLX retargeting buffers
        self.target_joint_positions = torch.zeros(
            self.num_envs, self.num_dof, dtype=torch.float,
            device=self.device, requires_grad=False
        )
        self.target_root_positions = torch.zeros(
            self.num_envs, 3, dtype=torch.float,
            device=self.device, requires_grad=False
        )
        self.target_contacts = torch.zeros(
            self.num_envs, 4, dtype=torch.bool,
            device=self.device, requires_grad=False
        )
        self.human_motion_time = torch.zeros(
            self.num_envs, dtype=torch.float,
            device=self.device, requires_grad=False
        )
        
        # Observation and reward buffers
        self.obs_buf = torch.zeros(
            self.num_envs, self.num_obs, dtype=torch.float,
            device=self.device, requires_grad=False
        )
        self.rew_buf = torch.zeros(
            self.num_envs, dtype=torch.float,
            device=self.device, requires_grad=False
        )
        self.reset_buf = torch.ones(
            self.num_envs, dtype=torch.long,
            device=self.device, requires_grad=False
        )
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long,
            device=self.device, requires_grad=False
        )
        self.time_out_buf = torch.zeros(
            self.num_envs, dtype=torch.bool,
            device=self.device, requires_grad=False
        )
        
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(
                self.num_envs, self.num_privileged_obs,
                dtype=torch.float, device=self.device, requires_grad=False
            )
        else:
            self.privileged_obs_buf = None
            
        # Joint PD gains and default positions
        self.default_dof_pos = torch.zeros(
            self.num_dof, dtype=torch.float,
            device=self.device, requires_grad=False
        )
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles.get(name, 0.)
            self.default_dof_pos[i] = angle
            
            # Set PD gains
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        
        # Noise
        self.add_noise = self.cfg.noise.add_noise
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

    def _prepare_reward_function(self):
        """Prepare reward function list."""
        # Remove zero scales
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
                
        # Prepare function list
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))
            
        # Episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float,
                            device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()
        }

    def _create_ground_plane(self):
        """Create ground plane."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        """Create robot environments."""
        asset_path = self.cfg.asset.file.format(
            G1_INTERACTION_ROOT_DIR=G1_INTERACTION_ROOT_DIR
        )
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        
        # Get body and DOF names
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long,
            device=self.device, requires_grad=False
        )
        
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
            
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
            
        # Create environments
        base_init_state_list = (
            self.cfg.init_state.pos + self.cfg.init_state.rot +
            self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        )
        
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        
        env_lower = gymapi.Vec3(-self.cfg.env.env_spacing, -self.cfg.env.env_spacing, 0.)
        env_upper = gymapi.Vec3(self.cfg.env.env_spacing, self.cfg.env.env_spacing, self.cfg.env.env_spacing)
        
        self.actor_handles = []
        self.envs = []
        
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            actor_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose,
                self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0
            )
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            
        # Store indices
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )
            
        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names), dtype=torch.long,
            device=self.device, requires_grad=False
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )
            
        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names), dtype=torch.long,
            device=self.device, requires_grad=False
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

    def _compute_torques(self, actions):
        """Compute torques from actions using PD control."""
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        
        if control_type == "P":
            torques = (
                self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) -
                self.d_gains * self.dof_vel
            )
        elif control_type == "V":
            torques = (
                self.p_gains * (actions_scaled - self.dof_vel) -
                self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
            )
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
            
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """Reset DOF positions and velocities."""
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
            0.8, 1.2, (len(env_ids), self.num_dof), device=self.device
        )
        self.dof_vel[env_ids] = 0.
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

    def _reset_root_states(self, env_ids):
        """Reset root state positions and velocities."""
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, 7:13] = 0.  # Zero velocities
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

    def _update_human_motion_reference(self):
        """Update target poses from SMPLX human motion."""
        self.human_motion_time += self.dt
        
        # Get target poses from SMPLX retargeter
        target_data = self.smplx_retargeter.get_target_pose(
            self.human_motion_time,
            self.num_envs
        )
        
        self.target_joint_positions = target_data['joint_positions']
        self.target_root_positions = target_data['root_position']
        self.target_contacts = target_data['contacts']

    def _reset_human_motion(self, env_ids):
        """Reset human motion playback for specified environments."""
        self.human_motion_time[env_ids] = 0.

    def _draw_debug_vis(self):
        """Draw debug visualizations."""
        pass

    def _get_noise_scale_vec(self, cfg):
        """Get noise scale vector for observations."""
        noise_vec = torch.zeros_like(self.obs_buf[0])
        # Simplified - should match observation structure
        return noise_vec

    def _class_to_dict(self, obj):
        """Convert class to dictionary."""
        if not hasattr(obj, "__dict__"):
            return obj
        result = {}
        for key in dir(obj):
            if key.startswith("_"):
                continue
            element = getattr(obj, key)
            if not callable(element) and not type(element).__name__.startswith("_"):
                result[key] = self._class_to_dict(element)
        return result

    # Reward functions
    def _reward_pose_tracking(self):
        """Reward for tracking target joint positions."""
        pose_error = torch.sum(
            torch.square(self.dof_pos - self.target_joint_positions), dim=1
        )
        return torch.exp(-pose_error / self.cfg.rewards.tracking_sigma)

    def _reward_contact_matching(self):
        """Reward for matching target contact states."""
        current_contacts = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_match = (current_contacts == self.target_contacts).float()
        return torch.mean(contact_match, dim=1)

    def _reward_smoothness(self):
        """Reward for smooth actions."""
        return -torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_torques(self):
        """Penalize large torques."""
        return -torch.sum(torch.square(self.torques), dim=1)

    def _reward_termination(self):
        """Terminal penalty."""
        return self.reset_buf * ~self.time_out_buf

    @property
    def torque_limits(self):
        """Get torque limits from config."""
        if not hasattr(self, '_torque_limits'):
            self._torque_limits = torch.ones(
                self.num_dof, dtype=torch.float, device=self.device
            ) * 100.0  # Default value
        return self._torque_limits

