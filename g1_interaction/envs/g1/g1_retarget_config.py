# SPDX-License-Identifier: MIT

from g1_interaction.envs.base.base_config import BaseConfig


class G1RetargetCfg(BaseConfig):
    """Configuration for G1 humanoid robot retargeting task."""
    
    class env(BaseConfig.env):
        num_envs = 4096
        num_observations = 300  # Robot state + target pose + contacts
        num_privileged_obs = None
        num_actions = 20  # G1 has 20 DOFs (lower + upper body)
        env_spacing = 3.
        send_timeouts = True
        episode_length_s = 20

    class terrain:
        mesh_type = 'plane'  # plane for now, can add complex terrain later
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0

    class retarget:
        # Path to SMPLX model (users need to download separately)
        smplx_model_path = "{G1_INTERACTION_ROOT_DIR}/resources/smplx_models"
        
        # Motion retargeting parameters
        use_contact_aware = True
        contact_threshold = 0.1  # meters
        
        # IK parameters
        ik_iterations = 100
        ik_tolerance = 1e-4
        
        # Retargeting weights
        position_weight = 1.0
        orientation_weight = 0.5
        contact_weight = 2.0

    class init_state:
        pos = [0.0, 0.0, 0.75]  # x,y,z [m] - G1 standing height
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        
        # Default joint angles for G1 robot (standing pose)
        default_joint_angles = {
            # Lower body
            "left_hip_yaw": 0.0,
            "left_hip_roll": 0.0,
            "left_hip_pitch": 0.0,
            "left_knee": 0.0,
            "left_ankle_pitch": 0.0,
            "left_ankle_roll": 0.0,
            "right_hip_yaw": 0.0,
            "right_hip_roll": 0.0,
            "right_hip_pitch": 0.0,
            "right_knee": 0.0,
            "right_ankle_pitch": 0.0,
            "right_ankle_roll": 0.0,
            # Upper body
            "left_shoulder_pitch": 0.0,
            "left_shoulder_roll": 0.0,
            "left_shoulder_yaw": 0.0,
            "left_elbow": 0.0,
            "right_shoulder_pitch": 0.0,
            "right_shoulder_roll": 0.0,
            "right_shoulder_yaw": 0.0,
            "right_elbow": 0.0,
        }

    class control:
        control_type = 'P'  # P: position, V: velocity, T: torques
        
        # PD Drive parameters for G1
        stiffness = {
            # Lower body - higher stiffness for legs
            'hip_yaw': 150.0,
            'hip_roll': 150.0,
            'hip_pitch': 200.0,
            'knee': 200.0,
            'ankle_pitch': 40.0,
            'ankle_roll': 40.0,
            # Upper body - moderate stiffness
            'shoulder_pitch': 80.0,
            'shoulder_roll': 80.0,
            'shoulder_yaw': 80.0,
            'elbow': 60.0,
        }
        
        damping = {
            # Lower body
            'hip_yaw': 5.0,
            'hip_roll': 5.0,
            'hip_pitch': 6.0,
            'knee': 6.0,
            'ankle_pitch': 2.0,
            'ankle_roll': 2.0,
            # Upper body
            'shoulder_pitch': 3.0,
            'shoulder_roll': 3.0,
            'shoulder_yaw': 3.0,
            'elbow': 2.0,
        }
        
        action_scale = 0.5
        decimation = 4  # Number of control steps per policy step

    class asset:
        file = "{G1_INTERACTION_ROOT_DIR}/resources/robots/g1/urdf/g1.urdf"
        name = "g1_robot"
        foot_name = "ankle"  # Used to identify foot bodies
        
        # Bodies where contact is penalized
        penalize_contacts_on = ["torso", "hip"]
        
        # Bodies where contact causes termination
        terminate_after_contacts_on = ["torso"]
        
        # Asset loading parameters
        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = False
        default_dof_drive_mode = 3  # 3: effort control
        self_collisions = 1  # 0: enable, 1: disable
        replace_cylinder_with_capsule = True
        flip_visual_attachments = True
        
        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-2., 2.]  # kg
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 0.5

    class rewards:
        class scales:
            # Main retargeting rewards
            pose_tracking = 2.0
            contact_matching = 1.5
            root_tracking = 1.0
            
            # Regularization
            smoothness = 0.5
            torques = -0.0001
            dof_vel = -0.0001
            dof_acc = -2.5e-7
            action_rate = -0.01
            
            # Penalties
            termination = -1.0
            collision = -0.5
            orientation = -0.2
            lin_vel_z = -0.5
            ang_vel_xy = -0.05
            
        only_positive_rewards = False
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.75
        max_contact_force = 500.0

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 10.

    class noise:
        add_noise = True
        noise_level = 1.0
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class viewer(BaseConfig.viewer):
        ref_env = 0
        pos = [5, 5, 3]
        lookat = [0, 0, 1]


class G1RetargetCfgPPO(BaseConfig):
    """PPO training configuration for G1 retargeting."""
    
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
        
    class algorithm:
        # Training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.e-3
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 3000
        
        # Logging
        save_interval = 100
        experiment_name = 'g1_retarget'
        run_name = ''
        
        # Load and resume
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

