# SPDX-License-Identifier: MIT

import os
import random
import numpy as np
import torch
from typing import Tuple
from isaacgym import gymapi, gymutil


def get_args():
    """Parse command line arguments for Isaac Gym."""
    custom_parameters = [
        {"name": "--task", "type": str, "default": "g1_retarget", "help": "Task name"},
        {"name": "--resume", "action": "store_true", "help": "Resume training from checkpoint"},
        {"name": "--experiment_name", "type": str, "help": "Experiment name"},
        {"name": "--run_name", "type": str, "help": "Run name"},
        {"name": "--load_run", "type": str, "default": "-1", "help": "Run to load for resume"},
        {"name": "--checkpoint", "type": int, "default": -1, "help": "Checkpoint to load"},
        {"name": "--num_envs", "type": int, "help": "Number of environments"},
        {"name": "--seed", "type": int, "help": "Random seed"},
        {"name": "--max_iterations", "type": int, "help": "Maximum training iterations"},
    ]
    
    args = gymutil.parse_arguments(
        description="G1 Robot Human Motion Retargeting",
        custom_parameters=custom_parameters
    )
    
    return args


def update_cfg_from_args(env_cfg, train_cfg, args):
    """Update config from command line arguments."""
    if env_cfg is not None:
        if args.seed is not None:
            env_cfg.seed = args.seed
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
            
    if train_cfg is not None:
        if args.seed is not None:
            train_cfg.seed = args.seed
        if args.experiment_name is not None:
            train_cfg.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            train_cfg.runner.run_name = args.run_name
        if args.resume:
            train_cfg.runner.resume = True
        if args.load_run != "-1":
            train_cfg.runner.load_run = args.load_run
        if args.checkpoint != -1:
            train_cfg.runner.checkpoint = args.checkpoint
        if args.max_iterations is not None:
            train_cfg.runner.max_iterations = args.max_iterations
            
    return env_cfg, train_cfg


def class_to_dict(obj) -> dict:
    """Convert a class to dictionary recursively."""
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = getattr(obj, key)
        if not callable(element) and not type(element).__name__.startswith("_"):
            result[key] = class_to_dict(element)
    return result


def get_load_path(root, load_run=-1, checkpoint=-1):
    """Get path to model checkpoint."""
    if load_run == -1:
        # Get most recent run
        runs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        if len(runs) == 0:
            raise ValueError(f"No runs found in {root}")
        runs.sort()
        load_run = runs[-1]
    
    run_dir = os.path.join(root, load_run)
    
    if checkpoint == -1:
        # Get most recent checkpoint
        models = [f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pt")]
        if len(models) == 0:
            raise ValueError(f"No checkpoints found in {run_dir}")
        models.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        checkpoint_file = models[-1]
    else:
        checkpoint_file = f"model_{checkpoint}.pt"
    
    return os.path.join(run_dir, checkpoint_file)


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def parse_sim_params(args, cfg):
    """Parse simulation parameters."""
    # Create sim params
    sim_params = gymapi.SimParams()
    
    # Set common parameters
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)
    
    # Override with args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = args.num_threads if hasattr(args, 'num_threads') else 10
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes if hasattr(args, 'subscenes') else 0
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
    
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.dt = cfg["sim"]["dt"] if "sim" in cfg else 0.005
    sim_params.substeps = cfg["sim"]["substeps"] if "sim" in cfg else 1
    
    if "gravity" in cfg.get("sim", {}):
        sim_params.gravity = gymapi.Vec3(*cfg["sim"]["gravity"])
    
    return sim_params


def torch_rand_float(lower, upper, shape, device):
    """Generate random float tensor."""
    return (upper - lower) * torch.rand(*shape, device=device) + lower

