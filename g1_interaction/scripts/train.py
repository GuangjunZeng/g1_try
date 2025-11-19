#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""
Training script for G1 robot human motion retargeting.

Usage:
    python -m g1_interaction.scripts.train --task=g1_retarget
    
Optional arguments:
    --headless: Run without rendering
    --num_envs: Number of parallel environments
    --seed: Random seed
    --max_iterations: Maximum training iterations
"""

import numpy as np
import os
from datetime import datetime

import isaacgym
from g1_interaction.envs import *
from g1_interaction.utils import get_args, task_registry
import torch


def train(args):
    """Main training loop."""
    print("=" * 80)
    print("G1 Robot Human Motion Retargeting Training")
    print("=" * 80)
    
    # Create environment
    print(f"\n[1/3] Creating environment: {args.task}")
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    print(f"  - Number of environments: {env_cfg.env.num_envs}")
    print(f"  - Observation dimension: {env_cfg.env.num_observations}")
    print(f"  - Action dimension: {env_cfg.env.num_actions}")
    
    # Create PPO algorithm
    print(f"\n[2/3] Creating PPO algorithm")
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    print(f"  - Policy architecture: {train_cfg.policy.actor_hidden_dims}")
    print(f"  - Learning rate: {train_cfg.algorithm.learning_rate}")
    print(f"  - Max iterations: {train_cfg.runner.max_iterations}")
    
    # Start training
    print(f"\n[3/3] Starting training")
    print(f"  - Experiment: {train_cfg.runner.experiment_name}")
    print(f"  - Run name: {train_cfg.runner.run_name}")
    print(f"  - Device: {args.rl_device}")
    print("-" * 80)
    
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True
    )
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)


if __name__ == '__main__':
    args = get_args()
    train(args)

