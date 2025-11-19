#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""
Play/test script for trained G1 retargeting policy.

Usage:
    python -m g1_interaction.scripts.play --task=g1_retarget
    
Optional arguments:
    --load_run: Run folder to load (default: latest)
    --checkpoint: Checkpoint to load (default: latest)
"""

import numpy as np
import os
import torch

import isaacgym
from g1_interaction.envs import *
from g1_interaction.utils import get_args, task_registry, get_load_path


def play(args):
    """Play trained policy."""
    print("=" * 80)
    print("G1 Robot Human Motion Retargeting - Play Mode")
    print("=" * 80)
    
    # Create environment
    print(f"\n[1/3] Creating environment: {args.task}")
    args.num_envs = min(args.num_envs if hasattr(args, 'num_envs') and args.num_envs else 16, 16)
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    print(f"  - Number of environments: {env_cfg.env.num_envs}")
    
    # Load policy
    print(f"\n[2/3] Loading trained policy")
    _, train_cfg = task_registry.get_cfgs(args.task)
    
    log_root = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        '..',
        'logs',
        train_cfg.runner.experiment_name
    )
    
    try:
        model_path = get_load_path(
            log_root,
            load_run=getattr(args, 'load_run', -1),
            checkpoint=getattr(args, 'checkpoint', -1)
        )
        print(f"  - Loading from: {model_path}")
        
        # Load policy weights
        loaded_dict = torch.load(model_path, map_location=args.rl_device)
        
        # Create policy
        from rsl_rl.modules import ActorCritic
        policy_cfg = train_cfg.policy
        
        policy = ActorCritic(
            num_actor_obs=env_cfg.env.num_observations,
            num_critic_obs=env_cfg.env.num_observations,
            num_actions=env_cfg.env.num_actions,
            actor_hidden_dims=policy_cfg.actor_hidden_dims,
            critic_hidden_dims=policy_cfg.critic_hidden_dims,
            activation=policy_cfg.activation,
            init_noise_std=policy_cfg.init_noise_std
        ).to(args.rl_device)
        
        policy.load_state_dict(loaded_dict['model_state_dict'])
        policy.eval()
        
        print("  - Policy loaded successfully!")
        
    except Exception as e:
        print(f"  - Warning: Could not load policy: {e}")
        print("  - Running with random policy")
        policy = None
    
    # Run policy
    print(f"\n[3/3] Running policy")
    print("  - Press ESC to quit")
    print("-" * 80)
    
    obs, _ = env.reset()
    
    episode_length = 0
    episode_reward = 0
    
    while True:
        if policy is not None:
            with torch.no_grad():
                actions = policy.act_inference(obs)
        else:
            # Random actions
            actions = torch.randn_like(env.actions) * 0.1
        
        obs, privileged_obs, rewards, dones, infos = env.step(actions)
        
        episode_length += 1
        episode_reward += rewards.mean().item()
        
        # Print episode info
        if dones.any():
            avg_reward = episode_reward / episode_length
            print(f"Episode finished - Length: {episode_length}, Avg Reward: {avg_reward:.3f}")
            episode_length = 0
            episode_reward = 0
        
        # Check for quit
        if hasattr(env, 'viewer') and env.viewer:
            if env.gym.query_viewer_has_closed(env.viewer):
                break


if __name__ == '__main__':
    args = get_args()
    play(args)

