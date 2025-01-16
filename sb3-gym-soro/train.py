"""
Train script using sb3-gym-template, without using Domain Randomization.

Examples:
    [Training]
    python train.py --env trunkcube-v0 --algo ppo --now 1 --seed 0 -t 2000000 \
        --run_path ./runs/no-DR --wandb_mode disabled
"""

from pprint import pprint
import argparse
import sys
import socket
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import torch
import wandb

from stable_baselines3.common.vec_env import SubprocVecEnv, make_vec_env

from utils.utils import *
from policy.policy import Policy

from sofagym import *


def main():
    """
    Main function to parse arguments, initialize WandB, create environments, and
    train an RL model using Stable-Baselines3. Resumes training if specified.
    """
    assert args.env is not None, "You must specify an environment via '--env'."
    # 'resume' XOR ('resume_path' and 'resume_wandb') must match
    assert not (args.resume ^ (args.resume_path is not None and args.resume_wandb is not None)), \
        "Inconsistent arguments for resume: either specify none or both 'resume_path'/'resume_wandb'."

    if args.test_env is None:
        # Use the same environment for testing if no separate env was provided
        args.test_env = args.env

    # Set the number of Torch threads
    torch.set_num_threads(args.now)

    # Display argument values for clarity
    pprint(vars(args))

    # Set the random seed
    set_seed(args.seed)

    # If resuming, extract the run ID from resume_path; else generate a random string
    resume_string = re.findall("_([^_]+)?$", args.resume_path)[0] if args.resume_path is not None else None
    random_string = get_random_string(5) if not args.resume else resume_string

    # Determine the WandB run ID
    run_id = args.resume_wandb if (args.resume and args.wandb_mode == "online") else wandb.util.generate_id()

    # Create a path for saving results
    if args.run_path is not None:
        run_path = (
            f"{args.run_path}/runs/{args.env}/"
            f"{get_run_name(args)}_{random_string}/"
        )
    else:
        run_path = f"runs/{args.env}/{get_run_name(args)}_{random_string}/"
    create_dirs(run_path)

    # Initialize WandB for experiment tracking
    wandb.init(
        config=vars(args),
        id=run_id,
        dir=run_path,
        project="SoRo-RL",
        group=(args.env + "_train" if args.group is None else args.group),
        name=f"{args.algo}_seed{args.seed}_{random_string}",
        save_code=True,
        notes=args.notes,
        mode=args.wandb_mode,
        resume="allow",
    )

    # Save initial metadata if not resuming
    if not args.resume:
        wandb.config.path = run_path
        wandb.config.hostname = socket.gethostname()

    # Create the vectorized training environment
    env = make_vec_env(
        args.env, 
        n_envs=args.now, 
        seed=args.seed, 
        vec_env_cls=SubprocVecEnv
        # If domain randomization or custom vec env is needed, we could use:
        # vec_env_cls=RandomSubprocVecEnv
    )

    # Build the network architecture
    size_layer = [args.n_neurons] * args.n_layers

    # If resuming from a checkpoint:
    if args.resume:
        # Retrieve the file from logs/ subdirectory in resume_path
        ckpt_list = os.listdir(os.path.join(args.resume_path, "logs"))
        assert ckpt_list, "No checkpoint found in the specified logs directory."
        ckpt = ckpt_list[0]
        load_path = os.path.join(args.resume_path, "logs", ckpt)
        assert os.path.exists(load_path), "model_ckpt_*_steps.zip hasn't been found."

        # Load the existing policy
        policy = Policy(
            algo=args.algo,
            env=env,
            lr=args.lr,
            batch_size=args.batch_size,
            size_layer=size_layer,
            device=args.device,
            seed=args.seed,
            load_from_pathname=load_path,
            reset_num_timesteps=False
        )
        n_previous_steps = policy.model.num_timesteps
        policy.model.num_timesteps = n_previous_steps - args.eval_freq
        print(f"Checkpoint model loaded. Resuming from step {policy.model.num_timesteps}.")
        remaining_steps = args.timesteps - policy.model.num_timesteps
        print(f"Training for additional {remaining_steps} steps.")

        if remaining_steps <= 0:
            print("\nAll requested training steps completed. Exiting.")
            sys.exit(0)

        timesteps_effective = remaining_steps
    else:
        # Create a new policy from scratch
        policy = Policy(
            algo=args.algo,
            env=env,
            lr=args.lr,
            batch_size=args.batch_size,
            size_layer=size_layer,
            device=args.device,
            seed=args.seed
        )
        timesteps_effective = args.timesteps

    print("--- Policy training start ---")
    # Train the model
    mean_reward, std_reward, best_policy, which_one = policy.train(
        timesteps=timesteps_effective,
        stopAtRewardThreshold=args.reward_threshold,
        n_eval_episodes=args.eval_episodes,
        eval_freq=args.eval_freq,
        best_model_save_path=run_path,
        return_best_model=True,
        save_freq=(int(args.save_freq) if args.save_freq is not None else None),
        keep_prev_ckpt=args.keep_prev_ckpt
    )

    # Save the final model
    policy.save_state_dict(os.path.join(run_path, "final_model.pth"))
    policy.save_full_state(os.path.join(run_path, "final_full_state.zip"))
    print("--- Policy training done ---")

    # Log training results
    print("\n\nMean reward:", mean_reward, "Std reward:", std_reward)
    wandb.run.summary["train_mean_reward"] = mean_reward
    wandb.run.summary["train_std_reward"] = std_reward
    wandb.run.summary["which_best_model"] = which_one

    # Save the best model
    torch.save(best_policy, os.path.join(run_path, "overall_best.pth"))
    wandb.save(os.path.join(run_path, "overall_best.pth"))

    # Evaluation on the test environment
    print("\n\n--- TEST DOMAIN EVALUATION ---")
    test_env = make_vec_env(
        args.test_env, 
        n_envs=args.now, 
        seed=args.seed, 
        vec_env_cls=SubprocVecEnv
    )
    policy = Policy(
        algo=args.algo,
        env=test_env,
        device=args.device,
        seed=args.seed,
        lr=args.lr,
        batch_size=args.batch_size,
        size_layer=size_layer
    )
    policy.load_state_dict(best_policy)

    mean_reward, std_reward = policy.eval(n_eval_episodes=args.test_episodes, render=args.test_render)
    print("Test reward:", mean_reward, "Std reward:", std_reward)

    # Log test results
    wandb.run.summary["test_mean_reward"] = mean_reward
    wandb.run.summary["test_std_reward"] = std_reward

    wandb.finish()


def parse_args():
    """
    Parse command-line arguments for training configuration.
    """
    parser = argparse.ArgumentParser()

    # Environment arguments
    parser.add_argument('--env', default='trunkcube-v0', type=str, help='Training environment name')
    parser.add_argument('--test_env', default=None, type=str, help='Testing environment name')

    # RL algorithm settings
    parser.add_argument('--algo', default='ppo', type=str, help='RL Algorithm to use (ppo, sac)')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=144, type=int, help='Batch size')

    # Network architecture
    parser.add_argument('--n_layers', default=3, type=int, help='Number of hidden layers')
    parser.add_argument('--n_neurons', default=512, type=int, help='Neurons in each hidden layer')

    # Parallelization
    parser.add_argument('--now', default=1, type=int, help='Number of CPU processes')
    
    # Training duration
    parser.add_argument('--timesteps', '-t', default=2000000, type=int, help='Total training timesteps')
    parser.add_argument('--reward_threshold', default=False, action='store_true', 
                        help='Stop training upon reaching a certain reward threshold')

    # Evaluation settings
    parser.add_argument('--eval_freq', default=10000, type=int, help='Timesteps between evaluations')
    parser.add_argument('--eval_episodes', default=50, type=int, help='Number of episodes per eval')
    parser.add_argument('--test_episodes', default=100, type=int, help='Number of episodes for final test')
    parser.add_argument('--test_render', default=False, action='store_true', help='Render the test episodes')

    # Reproducibility
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    # Compute device
    parser.add_argument('--device', default='cpu', type=str, help='Compute device: <cpu|cuda>')

    # Misc logging
    parser.add_argument('--verbose', default=0, type=int, help='Verbosity level (0,1,2)')
    parser.add_argument('--notes', default=None, type=str, help='Notes for WandB')
    parser.add_argument('--wandb_mode', default='online', type=str, help='WandB mode: online|offline|disabled')
    parser.add_argument('--group', default=None, type=str, help='WandB group name')

    # Resume training
    parser.add_argument('--resume', default=False, action='store_true', help='Resume training from a previous checkpoint')
    parser.add_argument('--resume_path', default=None, type=str, help='Path to the checkpoint directory for resume')
    parser.add_argument('--resume_wandb', default=None, type=str, help='WandB run ID to resume from')

    # Output paths
    parser.add_argument('--run_path', default=None, type=str, help='Base path to save run results')

    # Checkpoint saving
    parser.add_argument('--save_freq', default=None, type=str, 
                        help='Timesteps frequency for saving model checkpoints')
    parser.add_argument('--keep_prev_ckpt', default=False, action='store_true',
                        help='Keep previous checkpoints instead of overwriting')

    return parser.parse_args()


args = parse_args()

if __name__ == '__main__':
    main()
