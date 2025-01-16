# -*- coding: utf-8 -*-
"""
Test the MultiGaitRobotEnv by learning a policy to move in the x direction.

Usage:
-----
    python3.7 rl_multigait.py
"""

__authors__ = "emenager, pschegg"
__contact__ = "etienne.menager@ens-rennes.fr, pierre.schegg@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020,Inria"
__date__ = "Nov 10 2020"

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
# from AVEC.stable_baselines import PPO2
# from AVEC.stable_baselines.sac import SAC as SAC_AVEC

import gym

import sys
import os
import json
import pathlib
import numpy as np
import torch
import random
import argparse
import time
import pdb

# Insert the parent and current directories to the system path,
# so the sofagym modules can be imported properly
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()) + "/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

def load_environment(id, rank, seed=0):
    """
    Create a function that returns an environment initializer.

    Args:
        id (str): Environment ID (e.g., 'multigaitrobot-v0').
        rank (int): Index (used when creating multiple environments).
        seed (int): Seed for reproducibility.

    Returns:
        function: A function that initializes and returns the environment.
    """
    def _init():
        __import__('sofagym')  # Dynamically load the sofagym module
        env = gym.make(id)     # Create the environment from the specified ID
        env.seed(seed + rank)  # Set the random seed
        env.reset()            # Reset the environment to its initial state
        return env

    return _init

def test(env, model, epoch, n_test=1, render=False):
    """
    Test the trained model on the given environment.

    Args:
        env: The gym environment used for testing.
        model: The trained RL model to be tested.
        epoch (int): Current epoch index (for logging).
        n_test (int): Number of test rollouts to run.
        render (bool): If True, call env.render() during testing.

    Returns:
        (float, float): Tuple containing the mean episode reward and
                        the mean final reward across tests.
    """
    if render:
        env.config.update({"render": 2})  # Enable rendering if specified
    r, final_r = 0, 0
    for t in range(n_test):
        print("Start >> Epoch", epoch, "- Test", t)
        obs = env.reset()
        if render:
            env.render()
        rewards = []
        done = False
        step_id = 0  # renamed 'id' -> 'step_id' to avoid confusion with env ID
        while not done:
            # Predict the action given the current observation
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            if render:
                print("Test", t, "- Step", step_id, "- Action:", action, 
                      "- Reward:", reward)
                env.render()
            rewards.append(reward)
            step_id += 1
        print("Done >> Test", t, "- Rewards:", rewards, 
              "- Sum reward:", sum(rewards))
        r += sum(rewards)
        final_r += reward
    # Compute average rewards across all tests
    print("[INFO]  >> Mean reward:", r/n_test, " - Mean final reward:", final_r/n_test)
    return r/n_test, final_r/n_test

def sec_to_hours(seconds):
    """
    Convert the given time in seconds to a string in the format:
    '{hours} hours {minutes} mins {seconds} seconds'.

    Args:
        seconds (int): The number of seconds.

    Returns:
        list: A one-element list with the formatted time string.
    """
    h = str(seconds // 3600)
    m = str((seconds % 3600) // 60)
    s = str((seconds % 3600) % 60)
    return [f"{h} hours {m} mins {s} seconds"]

class Env:
    """
    A simple container class for environment-related parameters.
    
    Attributes:
        id (int): An index representing the environment.
        name (str): Gym environment ID used by 'gym.make()'.
        timer_limit (int): Maximum time steps (or similar) for one episode.
        continues (bool): Whether the action space is continuous.
        n_epochs (int): Maximum number of training epochs.
        gamma (float): Discount factor for RL algorithms.
        learning_rate (float): The learning rate.
        value_coeff (float): Value function coefficient (used in some algos).
        batch_size (int): Batch size for RL training.
        size_layer (list): A list specifying the hidden-layer sizes.
    """
    def __init__(self, id, name, timer_limit, continues, n_epochs, 
                 gamma, learning_rate, value_coeff, batch_size, size_layer):
        self.id = id
        self.name = name
        self.timer_limit = timer_limit
        self.continues = continues
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.value_coeff = value_coeff
        self.batch_size = batch_size
        self.size_layer = size_layer

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-ne", "--num_env", type=int, required=True,
                        help="Number/index of the environment")
    parser.add_argument("-na", "--num_algo", type=int, required=True,
                        help="Number/index of the algorithm")
    parser.add_argument("-nc", "--num_cpu", type=int,
                        help="Number of CPU processes")
    parser.add_argument("-s", "--seed", type=int, required=True,
                        help="Random seed for reproducibility")
    parser.add_argument("-train", "--training", action="store_true", default=False,
                        help="Enable training mode")
    parser.add_argument("-test", "--testing", action="store_true", default=False,
                        help="Enable testing mode")
    parser.add_argument("-restart", "--restart", type=int, default=0,
                        help="Restart training from the specified epoch index")

    args = parser.parse_args()

    # Ensure only one mode is active (training or testing)
    if args.training == args.testing:
        print("[ERROR] >> Pass only one between -train or -test")
        exit(0)

    # Create a dictionary of possible environments
    env_dict = {
        0: Env(0, 'cartstemcontact-v0', timer_limit=30, continues=True, n_epochs=600, 
               gamma=0.99, learning_rate=1e-4, value_coeff=0, batch_size=200, 
               size_layer=[512, 512, 512]),
        1: Env(1, 'cartstem-v0', timer_limit=80, continues=False, n_epochs=200, 
               gamma=0.99, learning_rate=1e-4, value_coeff=0, batch_size=256, 
               size_layer=[512, 512, 512]),
        2: Env(2, 'stempendulum-v0', timer_limit=50, continues=True, n_epochs=10001, 
               gamma=0.99, learning_rate=1e-4, value_coeff=0, batch_size=64, 
               size_layer=[512, 512, 512]),
        3: Env(3, 'catchtheobject-v0', timer_limit=30, continues=True, n_epochs=200, 
               gamma=0.99, learning_rate=1e-4, value_coeff=0, batch_size=256, 
               size_layer=[512, 512, 512]),
        4: Env(4, 'multigaitrobot-v0', timer_limit=18, continues=False, n_epochs=600, 
               gamma=0.99, learning_rate=1e-4, value_coeff=0, batch_size=144, 
               size_layer=[512, 512, 512]),
        5: Env(5, 'trunk-v0', timer_limit=18, continues=False, n_epochs=600, 
               gamma=0.99, learning_rate=1e-4, value_coeff=0, batch_size=144, 
               size_layer=[512, 512, 512]),
        6: Env(6, 'gripper-v0', timer_limit=18, continues=False, n_epochs=600, 
               gamma=0.99, learning_rate=1e-4, value_coeff=0, batch_size=144, 
               size_layer=[512, 512, 512]),
        7: Env(7, 'trunkcube-v0', timer_limit=18, continues=False, n_epochs=600, 
               gamma=0.99, learning_rate=1e-4, value_coeff=0, batch_size=144, 
               size_layer=[512, 512, 512]),
    }

    # Retrieve the environment-specific parameters from the dictionary `env_dict`
    # using the user-provided environment index (args.num_env). Each entry in
    # `env_dict` stores a set of hyperparameters and settings that define how
    # the RL algorithm should train and interact with the chosen environment.
    
    gamma = env_dict[args.num_env].gamma                  # Discount factor for the RL algorithm
    learning_rate = env_dict[args.num_env].learning_rate  # Optimizer's learning rate
    value_coeff = env_dict[args.num_env].value_coeff      # Coefficient for the value function loss
    batch_size = env_dict[args.num_env].batch_size        # Batch size used in training
    size_layer = env_dict[args.num_env].size_layer        # Size of the neural network layers
    
    # Additional environment details
    id = env_dict[args.num_env].name                      # Environment ID (string) for gym.make()
    timer_limit = env_dict[args.num_env].timer_limit      # Maximum steps/time allowed per episode
    cont = env_dict[args.num_env].continues               # Flag indicating if the action space is continuous
    n_epochs = env_dict[args.num_env].n_epochs            # Number of epochs (training iterations) to run

    # Choose and initialize the RL algorithm
    if args.num_algo == 0 and cont:
        # SAC (continuous action space)
        env = load_environment(id, rank=0, seed=args.seed * 10)()
        test_env = env
        algo = 'SAC'
        policy_kwargs = dict(net_arch=dict(pi=size_layer, qf=size_layer))
        model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, 
                    gamma=gamma, learning_rate=learning_rate, batch_size=batch_size, 
                    ent_coef='auto', learning_starts=500)

    elif args.num_algo == 1:
        # PPO (discrete or continuous, but commonly used for discrete as well)
        if args.num_cpu is None:
            print("[WARNING] >> Default number of cpu: 4.")
            n_cpu = 4
        else:
            n_cpu = args.num_cpu

        env = SubprocVecEnv([load_environment(id, i, seed=args.seed) for i in range(n_cpu)])
        test_env = load_environment(id, 0, seed=args.seed * 10)()
        algo = 'PPO'
        policy_kwargs = dict(net_arch=[dict(pi=size_layer, vf=size_layer)])
        model = PPO("MlpPolicy", env, n_steps=timer_limit * 20, batch_size=batch_size, 
                    gamma=gamma, policy_kwargs=policy_kwargs, verbose=1, 
                    learning_rate=learning_rate, device='cpu')

    elif args.num_algo == 2:
        # PPO version from the AVEC library 
        env = load_environment(id, rank=0, seed=args.seed * 10)()
        test_env = env
        algo = 'PPO_AVEC'
        policy_kwargs = dict(net_arch=[dict(pi=size_layer, vf=size_layer)])
        # model = PPO2(
        #    'MlpPolicy', env, avec_coef=1., vf_coef=value_coeff,
        #    n_steps=timer_limit*20, nminibatches=40, gamma=gamma,
        #    policy_kwargs=policy_kwargs, verbose=1, learning_rate=learning_rate
        # )

    elif args.num_algo == 3 and cont:
        # SAC version from the AVEC library 
        env = load_environment(id, rank=0, seed=args.seed * 10)()
        test_env = env
        algo = 'SAC_AVEC'
        layers = size_layer
        # model = SAC_AVEC(
        #    'CustomSACPolicy', env, avec_coef=1., value_coef=value_coeff,
        #    policy_kwargs={"layers": layers}, verbose=1, gamma=gamma,
        #    learning_rate=learning_rate, batch_size=batch_size,
        #    ent_coef='auto', learning_starts=500
        # )

    else:
        # Handle errors for incompatible action types or algo indexes
        if not cont and args.num_algo in [0, 3]:
            print("[ERROR] >> SAC is used with continuous action space.")
        else:
            print("[ERROR] >> num_algo is in {0, 1, 2, 3}")
        exit(1)

    # Prepare result directories
    name = algo + "_" + id + "_" + str(args.seed * 10)
    os.makedirs("./Results_benchmark/" + name, exist_ok=True)

    # Set random seeds for reproducibility
    seed = args.seed * 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.action_space.np_random.seed(seed)

    # Lists to keep track of training/testing progress
    rewards, final_rewards, steps = [], [], []
    best = -100000
    idx = 0

    print("\n-------------------------------")
    print(">>>    Start")
    print("-------------------------------\n")
    start_time = time.time()

    # If requested, restart training from a saved checkpoint
    if args.restart != 0:
        idx = args.restart
        print(">>>    Restart training from n°", idx + 1)
        del model
        save_path = "./Results_benchmark/" + name + "/latest"

        # Load the saved model according to the algorithm
        if args.num_algo == 0 and cont:
            model = SAC.load(save_path, env=env)
        elif args.num_algo == 1:
            model = PPO.load(save_path, env=env)
        # elif args.num_algo == 2:
        #     model = PPO2.load(save_path, env=env)
        # elif args.num_algo == 3 and cont:
        #     model = SAC_AVEC.load(save_path, env=env)
        else:
            if not cont and args.num_algo in [0, 3]:
                print("[ERROR] >> SAC is used with continuous action space.")
            else:
                print("[ERROR] >> num_algo is in {0, 1, 2, 3}")
            exit(1)

        # Restore logs from previous sessions
        with open("./Results_benchmark/" + name + f"/rewards_{id}.txt", 'r') as fp:
            rewards, steps = json.load(fp)
        with open("./Results_benchmark/" + name + f"/final_rewards_{id}.txt", 'r') as fp:
            final_rewards, steps = json.load(fp)
        best = max(rewards)
        print("Restored values:")
        print("\tSteps:", steps)
        print("\tRewards:", rewards)
        print("\tFinal rewards:", final_rewards)
        print("\tBest reward:", best)

    # Training loop
    if args.training:
        while idx < n_epochs:
            try:
                print("\n-------------------------------")
                print(">>>    Start training n°", idx + 1)
                print("[INFO]  >>    time:", sec_to_hours(time.time() - start_time))
                print("[INFO]  >>    scene:", id)
                print("[INFO]  >>    algo:", algo)
                print("[INFO]  >>    seed:", seed)
                print("-------------------------------\n")

                # Train the model for (timer_limit * 20) timesteps
                model.learn(total_timesteps=timer_limit * 20, log_interval=20)

                # Save the latest model
                model.save("./Results_benchmark/" + name + "/latest")

                print("\n-------------------------------")
                print(">>>    Start test n°", idx + 1)
                print("[INFO]  >>    scene:", id)
                print("[INFO]  >>    algo:", algo)
                print("[INFO]  >>    seed:", seed)
                print("-------------------------------\n")

                # Test the model (5 test rollouts)
                r, final_r = test(test_env, model, idx, n_test=5)
                final_rewards.append(final_r)
                rewards.append(r)
                steps.append(timer_limit * 20 * (idx + 1))

                # Save results to file
                with open("./Results_benchmark/" + name + f"/rewards_{id}.txt", 'w') as fp:
                    json.dump([rewards, steps], fp)
                with open("./Results_benchmark/" + name + f"/final_rewards_{id}.txt", 'w') as fp:
                    json.dump([final_rewards, steps], fp)

                # If this iteration's reward is better than previous best, save the model
                if r >= best:
                    print(">>>    Save training n°", idx + 1)
                    model.save("./Results_benchmark/" + name + "/best")

                idx += 1

            except:
                # If there's a runtime error in the environment, retry the same epoch
                print("[ERROR]  >> The simulation failed. Restart from previous id.")

        # Save the final model
        model.save("./Results_benchmark/" + name + "/latest")
        print(">>   End.")
        print("[INFO]  >>    time:", sec_to_hours(time.time() - start_time))

    # Testing loop
    if args.testing:
        print(">>>    Start testing")
        del model
        save_path = "./Results_benchmark/" + name + "/best"

        # Load the best model for testing
        if args.num_algo == 0 and cont:
            model = SAC.load(save_path)
        elif args.num_algo == 1:
            model = PPO.load(save_path)
        # elif args.num_algo == 2:
        #     model = PPO2.load(save_path)
        # elif args.num_algo == 3 and cont:
        #     model = SAC_AVEC.load(save_path)
        else:
            if not cont and args.num_algo in [0, 3]:
                print("[ERROR] >> SAC is used with continuous action space.")
            else:
                print("[ERROR] >> num_algo is in {0, 1, 2, 3}")
            exit(1)

        # Conduct tests with rendering
        r, final_r = test(test_env, model, -1, n_test=5, render=True)
        print("[INFO]  >>    Best reward:", r, " - Final reward:", final_r)

