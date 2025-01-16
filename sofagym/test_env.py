# -*- coding: utf-8 -*-
"""
Test script for a SofaGym environment, designed to demonstrate basic policy exploration
(either random actions or predefined strategies). This script extends
the original SofaGym framework with additional domain randomization techniques.

Usage:
-----
    python3.7 test_env.py
"""

__authors__ = "Andrea Protopapa, Gabriele Tiboni"
__contact__ = "andrea.protopapa@polito.it, gabriele.tiboni@polito.it"
__version__ = "1.0.0"
__copyright__ = "(c) 2023, Politecnico di Torino, Italy"
__date__ = "Oct 28 2023"

import sys
import os
import time
import gym
import argparse

from sofagym import *

RANDOM = False  # Flag for enabling random actions (unused here)

import psutil
pid = os.getpid()
py = psutil.Process(pid)  # Used to monitor memory usage

import random as rd

# Insert the parent directory into the system path to allow importing modules from there
sys.path.insert(0, os.getcwd() + "/..")

# Import SofaGym package (needed for environment registration)
__import__('sofagym')

# A dictionary mapping integer indices to specific Gym environment IDs
env_dict = {
    0: 'multigaitrobot-v0',
    1: 'trunk-v0',
    2: 'trunkcube-v0'
}

# Parse a command-line argument to select which environment to run
parser = argparse.ArgumentParser()
parser.add_argument("-ne", "--num_env", help="Number of the environment", type=int, required=True)
args = parser.parse_args()

# Retrieve the environment name from env_dict using the provided index
env_name = env_dict[args.num_env]
print("Start env", env_name)

# Create the specified SofaGym environment
env = gym.make(env_name)
# Configure the environment to render the simulation at each step
env.configure({"render": 2})
# Set the simulation time step
env.configure({"dt": 0.01})
# Initialize the environment
env.reset()

# Immediately render the environment once
env.render()

done = False

# Example continuous strategy for 'multigaitrobot' environment (repeated 100 times)
strat_multi = [
    [-1.0, -1.0, -1.0, 1, 1], 
    [1.0, -1.0, -1.0, 1, 1],
    [1.0, 1.0, 1.0, 1, 1], 
    [1.0, 1.0, 1.0, -1.0, -1.0],
    [-1.0, 1.0, 1.0, -1.0, -1.0], 
    [-1.0, -1.0, -1.0, -1.0, -1.0],
    [-1.0, -1.0, -1.0, 1, 1], 
    [1.0, -1.0, -1.0, 1, 1],
    [1.0, 1.0, 1.0, 1, 1], 
    [1.0, 1.0, 1.0, -1.0, -1.0],
    [-1.0, 1.0, 1.0, -1.0, -1.0], 
    [-1.0, -1.0, -1.0, -1.0, -1.0]
] * 100

# Example discrete strategy for 'multigaitrobot' environment
start_multi_discrete = [
    2, 0, 1, 5, 3, 4,
    2, 0, 1, 5, 3, 4,
    2, 0, 1, 5, 3, 4,
    2, 0, 1, 5, 3, 4,
    2, 0, 1, 5, 3, 4
] * 4

# Example action sequences for a "Jimmy" style environment (unused here but kept for reference)
strat_jimmy_1 = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-0.6, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-0.6, -0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.75, 0.75, 0.0, 0.0, 0.0, 1.0, 0.0]
] + [[0.0, 0.75, 0.75, 0.0, 0.0, 0.0, 1.0, 0.0]] * 100

strat_jimmy_0 = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ...
    [-0.8, 0.2, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
] + [[-0.8, 0.2, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]] * 100

print("Start ...")

num_episodes = 3  # Number of episodes to run
for i in range(num_episodes):
    print("\n--------------------------------")
    print("EPISODE -", i)
    print("--------------------------------\n")
    
    idx = 0
    tot_reward = 0
    tot_rtf = 0
    done = False
    
    # Each episode will continue until 'done' or until we exceed 100 steps
    while not done and idx < 100:
        idx += 1

        # Example ways to select an action (choose one):
        #   1. For 'multigaitrobot': continuous action from strat_multi or discrete from start_multi_discrete
        #   2. For 'trunk': random selection of a discrete action
        #   3. For 'trunkcube': also random discrete
        #   etc.

        # Continuous example for 'multigaitrobot':
        multigaitrobot = start_multi_discrete[idx - 1]

        # For trunk, trunkcube, trunkwall: pick a random discrete action
        trunk = rd.randint(0, 15)
        trunkcube = rd.randint(0, 15)
        trunkwall = rd.randint(0, 15)

        # Group actions by environment (env_name index)
        action_type = [multigaitrobot, trunk, trunkcube, trunkwall]

        # Select the action for the current environment
        action = action_type[args.num_env]

        # Step the environment forward using the chosen action
        start_time = time.time()
        state, reward, done, info = env.step(action)
        step_time = time.time() - start_time

        # Compute Real Time Factor (RTF): ratio of simulated to real elapsed time
        rtf = env.config["dt"] * env.config["scale_factor"] / step_time
        print("[INFO]   >>> Time:", step_time)
        print("[INFO]   >>> RTF:", rtf)

        tot_reward += reward
        tot_rtf += rtf

        # Render the current step
        env.render()

        print("Step", idx, 
              "action:", action, 
              "reward:", reward, 
              "done:", done, 
              "- info:", info)

    print("[INFO]   >>> TOTAL REWARD IS:", tot_reward)
    print("[INFO]   >>> FINAL REWARD IS:", reward)
    print("[INFO]   >>> MEAN RTF IS:", tot_rtf / idx)

    # Display memory usage and environment size
    memoryUse = py.memory_info()[0] / 2.0 ** 30
    print("[INFO]   >>> Memory usage:", memoryUse)
    print("[INFO]   >>> Object size:", sys.getsizeof(env))

    # Reset the environment for the next episode
    env.reset()

# Final log statement after all episodes are done
print(">> TOTAL REWARD IS:", tot_reward)
env.close()
print("... End.")

