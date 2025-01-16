# -*- coding: utf-8 -*-
"""
AbstractEnv provides the link between Gym and Sofa by implementing core functionalities
for controlling and monitoring a Sofa simulation via a Gym interface. This class is
designed to be extended with domain randomization techniques, building on top of the
original SofaGym framework (https://github.com/SofaDefrost/SofaGym).

Usage:
-----
    This class should be subclassed to define environment-specific logic.

Key functionalities:
    - Managing Sofa simulation through an RPC server.
    - Handling domain randomization (DR) parameters (sampling distributions, etc.).
    - Providing Gym-style methods (step, reset, render, etc.).
"""

__authors__ = "Andrea Protopapa, Gabriele Tiboni"
__contact__ = "andrea.protopapa@polito.it, gabriele.tiboni@polito.it"
__version__ = "1.0.0"
__copyright__ = "(c) 2023, Politecnico di Torino, Italy"
__date__ = "Oct 28 2023"

import gym
from gym.utils import seeding
from gym import spaces

import numpy as np
import copy
import os
import sys
from scipy.stats import truncnorm
import pdb

# Sofa libraries
import splib
from splib import animation

# SofaGym components
from sofagym.env.common.viewer import Viewer
from sofagym.env.common.rpc_server import start_server, add_new_step, get_result, clean_registry, close_scene


class AbstractEnv(gym.Env):
    """
    AbstractEnv is an intermediate class for connecting Gym to a Sofa simulation.
    It includes the logic for:
      - Sending actions to the Sofa simulation through an RPC server.
      - Receiving observations, rewards, and 'done' flags back from Sofa.
      - Handling environment resets and rendering.
      - Managing domain randomization by sampling new environment parameters.

    Methods to override/implement in subclasses:
      - get_search_bounds(self, index)
      - get_task_lower_bound(self, index)
      - get_task_upper_bound(self, index)
      - get_task(self)
      - set_task(self, *task)

    These methods define how domain randomization parameters are retrieved
    and manipulated within the environment.

    Attributes:
      config (dict): Dictionary specifying various simulation parameters.
      DEFAULT_CONFIG (dict): A default configuration to be copied and updated.
      goalList, goal: Define a set of possible goals and the current active goal.
      np_random (RandomState): Random generator for seeding.
      viewer (Viewer): An instance of the custom Viewer for rendering.
      timer (int): Counts the number of steps in the current episode.
      endless (bool): If True, episodes do not terminate when 'done' is True.
      dr_training (bool): If True, randomizes parameters at each reset.
      min_task, max_task, mean_task, stdev_task (np.ndarray): Used to store
        DR distribution parameters. Their usage depends on the configured
        sampling type (uniform, truncnorm, gaussian, etc.).

    Usage:
      1. Subclass AbstractEnv and override the required methods for your Sofa scene.
      2. Instantiate the subclass and call reset() before stepping in the environment.
    """

    def __init__(self, config=None):
        """
        Initializes the environment using the provided or default config.

        Args:
            config (dict, optional): A dictionary of configuration parameters.
        """
        # Copy the default configuration from the subclass
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        # Merge any custom configurations passed in
        if config is not None:
            self.config.update(config)

        self.goal_idx = 0
        self.initialization()

    def initialization(self):
        """
        Sets up core environment attributes and starts the Sofa RPC server.

        This method:
          - Initializes seed, viewer placeholder, and episode timer.
          - Creates directories for saving data, images, or videos if specified.
          - Starts the Sofa RPC server to handle simulation calls.
        """
        # Initialize goal-related attributes
        self.goalList = None
        self.goal = None

        # Store past actions for logging and planning
        self.past_actions = []

        # Number of environment clients (used by the server)
        self.num_envs = 40

        self.np_random = None

        # Seed the environment
        self.seed(self.config['seed'])

        # Viewer and callback placeholders
        self.viewer = None
        self.automatic_rendering_callback = None

        # Episode timer to track time steps
        self.timer = 0
        # Timeout for getting results from the server
        self.timeout = self.config["timeout"]

        # Start the Sofa RPC server
        start_server(self.config)

        # Handle optional data saving (csv, images, video)
        if 'save_data' in self.config and self.config['save_data']:
            save_path_results = self.config['save_path'] + "/data"
            os.makedirs(save_path_results, exist_ok=True)
        else:
            save_path_results = None

        if 'save_image' in self.config and self.config['save_image']:
            save_path_image = self.config['save_path'] + "/img"
            os.makedirs(save_path_image, exist_ok=True)
        else:
            save_path_image = None

        if 'save_video' in self.config and self.config['save_video']:
            save_path_video = self.config['save_path'] + "/video"
            os.makedirs(save_path_video, exist_ok=True)
        else:
            save_path_video = None

        # Update the config with the final paths for saving data/images/videos
        self.configure({
            "save_path_image": save_path_image,
            "save_path_results": save_path_results,
            "save_path_video": save_path_video
        })

    # --- Domain Randomization (DR) stubs to be overridden in a subclass ---

    def get_search_bounds(self, index):
        """
        Returns the search bounds for the randomizable parameter at 'index'.
        Must be implemented in a subclass.
        """
        raise NotImplementedError

    def get_task_lower_bound(self, index):
        """
        Returns the lower feasible bound for the randomizable parameter at 'index'.
        Must be implemented in a subclass.
        """
        raise NotImplementedError

    def get_task_upper_bound(self, index):
        """
        Returns the upper feasible bound for the randomizable parameter at 'index'.
        Must be implemented in a subclass.
        """
        raise NotImplementedError

    def get_task(self):
        """
        Returns the current set of dynamic parameters for domain randomization.
        Must be implemented in a subclass.
        """
        raise NotImplementedError

    def set_task(self, *task):
        """
        Sets the domain randomization parameters to 'task'.
        Must be implemented in a subclass.
        """
        raise NotImplementedError

    # --- Domain Randomization Utility Methods ---

    def set_random_task(self):
        """Samples a new set of random parameters and applies them to the environment."""
        self.set_task(*self.sample_task())

    def set_dr_training(self, flag):
        """
        Enables or disables domain randomization during reset.

        Args:
            flag (bool): If True, random parameters are sampled at each reset.
        """
        self.dr_training = flag

    def get_dr_training(self):
        """Returns the domain randomization training flag."""
        return self.dr_training

    def set_endless(self, flag):
        """
        Controls whether the environment's episodes ever terminate.

        Args:
            flag (bool): If True, 'done' will always be False.
        """
        self.endless = flag

    def get_endless(self):
        """Returns the endless flag, indicating if episodes never end."""
        return self.endless

    def get_reward_threshold(self):
        """Optional method to retrieve a target reward threshold for tasks."""
        return self.reward_threshold

    def sample_tasks(self, num_tasks=1):
        """
        Generates an array of tasks by sampling the configured DR distribution.

        Args:
            num_tasks (int): Number of different tasks to sample.
        """
        return np.stack([self.sample_task() for _ in range(num_tasks)])

    def set_dr_distribution(self, dr_type, distr):
        """
        Sets up domain randomization based on the distribution type and parameters.

        dr_type can be one of: 'uniform', 'truncnorm', 'gaussian', 'fullgaussian'.
        distr contains bounds or means/covariances needed by each type.

        Args:
            dr_type (str): The DR type (e.g., 'uniform', 'truncnorm', etc.).
            distr (dict or list): The specific parameters or bounds for that type.
        """
        if dr_type == 'uniform':
            self.set_udr_distribution(distr)
        elif dr_type == 'truncnorm':
            self.set_truncnorm_distribution(distr)
        elif dr_type == 'gaussian':
            self.set_gaussian_distribution(distr)
        elif dr_type == 'fullgaussian':
            self.set_fullgaussian_distribution(distr['mean'], distr['cov'])
        else:
            raise Exception('Unknown dr_type: ' + str(dr_type))

    def get_dr_distribution(self):
        """
        Returns the current DR distribution parameters depending on the sampling type.
        """
        if self.sampling == 'uniform':
            return self.min_task, self.max_task
        elif self.sampling == 'truncnorm':
            return self.mean_task, self.stdev_task
        elif self.sampling == 'gaussian':
            raise ValueError('Not implemented for gaussian in detail yet.')
        else:
            return None

    def set_udr_distribution(self, bounds):
        """
        Configures Uniform Domain Randomization (UDR).

        Args:
            bounds (list): Pairs of [min, max] for each DR parameter.
        """
        self.sampling = 'uniform'
        for i in range(len(bounds)//2):
            self.min_task[i] = bounds[i * 2]
            self.max_task[i] = bounds[i * 2 + 1]

    def set_truncnorm_distribution(self, bounds):
        """
        Configures Truncated Normal distribution for domain randomization.

        Args:
            bounds (list): Pairs of [mean, stdev] for each DR parameter.
        """
        self.sampling = 'truncnorm'
        for i in range(len(bounds)//2):
            self.mean_task[i] = bounds[i * 2]
            self.stdev_task[i] = bounds[i * 2 + 1]

    def set_gaussian_distribution(self, bounds):
        """
        Configures a simple Gaussian distribution for domain randomization.

        Args:
            bounds (list): Pairs of [mean, stdev] for each DR parameter.
        """
        self.sampling = 'gaussian'
        for i in range(len(bounds)//2):
            self.mean_task[i] = bounds[i * 2]
            self.stdev_task[i] = bounds[i * 2 + 1]

    def set_fullgaussian_distribution(self, mean, cov):
        """
        Configures a full multivariate Gaussian distribution for domain randomization.

        Args:
            mean (np.ndarray): Array of means for each parameter.
            cov (np.ndarray): Covariance matrix for the parameters.
        """
        self.sampling = 'fullgaussian'
        self.mean_task[:] = mean
        self.cov_task = np.copy(cov)

    def set_task_search_bounds(self):
        """
        Sets the task search bounds (min_task, max_task) based on subclass's get_search_bounds.
        """
        dim_task = len(self.get_task())
        for i in range(dim_task):
            b = self.get_search_bounds(i)
            self.min_task[i], self.max_task[i] = b[0], b[1]

    def get_task_search_bounds(self):
        """
        Retrieves arrays of the minimum and maximum DR bounds from get_search_bounds.

        Returns:
            (np.ndarray, np.ndarray): Arrays of min and max bounds.
        """
        dim_task = len(self.get_task())
        min_task = np.empty(dim_task)
        max_task = np.empty(dim_task)
        for i in range(dim_task):
            b = self.get_search_bounds(i)
            min_task[i], max_task[i] = b[0], b[1]
        return min_task, max_task

    def sample_task(self):
        """
        Samples a new set of parameters based on the current DR sampling type.
        """
        if self.sampling == 'uniform':
            return np.random.uniform(self.min_task, self.max_task, self.min_task.shape)

        elif self.sampling == 'truncnorm':
            a, b = -2, 2  # Truncation bounds in terms of standard deviations
            sample = []

            for i, (mean, std) in enumerate(zip(self.mean_task, self.stdev_task)):
                obs = truncnorm.rvs(a, b, loc=mean, scale=std)
                
                lower_bound = self.get_task_lower_bound(i)
                upper_bound = self.get_task_upper_bound(i)

                attempts = 0
                # Resample if outside feasible bounds
                while obs < lower_bound or obs > upper_bound:
                    obs = truncnorm.rvs(a, b, loc=mean, scale=std)
                    attempts += 1
                    if attempts > 2:
                        obs = (lower_bound + upper_bound) / 2
                        break
                sample.append(obs)

            return np.array(sample)

        elif self.sampling == 'gaussian':
            # Simple, uncorrelated Gaussian per parameter
            sample = []
            for i, (mean, std) in enumerate(zip(self.mean_task, self.stdev_task)):
                obs = np.random.randn() * std + mean
                lower_bound = self.get_task_lower_bound(i)
                upper_bound = self.get_task_upper_bound(i)

                attempts = 0
                # Ensure parameter is within feasible lower bound
                while obs < lower_bound:
                    obs = np.random.randn() * std + mean
                    attempts += 1
                    if attempts > 2:
                        raise Exception("Sample couldn't be above lower_bound after 2 attempts.")

                attempts = 0
                # Ensure parameter is within feasible upper bound
                while obs > upper_bound:
                    obs = np.random.randn() * std + mean
                    attempts += 1
                    if attempts > 2:
                        raise Exception("Sample couldn't be below upper_bound after 2 attempts.")

                sample.append(obs)
            return np.array(sample)

        elif self.sampling == 'fullgaussian':
            # Sample from a multivariate normal distribution
            sample = np.random.multivariate_normal(self.mean_task, self.cov_task)
            sample = np.clip(sample, 0, 4)  # Clip to [0,4] normalized space
            sample = self.denormalize_parameters(sample)
            return sample

        else:
            raise ValueError("Sampling type not set. Use set_dr_distribution() to configure it.")

    def denormalize_parameters(self, parameters):
        """
        Converts normalized parameters in the range [0, 4] back to their original DR bounds.

        Args:
            parameters (np.ndarray): Normalized parameter values.

        Returns:
            np.ndarray: Denormalized parameter values in the original task space.
        """
        assert parameters.shape[0] == self.task_dim

        min_task, max_task = self.get_task_search_bounds()
        parameter_bounds = np.empty((self.task_dim, 2), float)
        parameter_bounds[:, 0] = min_task
        parameter_bounds[:, 1] = max_task

        # Scale the normalized [0,4] range back to [min_task, max_task]
        orig_parameters = (parameters * (parameter_bounds[:, 1] - parameter_bounds[:, 0])) / 4 + parameter_bounds[:, 0]
        return np.array(orig_parameters)

    # --- Gym Environment Methods ---

    def seed(self, seed=None):
        """
        Initialize the random seed for the environment.

        Args:
            seed (int or None): Seed value or None for a random seed.

        Returns:
            list: The actual seed used.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _formataction(self, action):
        """
        Converts the action into a format suitable for the server (list/float/int).

        Args:
            action: The raw action (int, float, np.ndarray, dict, tuple).

        Returns:
            action: The formatted action.
        """
        if isinstance(action, np.ndarray):
            action = action.tolist()
        elif isinstance(action, np.int64):
            action = int(action)
        elif isinstance(action, np.float64):
            action = float(action)
        elif isinstance(action, tuple):
            action = self._formatactionTuple(action)
        elif isinstance(action, dict):
            action = self._formatactionDict(action)
        return action

    def _formatactionTuple(self, action):
        """
        Helper for converting tuple actions into valid formats recursively.

        Args:
            action (tuple): The raw tuple action.

        Returns:
            The formatted action (list/float/int) for each element in the tuple.
        """
        return self._formataction(action[0]), self._formataction(action[1])

    def _formatactionDict(self, action):
        """
        Helper for converting dict actions into valid formats.

        Args:
            action (dict): The raw dict action.

        Returns:
            dict: The dict with each value formatted appropriately.
        """
        for key in action.keys():
            action[key] = self._formataction(action[key])
        return action

    def clean(self):
        """
        Cleans the registry to remove references to old or unused steps (used in planning).

        This calls the SofaGym function 'clean_registry' with the environment's past actions.
        """
        clean_registry(self.past_actions)

    def step(self, action):
        """
        Performs a single action in the environment, advances the Sofa simulation,
        and returns the resulting observation, reward, done, and info.

        Args:
            action: The action to apply (discrete/continuous, depends on the environment).

        Returns:
            observation (np.ndarray): The next state observation.
            reward (float): The reward for the taken action.
            done (bool): Flag indicating if the episode has ended.
            info (dict): Additional diagnostic information.
        """
        # Convert the action to a server-compatible format
        action = self._formataction(action)

        # Add the action as a new step to the server and store the resulting ID
        result_id = add_new_step(self.past_actions, action)
        self.past_actions.append(action)

        # Get the result from the server, blocking until it's ready or hitting timeout
        results = get_result(result_id, timeout=self.timeout)
        obs = np.array(results["observation"])  # Next state
        reward = results["reward"]
        done = results["done"]
        info = results["info"]

        # Increment the step timer; if we exceed 'timer_limit', end the episode
        self.timer += 1
        if self.timer >= self.config["timer_limit"]:
            done = True

        # Clean up if in planning mode
        if self.config["planning"]:
            self.clean()

        # If endless mode is set, episodes never end
        if self.endless:
            done = False

        return obs, reward, done, info

    def async_step(self, action):
        """
        Performs an action in the environment asynchronously, returning a LateResult
        object that can be queried later via the 'get()' method.

        Args:
            action: The action to apply.

        Returns:
            LateResult: An object containing the result_id which can be used
                        to get the (obs, reward, done) at a later time.
        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        result_id = add_new_step(self.past_actions, action)
        self.past_actions.append(action)

        class LateResult:
            """
            Helper class to store the result_id and retrieve the results asynchronously.
            """
            def __init__(self, result_id):
                self.result_id = result_id

            def get(self, timeout=None):
                results = get_result(self.result_id, timeout=timeout)
                obs = results["observation"]
                reward = results["reward"]
                done = results["done"]
                return obs, reward, done, {}

        return LateResult(copy.copy(result_id))

    def reset(self):
        """
        Resets the environment to an initial state. This includes:
          - Closing any existing scene.
          - Re-initializing environment attributes via initialization().
          - Sampling and setting a new random goal if the environment uses goalList.
          - Clearing the timer and past actions.
        """
        self.close()
        self.initialization()
        splib.animation.animate.manager = None

        # If no predefined goalList, check config for either a random goal box or single goal
        if not self.goalList:
            self.goalList = self.config["goalList"]

        if isinstance(self.goalList, list):
            # Select a random goal index from the available goals
            id_goal = self.np_random.choice(range(len(self.goalList)))
            self.config.update({'goal_node': id_goal})
            self.goal = self.goalList[id_goal]
        elif self.goalList is None:
            # Possibly sample a goal from a range or use a test goal
            if self.config.get("test", False):
                self.goal = self.config["goalList_test"][self.goal_idx // 2]
                self.goal_idx += 1
            else:
                # Sample a goal from a continuous space
                self.goalList = spaces.Box(
                    np.array(self.config["goal_low"]),
                    np.array(self.config["goal_high"]),
                    dtype=np.float32
                )
                self.goal = self.goalList.sample().tolist()
                self.goalList = None

        self.timer = 0
        self.past_actions = []
        return

    def render(self, mode='rgb_array', createVideo=None):
        """
        Renders the current state of the simulation, if rendering is enabled.

        Args:
            mode (str): Optional rendering mode. Default is 'rgb_array'.
            createVideo (str): If provided, saves rendered frames under this path.
        """
        if self.config['render'] != 0:
            if not self.viewer:
                # Create the Viewer if not yet defined
                display_size = self.config["display_size"]
                zFar = self.config.get('zFar', 0)
                self.viewer = Viewer(
                    self,
                    display_size,
                    zFar=zFar,
                    save_path=self.config["save_path_image"],
                    create_video=createVideo
                )
            # Draw the environment
            self.viewer.render()
        else:
            print(">> No rendering")

    def _automatic_rendering(self):
        """
        Automatically renders intermediate frames if a callback is set or if manual rendering is configured.
        This method can be called after each small simulation step, giving a continuous visualization.
        """
        if self.viewer is not None:
            if self.automatic_rendering_callback:
                self.automatic_rendering_callback()
            else:
                self.render()

    def close(self):
        """
        Closes the environment by shutting down the viewer and the Sofa scene.
        """
        if self.viewer is not None:
            self.viewer.close()

        # Close the Sofa simulation (RPC server)
        close_scene()
        print("All clients are closed. Bye Bye.")

    def configure(self, config):
        """
        Updates the environment configuration.

        Args:
            config (dict): Dictionary of new/updated configuration parameters.
        """
        self.config.update(config)

