# -*- coding: utf-8 -*-
"""
Specific environment for the trunk, aiming to reach a target goal position. 
Developed as an extension of the original work https://github.com/SofaDefrost/SofaGym
by adding Domain Randomization techniques.
"""

__authors__ = "Andrea Protopapa, Gabriele Tiboni"
__contact__ = "andrea.protopapa@polito.it, gabriele.tiboni@polito.it"
__version__ = "1.0.0"
__copyright__ = "(c) 2023, Politecnico di Torino, Italy"
__date__ = "Oct 28 2023"

from sofagym.env.common.AbstractEnv import AbstractEnv
from sofagym.env.common.rpc_server import start_scene

from gym.envs.registration import register
from gym import spaces
import os
import numpy as np

import json

class TrunkEnv(AbstractEnv):
    """
    Sub-class of AbstractEnv, dedicated to the trunk scene. 
    
    The TrunkEnv environment is designed to simulate a soft robotic trunk 
    tasked with reaching a specified goal position. This class extends the
    SofaGym AbstractEnv and integrates domain randomization.
    
    Args:
        config (dict): Optional dictionary specifying environment configuration.
        unmodeled (bool): Flag to indicate the presence of unmodeled dynamics 
                          (currently not used in this code).
    
    Attributes:
        task_dim (int): The number of dynamic parameters to randomize.
        min_task, max_task, mean_task, stdev_task (np.array): Arrays to 
            store information about the parameter distributions.
        dynamics_indexes (dict): Maps each dynamic parameter index to its 
            configuration key in `config`.
        action_space (gym.Space): Defines the discrete set of available actions.
        observation_space (gym.Space): Defines the space of possible observations.
        dr_training (bool): Whether domain randomization training is enabled.
        sampling, preferred_lr, reward_threshold (None): Placeholders for 
            potential future extensions.
        endless (bool): Indicates if the environment will run indefinitely 
            without a set number of episodes.
    """
    # Setting a default configuration
    path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    DEFAULT_CONFIG = {
        "scene": "Trunk",
        "deterministic": True,

        # Source and target positions (3D coordinates) in the sofa scene
        "source": [100, 50, 400],
        "target": [0, 0, 0],

        # Possible goal positions that the trunk is asked to reach
        "goalList": [[0.0, -100.0, 100.0], [40, -60, 200], [-10, 20, 80]],

        # Bounds for random goal selection
        "goal_low": [-50., -50, -50],
        "goal_high": [50., 50., 50.],

        # "start_node" specifies an optional starting node in the SOFA simulation graph.
        "start_node": None,
        
        # "scale_factor" determines how much the environment (or robot) dimensions
        # are scaled in the simulation.
        "scale_factor": 5,
        
        # "timer_limit" represents the maximum number of time steps before the simulation
        # or episode is reset or deemed complete.
        "timer_limit": 100,
        
        # "dt" is the timestep duration for the SOFA simulation, controlling how much
        # virtual time elapses between updates.
        "dt": 0.01,

        # Timeout for the environment
        "timeout": 50,
        # Display size for rendering
        "display_size": (1400, 500),

        # Render level, saving options (data, video, images)
        "render": 2,
        "save_data": False,
        "save_video": True,
        "save_image": False,
        # Path for saving outputs
        "save_path": path + "/Results" + "/Trunk",

        "planning": False,
        "discrete": True,
        "seed": None,
        "start_from_history": None,
        "python_version": "python3.8"
    }

    def __init__(self, config=None, unmodeled=False):
        """
        Constructor for the TrunkEnv class.

        If a config is not provided, it tries to load a default JSON configuration
        from the 'Trunk_random_config.json' file. Then, sets up the internal
        parameters and calls the super-class constructor.

        Args:
            config (dict, optional): Custom configuration for the environment.
            unmodeled (bool, optional): Placeholder parameter for unmodeled dynamics.
        """
        if config is None:
            # Load default config from Trunk_random_config.json if not supplied
            config_path = os.path.dirname(os.path.abspath(__file__)) + "/Trunk/Trunk_random_config.json"
            with open(config_path) as config_random:
                config = json.load(config_random)

        # Number of dynamic parameters to randomize
        self.task_dim = len(config["dynamic_params"])

        # Initialize arrays for domain randomization tasks
        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)
        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        # Dictionary to map each parameter index to a name in config
        self.dynamics_indexes = {}
        for i in range(self.task_dim):
            self.dynamics_indexes[i] = config["dynamic_params"][i]

        # Call parent class constructor to handle the rest of the initialization
        super().__init__(config)

        # Define the discrete action space (16 possible actions)
        nb_actions = 16
        self.action_space = spaces.Discrete(nb_actions)
        self.nb_actions = str(nb_actions)

        # Define the observation space (66-dimensional)
        dim_state = 66
        low_coordinates = np.array([-1]*dim_state)
        high_coordinates = np.array([1]*dim_state)
        self.observation_space = spaces.Box(low_coordinates,
                                            high_coordinates,
                                            dtype='float32')

        # Additional flags and placeholders for domain randomization
        self.sampling = None
        self.dr_training = False
        self.preferred_lr = None
        self.reward_threshold = None
        self.endless = False

    # ------------------ Domain Randomization Methods ------------------

    def get_search_bounds(self, index):
        """
        Retrieve search bounds (min, max) for a given dynamic parameter index.
        
        Args:
            index (int): Index of the dynamic parameter for which bounds are requested.
        
        Returns:
            tuple: (min_search, max_search) bounds for the parameter.
        """
        search_bounds = {}
        for i in range(self.task_dim):
            # Construct key name "<param_name>_min_search"/"_max_search" from the config
            search_bounds[i] = (
                self.config[self.dynamics_indexes[i] + "_min_search"],
                self.config[self.dynamics_indexes[i] + "_max_search"]
            )
        return search_bounds[index]
    
    def get_search_bounds_all(self):
        """
        Retrieve the search bounds (min, max) for all dynamic parameters.
        
        Returns:
            (list, list): Two lists containing the min and max bounds for each parameter.
        """
        min_search = []
        max_search = []
        for i in range(self.task_dim):
            min_search.append(self.config[self.dynamics_indexes[i] + "_min_search"])
            max_search.append(self.config[self.dynamics_indexes[i] + "_max_search"])
        return min_search, max_search

    def get_task_lower_bound(self, index):
        """
        Retrieve the lowest feasible value for a given dynamic parameter index. 
        Used in domain randomization to avoid unfeasible values.
        
        Args:
            index (int): Index of the dynamic parameter.
        
        Returns:
            float/int: The lowest feasible value for the parameter.
        """
        lowest_value = {}
        for i in range(self.task_dim):
            # Construct key name "<param_name>_lowest" from the config
            lowest_value[i] = self.config[self.dynamics_indexes[i] + "_lowest"]
        return lowest_value[index]

    def get_task_upper_bound(self, index):
        """
        Retrieve the highest feasible value for a given dynamic parameter index. 
        Used in domain randomization to avoid unfeasible values.
        
        Args:
            index (int): Index of the dynamic parameter.
        
        Returns:
            float/int: The highest feasible value for the parameter.
        """
        highest_value = {}
        for i in range(self.task_dim):
            # Construct key name "<param_name>_highest" from the config
            highest_value[i] = self.config[self.dynamics_indexes[i] + "_highest"]
        return highest_value[index]

    def get_task(self):
        """
        Get the current dynamic parameters values from the configuration.
        
        Returns:
            np.array: Array of current dynamic parameters.
        """
        dynamic_params_values = np.array(self.config["dynamic_params_values"])
        return dynamic_params_values

    def set_task(self, *task):
        """
        Set new dynamic parameters (e.g., for domain randomization).

        Args:
            *task: Variable number of parameter values (one for each dynamic param).
        """
        self.config["dynamic_params_values"] = task

    # ------------------ Overridden AbstractEnv Methods ------------------

    def step(self, action):
        """
        Execute one step in the environment using the specified action.
        
        Args:
            action (int or np.ndarray): The action to be taken. If an array is given, 
                                        the first element is used.
        
        Returns:
            (observation, reward, done, info): Standard Gym step return values.
        """
        if isinstance(action, np.ndarray):
            action = action[0]
        return super().step(action)

    def reset(self):
        """
        Reset the simulation environment.

        This method re-initializes the environment to a start state. If 
        'dr_training' is enabled, randomizes the environment dynamics. 
        Then, starts the RPC server scene and retrieves the initial observation.

        Returns:
            np.ndarray: Initial observation after resetting.
        """
        super().reset()

        # Update config with the chosen goal
        self.config.update({'goalPos': self.goal})
        print(f"list: {self.goalList}")
        print(f"goal: {self.goal}")

        # If domain randomization is enabled, sample new dynamics
        if self.dr_training:
            self.set_random_task()

        # Start the SOFA scene, retrieve initial observation
        obs = start_scene(self.config, self.nb_actions)

        return obs['observation']

    def render(self, mode='rgb_array', createVideo=None):
        """
        Render the current state of the environment.
        
        Args:
            mode (str): The mode of rendering ('rgb_array' or 'human').
            createVideo (str, optional): Path where the video will be saved. 
                                         If None and config["save_video"] is True, 
                                         it uses the default save path.
        """
        if createVideo is None and self.config["save_video"]:
            createVideo = self.config['save_path_video'] + "/"
        # Leverage the parent class method to handle actual rendering
        super().render(mode, createVideo)

    def get_available_actions(self):
        """
        Return a list of all valid actions in this environment.
        
        Returns:
            list: A list of valid action indices.
        """
        return list(range(int(self.nb_actions)))


# Register the environment with OpenAI Gym
register(
    id='trunk-v0',
    entry_point='sofagym.env:TrunkEnv',
)
