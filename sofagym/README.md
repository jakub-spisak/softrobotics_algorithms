# SofaGym - original from https://github.com/SofaDefrost/SofaGym

Software toolkit to easily create an OpenAI Gym environment out of any SOFA scene.
The toolkit provides an API based on the standard OpenAI Gym API, allowing to train classical Reinforcement Learning algorithms. The toolkit also comprises example scenes based on the SoftRobots plugin for SOFA to illustrate how to include SOFA simulations and train learning algorithms on them.

## Usage

The Gym framework allows to interact with an environment using well-known keywords:
- *step(a)*: allows to perform a simulation step when the agent performs the action *a*. Given the current state of the system *obs_t* and the action *a*, the environment then changes to a new state *obs_{t+1}* and the agent receives the reward *rew*. If the goal is reached, the *done* flag changes to *True*.
- *reset*: resets the environment.
- *render*: gives a visual representation of *obs_t*.

The use of this interface allows intuitive interaction with any environment, and this is what SofaGym allows when the environment is a Sofa scene. For more information on Gym, check the official documentation page [here](https://gym.openai.com/docs/).

## The tools

### Server/worker architecture

The major difficulty encountered in this work is the fact that it is not possible to copy the *root* from a Sofa simulation. This implies that when two sequences of actions *A_1 = [a_1, ..., a_n, new_action_1]* and *A_2 = [a_1, ..., a_n, new_action_2]* have to be tried, it is necessary to start again from the beginning each time and simulate again *[a_1, ..., a_n]*. This leads to a huge loss of performance. To solve this problem a server/worker architecture is set up.

A server takes care of distributing the calculations between several clients. Each client *i* is associated with an action sequence *A_i = [a_{i1}, ...., a_{in}]*. Given an action sequence *A = [a_{1}, ...., a_{n}]* and a new action *a*, the server looks for the client with the action sequence *A_i*. This client forks and the child executes the new action *a*. The father and son are referenced to the server as two separate clients and the action sequence *[a_{1}, ...., a_{n}]* and *[a_{1}, ...., a_{n}, a]* can be accessed.

A cleaning system is used to close clients that are no longer used. This makes it possible to avoid having an exponential number of open clients.

When it is not necessary to have access to the different states of the environment, i.e. when the actions are used sequentially, only one client is open and performs the calculations sequentially.

### Vectorized environment


Simulation training can be time consuming. It is therefore necessary to be able to parallelise the calculations. Since the actions are chosen sequentially, it is not possible to parallelise the calculations for one environment. The result depends on the previous result. However, it is possible to parallelise on several environments, meaning to run several simulations in parallel. This is done with the baseline of OpenAI: SubprocVecEnv.


### Separation between visualisation and computations

SofaGym separates calculations and visualisation. In order to achieve this, two scenes must be created: a scene *A* with all visual elements and a scene *B* with calculation elements (solvers, ...). Scene *A* is used in a viewer and scene *B* in the clients. Once the calculations have been performed in scene *B*, the positions of the points are given to the viewer which updates scene *A*.

### Adding new environment


It is possible to define new environments using SofaGym. For this purpose different elements have to be created:
- *NameEnv*: inherits from *AbstractEnv*. It allows to give the specificity of the environment like the action domain (discrete or continuous) and the configuration elements.
- *NameScene*: allows to create the Sofa scene. It must have the classic createScene function and return a *root*. To improve performance it is possible to separate the visual and computational aspects of the scene using the *mode* parameter (*'visu'* or *'simu'*). It allows you to choose the elements in the viewer-related scene or in the client-related scene. We also integrate two Sofa.Core.Controller (rewardShaper and goalSetter) that allow to integrate goal and reward in the scene.
- *NameToolbox*: allows to customize the environment. It defines the functions to retrieve the reward and the state of the environment as well as the command to apply to the system (link between the Gym action and the Sofa command). Note that we can define the Sofa.Core.Controller here. 

These different elements make it possible to create and personalise the task to be performed. See examples of environments for implementation.

## The environments

### Gripper

The  Gripper  Environmentoffers  two  different  scenes.   In  both  scenes,  the objective is to grasp a cube and bring it to a certain height.  The closer the cube is to the target, the greater the reward.

The two scenes are distinguished by their action  space.   In  one  case  the  actions  are  discrete and correspond to a particular movement. We define a correspondence between a Gym action (int) and corresponding Sofa displacement and direction.

In the second case,  the actions are continuous  and  correspond  directly  to  a  movement  ofthe gripper’s fingers.  This difference is indicated when defining the environment

```python
env = gym.make("continuegripper-v0")
```

### Trunk

The Trunk environment offers two scenarios.  Both are based on the trunk robot.  The first is to bring the trunk’s tip to a certain position.

```python
env = gym.make("trunk-v0")
```

The second scenario is to manipulate a cup using the trunk to get the cup’s center of gravity in a predefined position.

```python
env = gym.make("trunkcup-v0")
```

The  Trunk  is  controlled  by  eight  cables  that can be contracted or extended by one unit.  There are therefore 16 possible actions. The action space presented here is discrete but could easily be ex-tended to become continuous.


### MultiGait Robot

The multigait Softrobot has  one  scene. The goal is to move the robot forward in the *x* direction with the highest speed.

```python
env = gym.make("multigaitrobot-v0")
```

### Maze

The Maze environment offers one scene  of a ball navigating in a maze. The maze is attached to the tripod robot and the ball is moved by gravity by modifying the maze’s orientation.

```python
env = gym.make("maze-v0")
```

The tripod is actuated by three servomotors. Similarly to the Trunk Environment, the Maze environment has a dicrete action space of 6 actions, moving  each  servomotor  by  one  increment,  and could easily be extended to be continuous.
