# SofaGym

**SofaGym** is a toolkit that helps you turn **SOFA** simulation scenes into environments for **Reinforcement Learning (RL)**, using the **OpenAI Gym API**. It allows you to train RL algorithms on simulations like robotic movements and other physics-based tasks.

## Key Features:

- **Gym-like Interface**: Use actions, states, rewards, and resets similar to OpenAI Gym.
- **Server/Worker Architecture**: Simulates actions efficiently by distributing tasks between multiple workers to avoid performance loss.
- **Parallel Environments**: You can run multiple simulations at once to speed up training.
- **Separate Visual and Computational Scenes**: Visual elements are in one scene, calculations are in another for better performance.

## Creating a New Environment:
To create a new environment in SofaGym, you need to set up:
1. **Action Space**: Define how actions are made (discrete or continuous).
2. **Scene Setup**: Create the simulation scene with SOFA's configuration.
3. **Customization**: Define how rewards and states are computed.

## Example Environment:

### Trunk

The Trunk environment offers two scenarios.  Both are based on the trunk robot.  The first is to bring the trunk’s tip to a certain position. The second scenario is to manipulate a cup using the trunk to get the cup’s center of gravity in a predefined position. The  Trunk  is  controlled  by  eight  cables  that can be contracted or extended by one unit.  There are therefore 16 possible actions. The action space presented here is discrete but could easily be ex-tended to become continuous.
