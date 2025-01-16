# Soft Robotics: Exploring Algorithms from "**SOFA-DR-RL: Training Reinforcement Learning Policies for Soft Robots with Domain Randomization in SOFA Framework**"

This repository is part of a **School Project** regarding **Reinforcement Learning** algorithms used in soft-robotics. The repository contains the results of our experiments done based on "**Domain Randomization for Robust, Affordable and Effective Closed-loop Control of Soft Robots**" (Gabriele Tiboni, Andrea Protopapa, Tatiana Tommasi, Giuseppe Averta - IROS2023)[[1]](#citation-1) The results we present focus on the PPO algorithms and ,additionally, we compare our results with the RFDROPO algorithm of the original creators. Our experiments are meant to validate the results from the original work. For additional info on their experiment, please visit their [Site](https://github.com/andreaprotopapa/sofa-dr-rl).
## Abstract
###Finish dano skusobny komit

## Table of Contents
1. [Task overview](#task)
2. [Theory](#theory)
3. [Experiments](#experiments)
4. [References](#references)

## Task
The task for our **School Project** was to explore soft-robotics algorithms, mainly Proximal Policy Optimization (PPO), and compare it to more advanced methods, specifically Reset Free Domain Randomization Off Policy Optimization (RFDROPO). 

## Theory
## Reinforcement Learning in Soft Robotics

Reinforcement Learning (RL) has emerged as a transformative approach for controlling soft robotic systems. Unlike traditional control methods, RL enables robots to learn and adapt their behaviors through interactions with their environments, eliminating the need for explicit programming of complex control strategies. This is particularly advantageous in soft robotics, where the continuous deformations and high-dimensional dynamics present significant challenges for conventional approaches.

State-of-the-art RL techniques, such as Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC), have shown great promise in addressing the unique demands of soft robotics. These algorithms excel in high-dimensional control scenarios, enabling soft robots to accomplish tasks such as locomotion, manipulation, and adaptive grasping. For example, RL has been leveraged to train soft robotic grippers to adjust their grasping strategies for diverse objects, including fragile or irregularly shaped items. Similarly, RL-based methods have been applied to soft robotic locomotion, allowing robots to navigate complex terrains by discovering efficient gaits and utilizing their inherent compliance to overcome obstacles.

Beyond task-specific applications, RL provides a framework for optimizing soft robotic behaviors in scenarios that require adaptability and precision. Through trial-and-error interactions in simulated environments, RL agents can uncover robust and efficient control policies that are resilient to environmental or task-related variations.

## Proximal Policy Optimization (PPO)

Proximal Policy Optimization (PPO) is a model-free reinforcement learning algorithm designed to balance learning stability and efficiency. It simplifies the optimization process by replacing the hard trust region constraint of Trust Region Policy Optimization (TRPO) with a clipping mechanism, which prevents large updates to the policy. This approach allows PPO to use first-order optimization methods, such as gradient descent, while maintaining robust performance and ease of implementation across a wide range of tasks.

## RF-DROPO: Handling Partially Observable Environments

RF-DROPO builds on the DROPO methodology to handle partially observable environments. It estimates posterior distributions over dynamics parameters using a likelihood-based objective function and open-loop action replay. This method is particularly effective for soft robotic tasks with high degrees of freedom.

## Experiments
The experiments were conducted using the simulation platform SOFA-Gym. In this environment we mainly focused on TrunkReach model training and testing. The **Training** used PPO algorithm for its **Policy** giving us limited results, compared to RFDROPO. The robotic arm reached the target only in very few cases, the training lasted for numerous hours and it takes a lot of time to reach the target. Using RFDROPO on the other hand, the robot needed much less training to get a faster and more precise results. 

### TrunkReach - PPO

<p align="center">
<img src=https://github.com/jakub-spisak/softrobotics_algorithms/blob/main/repo_assets/ppo_video.gif/>
</p>

### TrunkReach - RFDROPO

<p align="center">
  <img src=https://github.com/andreaprotopapa/sofa-dr-rl/assets/44071949/47170f5d-9b51-48db-9f42-0e61ff083476 alt="trunkreach" width="400"/>
</p>

Our experiments provided additional data regarding the deterministic and timeout attribute. The following graphs will shows the testing performance of the PPO algorithm. The bottom represents how many iterations it has been testen on, the side represents rewards. The rewards were given based on the distance from the target. The higher the better.

Deterministic: If True, the simulation will be deterministic, meaning that the same inputs will always generate the same outputs. If False, the simulation may include random elements. Our research has shown, that without the random element, the simulation reaches almost twice as good results. 

### Deterministic false
<p align="center">
<img src=https://github.com/jakub-spisak/softrobotics_algorithms/blob/main/repo_assets/reward_analysis_deterministic-false_sf-5_dt-0.04_timeout-10_.png/>
</p>

### Deterministic true
<p align="center">
<img src=https://github.com/jakub-spisak/softrobotics_algorithms/blob/main/repo_assets/reward_analysis_deterministic-true_sf-5_dt-0.04_timeout-10_.png.png/>
</p>

Timeout: The time limit (in seconds) after which the simulation will terminate if it does not reach its goal. Our experiments show, that longer timeout has a drastic effect on the performance of the algorithm.

### Timeout Value: 50

<p align="center">
<img src=https://github.com/jakub-spisak/softrobotics_algorithms/blob/main/repo_assets/reward_analysis_deterministic-true_sf-5_dt-0.01_timeout-50_.png/>
</p>

### Timeout Value: 10
<p align="center">
<img src=https://github.com/jakub-spisak/softrobotics_algorithms/blob/main/repo_assets/reward_analysis_deterministic-true_sf-5_dt-0.01_timeout-10.png/>
</p>

## References
1. [Domain randomization for robust, affordable and effective closed-loop control of soft robots](https://arxiv.org/pdf/2303.04136)[[1]](#citation-1)
---

### Citation

#### Citation 1
```bibtex
@inproceedings{tiboni2023domain,
  title={Domain randomization for robust, affordable and effective closed-loop control of soft robots},
  author={Tiboni, Gabriele and Protopapa, Andrea and Tommasi, Tatiana and Averta, Giuseppe},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={612--619},
  year={2023},
  organization={IEEE}
}
```
