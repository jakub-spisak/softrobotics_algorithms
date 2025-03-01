# Soft Robotics: Exploring Algorithms from "**SOFA-DR-RL: Training Reinforcement Learning Policies for Soft Robots with Domain Randomization in SOFA Framework**"

This repository is part of a **School Project** regarding **Reinforcement Learning** algorithms used in soft-robotics. The repository contains the results of our experiments done based on "**Domain Randomization for Robust, Affordable and Effective Closed-loop Control of Soft Robots**" (Gabriele Tiboni, Andrea Protopapa, Tatiana Tommasi, Giuseppe Averta - IROS2023)[[1]](#citation-1) The results we present focus on the PPO algorithms and ,additionally, we compare our results with the RFDROPO algorithm of the original creators. Our experiments are meant to validate the results from the original work. For additional info on their experiment, please visit their [Site](https://github.com/andreaprotopapa/sofa-dr-rl).

## Table of Contents
1. [Task overview](#task)
2. [Theory](#theory)
3. [Experiments](#experiments)
4. [References](#references)
5. [Conclusion](#conclusion)
## Task
The task for our **School Project** was to explore soft-robotics algorithms, mainly Proximal Policy Optimization (PPO), and compare it to more advanced methods, specifically Reset Free Domain Randomization Off Policy Optimization (RFDROPO). 

# Theory

## Reinforcement Learning in Soft Robotics

Reinforcement Learning (RL) has emerged as a transformative approach for controlling soft robotic systems. Unlike traditional control methods, RL enables robots to learn and adapt their behaviors through interactions with their environments, eliminating the need for explicit programming of complex control strategies. This capability is particularly advantageous in soft robotics, where continuous deformations and high-dimensional dynamics pose significant challenges for conventional approaches.

State-of-the-art RL techniques, such as **Proximal Policy Optimization (PPO)** and **Soft Actor-Critic (SAC)**, have demonstrated remarkable potential in addressing the unique demands of soft robotics. These algorithms excel in high-dimensional control scenarios, enabling soft robots to accomplish tasks such as locomotion, manipulation, and adaptive grasping. For instance, RL has been used to train soft robotic grippers to adapt their grasping strategies to handle diverse objects, including fragile and irregularly shaped items. Similarly, RL-based methods have been applied to soft robotic locomotion, allowing robots to navigate complex terrains by discovering efficient gaits and leveraging their compliance to overcome obstacles.

Beyond task-specific applications, RL provides a powerful framework for optimizing soft robotic behaviors in scenarios requiring adaptability and precision. Through trial-and-error interactions in simulated environments, RL agents uncover robust and efficient control policies resilient to environmental and task-related variations.

---

## Policy Gradient Methods

Policy gradient methods are a class of reinforcement learning techniques that optimize parameterized policies using gradient descent to maximize the expected return (long-term cumulative reward). Unlike traditional RL approaches, they avoid several common challenges, such as:

- The lack of guarantees for a value function.
- Difficulties caused by uncertain state information.
- Complexity in handling continuous states and actions.

These methods are particularly suited for high-dimensional, continuous control problems encountered in soft robotics.

---

## Trust Region Policy Optimization (TRPO)

**Trust Region Policy Optimization (TRPO)** is a reinforcement learning algorithm designed to improve policies by minimizing a surrogate objective function while ensuring stable and reliable policy updates. It leverages theoretical guarantees to ensure policy improvement with meaningful step sizes. TRPO is practical for optimizing complex, nonlinear policies with many parameters and has been successfully applied to tasks like locomotion and playing Atari games.

---

## Proximal Policy Optimization (PPO)

**Proximal Policy Optimization (PPO)** is a model-free reinforcement learning algorithm designed to balance learning stability and efficiency. It simplifies the optimization process by replacing TRPO’s hard trust region constraint with a clipping mechanism that prevents overly large updates to the policy. This design enables PPO to use first-order optimization methods, such as gradient descent, while maintaining robust performance and ease of implementation across a wide range of tasks.

### Limitations of PPO:

Despite its popularity as an on-policy reinforcement learning method, PPO has limitations:

- Frequent resets to a nominal start state hinder exploration and slow convergence.
- The absence of domain randomization reduces robustness in environments requiring broad adaptability, such as soft robotic tasks.

---

## RF-DROPO: Handling Partially Observable Environments

**RF-DROPO** builds upon the DROPO methodology to address partially observable environments. By estimating posterior distributions over dynamics parameters using a likelihood-based objective function and open-loop action replay, RF-DROPO is particularly effective for soft robotic tasks with high degrees of freedom.

### Key Components of RF-DROPO:

1. **Reset-Free (RF):**  
   Training continues from diverse intermediate trunk states rather than resetting after each episode. This promotes comprehensive exploration and reduces dependence on predefined starting conditions.

2. **Domain Randomization (DR):**  
   Systematic randomization of environment parameters (e.g., stiffness, friction) ensures the learned policy is robust and capable of handling a wide range of real-world conditions.

3. **Off-Policy Optimization (OPO):**  
   A replay buffer stores collected transitions, enabling policy updates using off-policy methods. This accelerates learning by reusing past data and efficiently integrating experiences from randomized domains.

## Experiments
The experiments were conducted using the SOFA-Gym simulation platform, with a primary focus on training and testing the TrunkReach model. The training utilized the PPO (Proximal Policy Optimization) algorithm for its policy, which yielded limited results compared to RF-DROPO (Reset-Free Domain Randomization Off-Policy Optimization).

With PPO, the robotic arm reached the target in very few cases, and the training process was time-consuming, lasting for numerous hours. The time required to reach the target was considerable, indicating inefficiency in the learning process. In contrast, when using RF-DROPO, the robot required much less training time to achieve faster and more precise results.

### TrunkReach - PPO

<p align="center">
<img src=https://github.com/jakub-spisak/softrobotics_algorithms/blob/main/repo_assets/ppo_video.gif/>
</p>

### TrunkReach - RFDROPO

<p align="center">
  <img src=https://github.com/andreaprotopapa/sofa-dr-rl/assets/44071949/47170f5d-9b51-48db-9f42-0e61ff083476 alt="trunkreach" width="400"/>
</p>

The inconsistency of the PPO algorithm is a major factor contributing to its performance limitations. This inconsistency stems from several factors, including its reliance on **On-Policy learning**. While adjusting some of its parameters did lead to mild improvements, these gains were often short-lived and couldn't match the overall effectiveness of **RF-DROPO**. RF-DROPO consistently outperformed PPO due to its more robust learning mechanisms, such as off-policy learning and reset-free training, which led to faster and more reliable results.

## Conclusion
We have compared standard on-policy Proximal Policy Optimization (PPO) with Reset-Free Domain Randomization Off-Policy Optimization (RF-DROPO) in the context of a trunk-like soft robotic reaching environment. While our focus has been on the training curves of the PPO baseline, we emphasize the core reasons for its limitations:

PPO often converges slowly and may plateau due to frequent resets, limited variability in the environment, and the inefficiency of discarding past experiences.
RF-DROPO overcomes these limitations by integrating reset-free training, domain randomization, and off-policy data reuse. This combination enables faster learning, increased robustness, and better adaptation to parameter changes.
Although our study involved a shorter learning phase, we successfully demonstrated the inherent drawbacks of PPO and achieved our goal of showing RF-DROPO's superiority across all metrics in the TrunkReach task. Future work could include side-by-side comparisons of training curves for PPO and RF-DROPO, along with additional metrics such as success rates and time-to-target. Ablation studies on various domain randomization strategies would also provide further insights.

The experiment is given further exploration in our [Research Paper](https://github.com/jakub-spisak/softrobotics_algorithms/tree/main/repo_assets/research_paper.pdf)
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
