# Soft Robotics: Exploring Algorithms from "**SOFA-DR-RL: Training Reinforcement Learning Policies for Soft Robots with Domain Randomization in SOFA Framework**"
###Finish
This repository is part of a **School Project** regarding **Reinforcement Learning** algorithms used in soft-robotics. The repository contains the results of experiments done according to "**Domain Randomization for Robust, Affordable and Effective Closed-loop Control of Soft Robots**" (Gabriele Tiboni, Andrea Protopapa, Tatiana Tommasi, Giuseppe Averta - IROS2023)[[1]](#citation-1) The results we present focus on the PPO algorithms being compared to RF-DROPO. Our experiments are meant to validate the result of the original creators. 
## Abstract
###Finish

## Table of Contents
1. [References](#references)

### 2. Dynamics Parameters Inference
We offer two distinct methods for inferring the dynamics parameters:

1. **ResetFree-DROPO** (**RF-DROPO**): Our method, developed as an extension of [DROPO](https://github.com/gabrieletiboni/dropo). In this approach, we relax the original assumption of resetting the simulator to each visited real-world state. Instead, we consider that we only know the initial full configuration of the environment, and actions are replayed in an open-loop fashion, always starting from the initial state configuration. For further details, please refer to Sec. IV-A in our [paper](https://arxiv.org/abs/2303.04136).

2. **[BayesSim](https://github.com/rafaelpossas/bayes_sim/tree/master)**: This method represents the classical baseline in Domain Randomization, adapted here to the offline inference setting by replaying the original action sequence during data collection.

Both of these methods are accessible within the `sb3-gym-soro/methods` directory.
As the output, we generate a distribution of the dynamics parameters saved in an `.npy` file. You can refer to the `sb3-gym-soro/BestBounds` directory to access previous inference results that we have made available.

### 3. Policy Training
The primary objective of Domain Randomization is to randomly sample new dynamics parameters, denoted as $\xi$, from the distribution $p_\phi(\xi)$ at the beginning of each training episode. If an inference algorithm like *RF-DROPO* or *BayesSim* has been used, then $p_\phi(\xi)$ represents the output from the previous step.

Additionally, we have included another baseline method known as **Uniform Domain Randomization** (**UDR**). Unlike the aforementioned inference-based approaches, UDR does not require an inference step, as $p_\phi(\xi)$ is a uniform distribution that is statically fixed in the randomized configuration file of the environment.

Upon training the agent in the source environment for a specified number of `timesteps`, the optimal policy is obtained as output and is saved in `best_model.zip`.

### 4. Evaluation
To evaluate the effectiveness of various methods in a Sim-to-Real setting, it is common practice to start with a Sim-to-Sim scenario. This allows us to test the transferability of learned policies using simulation alone. To do this, we initially worked in a source environment where the dynamics parameters were unknown. Our aim was to estimate an optimal policy that would be suitable for the unknown target domain.
Subsequently, we can now evaluate the learned policy by applying it to a target simulated environment with the nominal target dynamics parameters that we attempted to infer during the inference phase.

## Examples
Notes:
- Each of the following examples should be executed within the training directory `sb3-gym-soro`. Therefore, please ensure that you change the current working directory to this location (i.e., `cd sb3-gym-soro`).
- Our toolkit is integrated with `wandb`. If you wish to use it, remember to log in beforehand and include the corresponding option in the command (i.e., `--wandb_mode online`).
- To parallelize the inference or policy training execution, use the dedicated `--now` parameter.
- Please note that both the *inference* phase and *policy training* are relatively time-consuming experiments required to reach convergence. If you are primarily interested in our results, you can quickly evaluate some pre-trained policies that we have made available in the `sb3-gym-soro/example-results` directory or following the commands reported in **Evaluation**.
  - During the evaluation of a learned policy, it is possible to visualize the execution of the task with the option `--test_render`.
- Additionally, the datasets and distributions of dynamics parameters that have already been inferred are provided in the `sb3-gym-soro/Dataset` and `sb3-gym-soro/BestBounds` directories, respectively.
### TrunkReach

<p align="center">
  <img src=https://github.com/andreaprotopapa/sofa-dr-rl/assets/44071949/47170f5d-9b51-48db-9f42-0e61ff083476 alt="trunkreach" width="400"/>
</p>

For this task, we offer various methods for training with Domain Randomization, including *RF-DROPO* (our method), *BayesSim*, and *UDR*. To keep it simple, we will provide example commands for *RF-DROPO* here. However, you can refer to the in-code documentation of each method if you wish to try them as well.
- **Inference**
  - Dataset is here collected by executing a set of 100 random actions before the inference phase.
  - ```
    python train_dropo.py --env trunk-v0 --test_env trunk-v0 --seed 0 --now 1 -n 1 --budget 5000 --data random --clipping 100 --inference_only --run_path ./runs/RFDROPO --wandb_mode disabled
    ```
- **Policy Training**
  - Inference bounds (i.e., the dynamics parameters distributions) have here already been determined in a previous inference step and are simply loaded.
  - ```
    python train_dropo.py --env trunk-v0 --test_env trunk-v0 --seed 0 --now 1 -t 2000000  --training_only --run_path ./runs/RFDROPO --bounds_path ./BestBounds/Trunk/RFDROPO/seed0_8CK3V_best_phi.npy --wandb_mode disabled
    ```
- **Evaluation** (suggested for an out-of-the-box testing)
  - A control policy has here already been trained in a previous policy training step and is simply loaded.
  - ```
    python test.py --test_env trunk-v0 --test_episodes 1 --seed 0 --offline --load_path ./example-results/trunk/RFDROPO/2023_02_28_20_31_32_trunk-v0_ppo_t2000000_seed2_login027851592_TM84F --test_render
    ```
### TrunkPush

<p align="center">
  <img src=https://github.com/andreaprotopapa/sofa-dr-rl/assets/44071949/87781dcb-ca14-487e-b276-f47795910501 alt="trunkpush" width="400"/>
</p>

For this task, we offer various methods for training with Domain Randomization, including *RF-DROPO* (our method), *BayesSim*, and *UDR*. To keep it simple, we will provide example commands for *RF-DROPO* here. However, you can refer to the in-code documentation of each method if you wish to try them as well.

It is also possible to train on an unmodeled setting, by using the option `--unmodeled`, which referers to the use of a different randomized configuration file (i.e., `TrunkCube_random_unmodeled_config.json`).

- **Inference**
  -  Dataset has here been pre-collected by a semi-converged policy and is simply loaded.
  - ```
    python train_dropo.py --env trunkcube-v0 --test_env trunkcube-v0  --seed 0 --now 1 -eps 1.0e-4 -n 1 --budget 5000 --data custom --data_path ./Dataset/TrunkCube/20230208-091408_1episodes.npy --inference_only --run_path ./runs/RFDROPO --wandb_mode disabled
    ```
- **Policy Training**
  - Inference bounds (i.e., the dynamics parameters distributions) have here already been determined in a previous inference step and are simply loaded.
  - ```
    python train_dropo.py --env trunkcube-v0 --test_env trunkcube-v0 --seed 0 --now 1 -t 2000000  --training_only --run_path ./runs/RFDROPO --bounds_path ./BestBounds/TrunkCube/RFDROPO/bounds_A1S0X.npy --wandb_mode disabled
    ```
- **Evaluation** (suggested for an out-of-the-box testing)
  - A control policy has here already been trained in a previous policy training step and is simply loaded.
  - ```
    python test.py --test_env trunkcube-v0 --test_episodes 1 --seed 0 --offline --load_path ./example-results/trunkcube/RFDROPO/2023_07_10_11_34_58_trunkcube-v0_ppo_t2000000_seed1_7901a3c94a22_G0QXG --test_render
    ```
## References
1. [Domain randomization for robust, affordable and effective closed-loop control of soft robots](#https://arxiv.org/pdf/2303.04136) [1](#citation-1)
2. [SofaGym: An OpenAI Gym API for SOFASimulations](#citation-2)
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
#### Citation 2
```bibtex
@misc{SofaGym,
  authors = {Ménager, Etienne and Schegg, Pierre and Duriez, Christian and Marchal, Damien},
  title = {SofaGym: An OpenAI Gym API for SOFASimulations},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
}
```

