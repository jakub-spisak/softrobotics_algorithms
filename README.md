# Soft Robotics: Exploring Algorithms from "**SOFA-DR-RL: Training Reinforcement Learning Policies for Soft Robots with Domain Randomization in SOFA Framework**"
###Finish
This repository is part of a **School Project** regarding **Reinforcement Learning** algorithms used in soft-robotics. The repository contains the results of experiments done according to "**Domain Randomization for Robust, Affordable and Effective Closed-loop Control of Soft Robots**" (Gabriele Tiboni, Andrea Protopapa, Tatiana Tommasi, Giuseppe Averta - IROS2023)[[1]](#citation-1) The results we present focus on the PPO algorithms being compared to RF-DROPO. Our experiments are meant to validate the result of the original creators. 
## Abstract
###Finish

## Table of Contents
1. [References](#references)

1. **ResetFree-DROPO** (**RF-DROPO**): Our method, developed as an extension of [DROPO](https://github.com/gabrieletiboni/dropo). In this approach, we relax the original assumption of resetting the simulator to each visited real-world state. Instead, we consider that we only know the initial full configuration of the environment, and actions are replayed in an open-loop fashion, always starting from the initial state configuration. For further details, please refer to Sec. IV-A in our [paper](https://arxiv.org/abs/2303.04136).

### 4. Evaluation
To evaluate the effectiveness of various methods in a Sim-to-Real setting, it is common practice to start with a Sim-to-Sim scenario. This allows us to test the transferability of learned policies using simulation alone. To do this, we initially worked in a source environment where the dynamics parameters were unknown. Our aim was to estimate an optimal policy that would be suitable for the unknown target domain.
Subsequently, we can now evaluate the learned policy by applying it to a target simulated environment with the nominal target dynamics parameters that we attempted to infer during the inference phase.

## Examples
### TrunkReach

<p align="center">
  <img src=https://github.com/andreaprotopapa/sofa-dr-rl/assets/44071949/47170f5d-9b51-48db-9f42-0e61ff083476 alt="trunkreach" width="400"/>
</p>

### TrunkPush

<p align="center">
  <img src=https://github.com/andreaprotopapa/sofa-dr-rl/assets/44071949/87781dcb-ca14-487e-b276-f47795910501 alt="trunkpush" width="400"/>
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
