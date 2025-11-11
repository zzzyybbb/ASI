# ASI

Code implementation of paper ``Enhancing Exploration and Exploitation in Hierarchical Reinforcement Learning with Subgoal Graph Learning'' (AAAI'26).

This code repository is adapted from [HESS](https://github.com/SiyuanLee/HESS).

## Prerequisites

* Python 3.6 or above
* [PyTorch == 1.4.0](https://pytorch.org/)
* [Gym](https://gym.openai.com/)
* [Mujoco](https://www.roboti.us/license.html)


#### Train the agent

```
python train_hier_sac.py --c 20 --abs_range 10  --env-name AntMaze1-v1 --test AntMaze1Test-v1 --weight_decay 1e-5 --device cuda:0 --seed 24
```

The parameter setting can be found in `./arguments`.
