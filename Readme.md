# Pac-man Bot

This repository contains the implementation of a simple reinforcement learning task. So it's not exactly the same as the original Pac-man game. In this environment, the episode ends when the agent gets all the stars (unlike the original Pac-man game that the episode ends when all coins are removed).

<img src = "./img/video-game.png" width="60%">

*<Taken from: https://pixabay.com/images/id-1332694/>*


## Contents

1. Value Iteration
2. Policy Iteration
3. Monte-Carlo Method
4. SARSA
5. Q Learning
6. Actor-Critic Method
7. REINFORCE

## Overview

### Environments

#### SmallGridEnv

In this environment, the ghost does not move. And since there is only one star, the episode ends when the agent arrives at the star. The figure below is a visualization of the environment through `visualize_matrix()`. Visualization is also possible through the `env.render()` function.

- Observation space: 5 x 5 grid world - wall positon 
  - `observation_space.n`  = (25 - 5) = *20*

- Action space: { up: 0, down: 1, left: 2, right: 3 }
- Reward: { ghost: -10, others: -1 }

![img](./img/SGE.png)

#### BigGridEnv

In this environment, the ghost randomly moves left and right. And since there are multiple stars, the episode ends when all the stars are obtained. The visualization method is the same as SmallGridEnv.

- Observation space: (11 x 11 grid world - wall positon) x (Star state) x (Ghost position state)
  - `observation_space.n` = (121 - 40) * (2^4) * (3 * 7) = *27216*

- Action space: { up: 0, down: 1, left: 2, right: 3 }
- Reward: { star: 50, ghost: -10, others: -1 }

![img](./img/BGE.png)

## How To Run

```sh
# TBA
```

## References

- Sutton, Richard S., and Andrew G. Barto. *Reinforcement learning: An introduction*. MIT press, 2018.
- [UNIST AI51201 Reinforcement Learning](https://sites.google.com/view/rl-unist-2021-fall/home), Instructor [Sungbin Lim](https://www.google.com/url?q=https%3A%2F%2Fsites.google.com%2Fview%2Fsungbin%2F&sa=D&sntz=1&usg=AFQjCNF8rjDRU3_7d8WL6v4kWLEzeyCZbw)

