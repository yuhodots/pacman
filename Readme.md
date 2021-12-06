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

I used this environment for experiments in situation of knowing the transition probabilities, such as value iteration.

- Observation space: 5 x 5 grid world
- Action space: { up: 0, down: 1, left: 2, right: 3 }
- Reward: { ghost: -100, others: -1 }

![img](./img/SGE.png)

#### BigGridEnv

I used this environment for experiments in situations where don't know the transition probabilities, such as MC method.

- Observation space: (11 x 11 grid world) x (Coin state) x (Star state) x (Ghost position state)
- Action space: { up: 0, down: 1, left: 2, right: 3 }
- Reward: { coin: 1, star: 50, ghost: -100, others: -1 }

![img](./img/BGE.png)

## How To Run

```sh
# TBA
```

## References

- Sutton, Richard S., and Andrew G. Barto. *Reinforcement learning: An introduction*. MIT press, 2018.
- [UNIST AI51201 Reinforcement Learning](https://sites.google.com/view/rl-unist-2021-fall/home), Instructor [Sungbin Lim](https://www.google.com/url?q=https%3A%2F%2Fsites.google.com%2Fview%2Fsungbin%2F&sa=D&sntz=1&usg=AFQjCNF8rjDRU3_7d8WL6v4kWLEzeyCZbw)

