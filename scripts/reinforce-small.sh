#! /bin/sh
cd ../
python run.py \
    -env 'SmallGridEnv' \
    -agent 'REINFORCE' \
    -epsilon 0.1 \
    -alpha 0.01 \
    -gamma 0.99 \
    -n_episode 3000