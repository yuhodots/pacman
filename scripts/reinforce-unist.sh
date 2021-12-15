#! /bin/sh
cd ../
python run.py \
    -env 'UnistEnv' \
    -agent 'REINFORCE' \
    -epsilon 0.1 \
    -alpha 0.01 \
    -gamma 0.99 \
    -n_episode 5000 \
    -step_ghost True