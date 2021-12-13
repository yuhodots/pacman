#! /bin/sh
cd ../
python run.py \
    -env 'BigGridEnv' \
    -agent 'QlearningAgent' \
    -epsilon 1.0 \
    -alpha 0.1 \
    -gamma 0.999 \
    -n_episode 100000 \
    -memo '_1e+5epi'