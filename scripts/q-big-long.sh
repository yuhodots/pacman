#! /bin/sh
cd ../
python run.py \
    -env 'BigGridEnv' \
    -agent 'QlearningAgent' \
    -epsilon 1.0 \
    -alpha 0.1 \
    -gamma 0.999 \
    -n_episode 10000000 \
    -seed 42 \
    -save_dir './results/' \
    -memo '_1e+7epi'