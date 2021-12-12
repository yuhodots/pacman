#! /bin/sh
cd ../
python run.py \
    -env 'BigGridEnv' \
    -agent 'SARSAAgent' \
    -epsilon 1.0 \
    -alpha 0.1 \
    -gamma 0.99 \
    -n_episode 1000000 \
    -seed 42 \
    -save_dir './results/'