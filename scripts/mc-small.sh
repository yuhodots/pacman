#! /bin/sh
cd ../
python run.py \
    -env 'SmallGridEnv' \
    -agent 'MCAgent' \
    -epsilon 1.0 \
    -alpha 0.1 \
    -gamma 0.995 \
    -n_episode 30000 \
    -seed 42 \
    -save_dir './results/'