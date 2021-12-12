#! /bin/sh
cd ../
python run.py \
    -env 'SmallGridEnv' \
    -agent 'MCAgent' \
    -epsilon 1.0 \
    -alpha 0.1 \
    -gamma 0.995 \
    -n_episode 3000 \
    -seed 42 \
    -save_dir_plot './results/plot/' \
    -save_dir_value './results/q_value/'