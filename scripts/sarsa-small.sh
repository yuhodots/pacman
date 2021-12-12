#! /bin/sh
cd ../
python run.py \
    -env 'SmallGridEnv' \
    -agent 'SARSAAgent' \
    -epsilon 1.0 \
    -alpha 0.1 \
    -gamma 0.99 \
    -n_episode 30000 \
    -seed 42 \
    -save_dir_plot './results/plot/' \
    -save_dir_value './results/q_value/'