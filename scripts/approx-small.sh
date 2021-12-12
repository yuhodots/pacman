#! /bin/sh
cd ../
python run.py \
    -env 'SmallGridEnv' \
    -agent 'LinearApprox' \
    -epsilon 0.1 \
    -alpha 0.01 \
    -gamma 1.0 \
    -n_episode 30000 \
    -seed 42