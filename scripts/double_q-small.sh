#! /bin/sh
cd ../
python run.py \
    -env 'SmallGridEnv' \
    -agent 'DoubleQlearningAgent' \
    -epsilon 1.0 \
    -alpha 0.1 \
    -gamma 0.999 \
    -n_episode 5000