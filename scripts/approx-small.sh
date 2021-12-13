#! /bin/sh
cd ../
python run.py \
    -env 'SmallGridEnv' \
    -agent 'LinearApprox' \
    -epsilon 0.1 \
    -alpha 0.01 \
    -gamma 0.999 \
    -n_episode 10000