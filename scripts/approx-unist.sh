#! /bin/sh
cd ../
python run.py \
    -env 'UnistEnv' \
    -agent 'LinearApprox' \
    -epsilon 0.1 \
    -alpha 0.01 \
    -gamma 1.0 \
    -n_episode 3000 \
    -step_ghost True