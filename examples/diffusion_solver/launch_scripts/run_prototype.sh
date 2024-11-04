#!/bin/bash

python ../workflow/launch_workflow_prototype.py \
    --num_phases 3 \
    --sim_time 10 \
    --epochs 5 \
    --al_func tod \
    --conda_env /eagle/RECUP/twang/env/rose-task-base-clone \
    --src_dir /eagle/RECUP/twang/rose/rose_github/examples/diffusion_solver/src/ \
    --project_id RECUP \
    --queue debug \
    --num_nodes 2
