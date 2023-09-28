#!/bin/bash

cd $SLURM_SUBMIT_DIR
nsys profile --stats=true ./compare &> prof
# ./compare &> compare_out
# python compare_plot.py
