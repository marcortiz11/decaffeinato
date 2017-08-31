#!/bin/bash

# @ wall_clock_limit=23:59:59
# @ initialdir=.
# @ job_name=2TASK
# @ output=serial.out
# @ error=serial.err
# @ total_tasks=1

mkdir TRAINING/SOLVER_SR\[2,14].prototxt

../../build/tools/caffe_double train -solver solvers/SOLVER_SR\[14].prototxt -weights HEEEEY_iter_6000.caffemodel 2>TRAINING/SOLVER_SR\[2,14].prototxt/3SOLVER_SR\[2,14]_AFTER_FLOATING[2,5].prototxt
