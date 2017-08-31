#!/bin/bash

# @ wall_clock_limit=23:59:59
# @ initialdir=.
# @ job_name=TASK
# @ output=serial.out
# @ error=serial.err
# @ total_tasks=1



for net in ./TRAIN/*.prototxt;
do
     name="${net##*/}"
     [[ $name =~ ^DONE* ]] && continue
     folder=$name;
     [[ $name =~ ^[0-9] ]] && folder=${name:1:40}
     mkdir TRAINING/$folder
     cp plot.py TRAINING/$folder/.
     mv TRAIN/$name TRAIN/DONE$name
     mnsubmit train.sh
     ../../build/tools/caffe_double train -solver TRAIN/DONE$name 2>TRAINING/$folder/$name.log >TRAINING/$folder/$name.debug
done
