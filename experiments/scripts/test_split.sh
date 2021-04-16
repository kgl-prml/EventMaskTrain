#!/bin/bash

export PYTHONPATH="`pwd`:${PYTHONPATH}"
if [ $# != 4 ]
then 
  echo "Please specify 1) cfg; 2) gpus; 3) exp_name; 4) split."
  exit
fi

cfg=${1}
gpus=${2}
exp_name=${3}
split=${4}

out_dir=./experiments/ckpt/${exp_name}
if [ -d ${out_dir} ]
then
  rm -rf ${out_dir}
fi
mkdir -p ${out_dir}

CUDA_VISIBLE_DEVICES=${gpus} python ./tools/test.py --cfg ${cfg} \
           --exp_name ${exp_name} --set DATASET.TEST_SPLIT_NAME ${split} 2>&1 | tee ${out_dir}/log.txt
