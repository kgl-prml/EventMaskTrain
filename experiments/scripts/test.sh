#!/bin/bash

export PYTHONPATH="`pwd`:${PYTHONPATH}"
if [ $# != 3 ]
then 
  echo "Please specify 1) cfg; 2) gpus; 3) exp_name."
  exit
fi

cfg=${1}
gpus=${2}
exp_name=${3}

out_dir=./experiments/ckpt/${exp_name}
if [ -d ${out_dir} ]
then
  rm -rf ${out_dir}
fi
mkdir -p ${out_dir}

CUDA_VISIBLE_DEVICES=${gpus} python ./tools/test.py --cfg ${cfg} \
           --exp_name ${exp_name} 2>&1 | tee ${out_dir}/log.txt
