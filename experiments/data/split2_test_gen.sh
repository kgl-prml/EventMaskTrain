#!/bin/bash

all=`cat ./video_list_train.txt`
split2_train=`cat ./split2_train.lst`

rm ./split2_test.lst
for vid in ${all}
do
  res=`echo ${split2_train} | grep ${vid}`
  if [ x"$res" = x"" ]
  then
    echo ${vid} >> ./split2_test.lst
  fi
done
