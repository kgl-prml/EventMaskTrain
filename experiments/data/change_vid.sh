#!/bin/bash

vids=`ls ./videos`

for vid in ${vids}
do
  new_vid=`echo ${vid} | sed "s/_/\./g"`
  echo ${new_vid}
  mv ./videos/${vid} ./videos/${new_vid}
done
