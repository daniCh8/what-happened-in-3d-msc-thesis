#!/bin/bash

batch=$1
step=$2

path="/local/crv/danich/thesis/3Rscan/data/3rscan/"
tot=0
scan_arr=()
for dir in ${path}*/
do
    scan=${dir%*/}
    scan="${scan##*/}"
    scan_arr+=(${scan})
    ((++tot))
done

batches=$(($tot/$step))

min=$(($batch*$batches))
max=$(((($batch+1)*$batches)-1))
if [ $step -eq $batch ]; then
    max=$tot
fi

for ((i=min;i<max;i++));
do
    export DISPLAY=:0
    cd "/local/crv/danich/renderer/3RScan/c++/rio_renderer/build/"
    ./rio_renderer_render_all ../../../../../3rscan/ ${scan_arr[$i]} sequence 0
    echo "${i}/${max}"
done