#!/bin/bash

path="/local/crv/danich/thesis/3Rscan/data/3rscan/"
tot=0
for dir in ${path}*/
do
    ((++tot))
done

i=0
for dir in ${path}*/
do
    if [ ! -d "${dir}sequence/" ]; then
        mkdir "${dir}sequence/"
        unzip "${dir}sequence.zip" -d "${dir}sequence/" > /dev/null
    fi
    ((++i))
    echo "${i}/${tot}"
done