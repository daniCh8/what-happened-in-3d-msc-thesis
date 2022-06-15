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
    mv "${dir}sequence/" "${dir}old_sequence/"
    mkdir "${dir}sequence/"
    ((++i))
    echo "${i}/${tot}"
done