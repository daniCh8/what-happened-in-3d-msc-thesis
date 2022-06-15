#!/bin/bash

for i in {0..10}
do
    screen -dmS "renderer_$i"
    screen -S "renderer_$i" -p 0 -X stuff "cd /local/crv/danich/thesis && ./render-scenes.sh $i 10 && exit\n"
done