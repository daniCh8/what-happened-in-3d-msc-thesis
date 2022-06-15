#!/bin/bash

cd /local/crv/danich/thesis
chmod +x ./unzip-sequences.sh && ./unzip-sequences.sh
chmod +x ./render-scenes.sh
chmod +x ./create-renderer-screens.sh && ./create-renderer-screens.sh
chmod +x ./move-sequences.sh && ./move-sequences.sh

cd /local/crv/danich/thesis/src/scripts
eval "$(conda shell.bash hook)"
conda activate thesisenvtorch1
python create_render_metadata.py
cd /local/crv/danich/thesis && ./create-renderer-screens.sh