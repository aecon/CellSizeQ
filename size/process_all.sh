#!/bin/bash
set -eu

inputdir0="/media/user/SSD1/Hao/aSynData"

directories=`ls -d ${inputdir0}/HA*`

for dir in ${directories}; do
    echo $dir
    platename=`basename $dir`
    echo $platename

    python process_FastParallel_hestia_cell_and_nuclei_area_diameter.py -n ${platename} -d ${dir}
done
