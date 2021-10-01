#!/bin/bash

scenarios=(
'./crash_ahead.xosc'
)

parameters=(
'--time-of-day day --rain none --clouds none --fog none'
'--time-of-day day --rain heavy --clouds heavy --fog heavy'
)

for scenario in "${scenarios[@]}"
do
  IDX=0
  for params in "${parameters[@]}"
  do
    ../../../CarlaUE4.sh -quality-level=Epic &
    sleep 8
    PYTHONPATH=PYTHONPATH:../../carla ../../venv/bin/python ../scenario_executor.py --output-dir ${scenario}${IDX} --annotation-format coco --sensors rgb_camera event_camera semantic_segmentation_camera --scenario-path ${scenario} ${params}

    pkill Carla
    sleep 5
    pkill -9 Carla

    (( IDX = (IDX + 1) % ${#parameters[@]}))
    echo $IDX
  done
done
