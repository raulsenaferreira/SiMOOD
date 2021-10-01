#!/bin/bash

scenarios=(
'./crash_ahead.xosc'
)

parameters=(
'--time-of-day day --rain none --clouds none --fog none'
)


for scenario in "${scenarios[@]}"
do
  for params in "${parameters[@]}"
  do
    ../../../CarlaUE4.sh &
    #PYTHONPATH=PYTHONPATH:../../yolov4_inference ../../yolov4_inference/venv/bin/python ../../yolov4_inference/serve_ipc.py &
    PYTHONPATH=PYTHONPATH:../../yolov4_inference ../../yolov4_inference/serve_ipc.py &
    sleep 8
    #PYTHONPATH=PYTHONPATH:../../carla ../../venv/bin/python ../scenario_executor.py --output-dir none --no-recording --integrate-object-detector --scenario-path ${scenario} ${params}
    PYTHONPATH=PYTHONPATH:../../carla ../scenario_executor.py --output-dir none --no-recording --integrate-object-detector --scenario-path ${scenario} ${params}

    pkill Carla
    pkill -P $$
    sleep 5
    pkill -9 Carla
    pkill -9 -P $$
  done
done
