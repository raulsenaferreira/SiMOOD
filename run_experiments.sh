#!/bin/bash

#HOME='/home/rsenaferre/Bureau/collab_laas/'
HOME='/opt/carla-simulator/PythonAPI/collab_laas/'
CARLA_HOME='/opt/carla-simulator/'
#SCENARIO_EXECUTOR_PATH='src/scenario_executor.py'
SCENARIO_EXECUTOR_PATH='src/server_ipc_yolo5.py'
OUTPUT_DIR='scenario_test'
SCENARIO_PATH='pedestrian_crossing_between_cars0/scenario.xosc'
INFERENCE_SCRIPT='src/yolov4_inference/serve_ipc.py'

# PARAMS
day_times=('day')
# 'night'
threats=('fog')
# 'rain'
severity_types=('1')
# '3'

# starts the object detection model in the server
#python3.8 ${INFERENCE_SCRIPT} &

# iterates through all fault types and their intensity during day or night
for day_time in "${day_times}"; do

    for threat in "${threats[@]}"; do

        for severity in "${severity_types[@]}"; do

            cd ${CARLA_HOME}
            ./CarlaUE4.sh -opengl &
            sleep 4s 

            cd ${HOME}
            python ${SCENARIO_EXECUTOR_PATH} --output-dir ${OUTPUT_DIR} --integrate-object-detector --scenario-path ${SCENARIO_PATH} --time-of-day "$day_time" --fault_type "$threat" --severity "$severity" --no-recording
            
            kill -SIGINT $(ps aux | grep '[C]arla' | awk '{print $2}')
            sleep 3s
            kill -SIGINT $(ps aux | grep '[C]arla' | awk '{print $2}')
        done
        
    done

done