#!/bin/bash

CARLA_HOME='/opt/carla-simulator/'
HOME=${CARLA_HOME}'PythonAPI/collab_laas/'
LOG_PATH=${HOME}'src/log/'
SCENARIO_EXECUTOR_PATH='src/server_ipc_yolo5.py'
OUTPUT_DIR='src/scenario_test'
SCENARIO_PATH='src/pedestrian_crossing_between_cars0/scenario.xosc'


# PARAMS
day_times=('day')
#  'sunset' 'night'
threats=('brightness')
# 'contrast' 'fog' 'snow' 'rain' 'sun_flare' 'gaussian_noise'
severity_types=('1' '2' '3')
# '3' '4' '5'


# iterates through all fault types and their intensity during day or night
for day_time in "${day_times[@]}"; do

    for threat in "${threats[@]}"; do

        for severity in "${severity_types[@]}"; do
            echo "Simulation with daytime= $day_time, OOD= $threat, and severity= $severity..."
            
            cd ${CARLA_HOME}
            ./CarlaUE4.sh -opengl -quality-level=Epic &
            sleep 4s 

            cd ${HOME}
            python3.8 ${SCENARIO_EXECUTOR_PATH} --output-dir ${OUTPUT_DIR} --integrate-object-detector --scenario-path ${SCENARIO_PATH} --time-of-day "$day_time" --fault_type "$threat" --severity "$severity" --no-recording  >> ${LOG_PATH}log_client.log
            
            kill -SIGINT $(ps aux | grep '[C]arla' | awk '{print $2}')
            sleep 3s
            kill -SIGINT $(ps aux | grep '[C]arla' | awk '{print $2}')
            sleep 3s
        done
        
    done

done