#!/bin/bash

CARLA_HOME='/opt/carla-simulator/'
HOME=${CARLA_HOME}'PythonAPI/Fraunhofer_LAAS_Safety_Monitors/'
LOG_PATH=${HOME}'src/log/'
MAIN_PATH='src/main.py'
OUTPUT_DIR='src/scenario_test'
SCENARIO_PATH='src/pedestrian_crossing_between_cars0/scenario.xosc'
THREATS_FILE_PATH='src/evolutionary_step/evolutionary_results/selected_population_for_'


# PARAMS
day_times=('day')
# 'sunset' 'night'
threat='smoke' #'default'
ood_types=('noise')
# 'distributional_shift' 'anomaly' 'novelty' 'adversarial' 'occlusion'
# --threatspath "${THREATS_FILE_PATH}${ood_type}.npy"
# --use_expert_model --fault_type ${threat}
for day_time in "${day_times[@]}"; do
    for ood_type in "${ood_types[@]}"; do
        echo "*** STARTING simulation with daytime= $day_time ***" 
        >> ${LOG_PATH}log_client.log 
        
        cd ${CARLA_HOME}
        ./CarlaUE4.sh -opengl -quality-level=Epic SendNonPlayerAgentsInfo=true &
        sleep 4s 

        cd ${HOME}
        python3.8 ${MAIN_PATH} --output-dir ${OUTPUT_DIR} --fault_type ${threat} --augmented_data_percentage 1  --scenario-path ${SCENARIO_PATH} --time-of-day "$day_time"  --no-recording
        
        echo "*** ENDING simulation ***" 
        >> ${LOG_PATH}log_client.log 

        kill -SIGINT $(ps aux | grep '[C]arla' | awk '{print $2}')
        sleep 3s
        kill -SIGINT $(ps aux | grep '[C]arla' | awk '{print $2}')
        sleep 3s
    done
done