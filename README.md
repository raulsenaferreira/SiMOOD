## SiMOOD: Evolutionary Testing Simulation with Out-Of-Distribution Images

Important notice: this is an experimental image transformation tool for CARLA simulator. It was developed in an academic context. Therefore, several code improvements and cleaning are needed to be considered a mature tool. Feel free to contribute and open new issues.

This repository is divided into two parts: A) Setup for OOD perturbations live in CARLA B) Evolutionary search for OOD perturbations.

Details about the concept, architecture and experiments can be [seen in our paper](https://hal.science/hal-03779723/) presented at the 27th IEEE Pacific Rim International Symposium on Dependable Computing (PRDC 2022).
 

#### A) Setup for CARLA simulator with real-time perturbations

1. Install CARLA. The scenario generator has been tested with version 0.9.11
2. Clone this repository into the PythonAPI directory of your CARLA installation.
3. Create a new virtual environment and install dependencies with `pip install -r requirements`. It has been tested with Python 3.7 and 3.8.
4. Install CARLA's python package via the provided .egg file `easy_install ../carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg`. 
If you use another Python version maybe you have to build the package yourself from CARLA's sources.
5. Execute this line to correctly set your python path and test if you can execute the scenario_executor script (if it shows some options on your terminal it means that is OK): 
PYTHONPATH=PYTHONPATH:/opt/carla-simulator/PythonAPI/carla/ ../path_to_your_virtual_env/bin/python src/scenario_executor.py -h
6. Run the src/main.py file, after choosing the OOD perturbations and levels that you want to see in the simulation. 
You also have the option to chose the perturbations chosen by the evolutionary algorithm search (to run the evolutionary algorithm search, please follow the part B of this README).

Example of the two command lines that runs the CARLA server & CARLA simulation with smoke level 0.5 and YOLO as an object detector:

`./CarlaUE4.sh -opengl -quality-level=Epic`. You can change from -opengl to the -Vulkan backend if you want to.

`python3.8 src/main.py --output-dir src/scenario_test --scenario-path src/pedestrian_crossing_between_cars0/scenario.xosc --time-of-day day --fault_type smoke=0.5 --object_detector_model_type yolo  --no-recording --execution_mode single`

### Additional details

To get all configurable parameters run `PYTHONPATH=PYTHONPATH:../carla ../venv/bin/python scenario_executor.py -h`. For instance, to generate a scenario at sunset with heavy rain and an overcast sky run `PYTHONPATH=PYTHONPATH:../carla ../venv/bin/python scenario_executor.py --output-dir scenario1 --scenario-path example_scenarios/group_of_cyclists.xosc --time-of-day sunset --rain heavy --clouds heavy`

If you haven't placed your virtual environment in the PythonAPI directory, you have to specify the correct path in the above command
If you haven't cloned this repository in the PythonAPI directory, you have to adjust the PYTHONPATH in the above command

See also `example_scenarios/batch_scenario_execution.sh` for further usage

See the scripts under example_scenarios/ for generating data for a batch of different scenario configurations

#### B) Setup for running the evolutionary search for OOD perturbations
1. go to the folder src/evolutionary_steps
2. Create your virtual environment (for example python -m venv v_env) and activate it
3. pip install -r requirements.txt
4. Run the file main.py ... You can modify the different parts in this file to run the GA algorithm with the params as you prefer.
5. Further details can be seen at its own [README](https://github.com/raulsenaferreira/SiMOOD/blob/main/src/evolutionary_steps/README.md). 

#### Troubleshooting
1. After starting the server, and the client, the client screen keeps black with no rendering image:
- It is recommended to restart the CARLA server after each run to avoid this problem.

### Possible customizations

1. To create additional scenario templates refer to [CARLA ScenarioRunner](https://github.com/carla-simulator/scenario_runner/)
2. Custom ego vehicle behavior can be defined by extending/overriding CustomAgent or Basic Agent in, e.g., `custom_agent.py`
3. To modify the data recording and environment setup (e.g. weather) look into `scenario_generation_and_recording.py`


### License

ScenarioRunner-specific code is distributed under MIT License.

CARLA-specific code is distributed under MIT License.

CARLA-specific assets are distributed under CC-BY License.

The ad-RSS-lib library compiled and linked by the RSS Integration build variant introduces LGPL-2.1-only License.

Note that UE4 itself follows its license terms.
