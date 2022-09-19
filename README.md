## QUICK Setup

1. Install CARLA. The scenario generator has been tested with version 0.9.11
2. Clone this repository into the PythonAPI directory of your CARLA installation
3. Create a new virtual environment and install dependencies with `pip install -r requirements`. It has been tested with Python 3.7 and 3.8
4. Install CARLA's python package via the provided .egg file `easy_install ../carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg`. If you use another Python version maybe you have to build the package yourself from CARLA's sources
5. Execute this line to correctly set your python path and test if you can execute the scenario_executor script (if it show some options on your terminal it means that is OK): 
PYTHONPATH=PYTHONPATH:/opt/carla-simulator/PythonAPI/carla/ ../collab_laas/VENV/bin/python src/scenario_executor.py -h
6. execute the main file using the OOD perturbations and levels that were outputted from the evolutionary step [to execute the evolutionary step, please go to the folder src/evolutionary_step].

Example of command line that runs the carla simulator with smoke level 0.5 with yolo as object detector:

python3.8 src/main.py --output-dir src/scenario_test --scenario-path src/pedestrian_crossing_between_cars0/scenario.xosc --time-of-day day --fault_type smoke=0.5 --object_detector_model_type yolo  --no-recording --execution_mode single



## Usage

1. Start the CARLA server with `./CarlaUE4.sh`. It is best to keep the default options and use the Vulkan backend, as the OpenGL backend has some problems, e.g., with rendering fog properly
2. Navigate to the subdirectory in PythonAPI where you have placed the contents of this repository
3. Generate a specific scenario by running `PYTHONPATH=PYTHONPATH:../carla ../venv/bin/python scenario_executor.py --output-dir scenario1 --scenario-path example_scenarios/crash_ahead.xosc`. This will open another window rendering the scenarios output and afterwards will store all recorded data in the specified output directory
    - The scenario is configured via a cli. To get all configurable parameters run `PYTHONPATH=PYTHONPATH:../carla ../venv/bin/python scenario_executor.py -h`. For instance, to generate a scenario at sunset with heavy rain and overcast sky run `PYTHONPATH=PYTHONPATH:../carla ../venv/bin/python scenario_executor.py --output-dir scenario1 --scenario-path example_scenarios/group_of_cyclists.xosc --time-of-day sunset --rain heavy --clouds heavy`
    - If you haven't placed your virtual environment in the PythonAPI directory, you have to specify the correct path in the above command
    - If you haven't cloned this repository in the PythonAPI directory, you have to adjust the PYTHONPATH in the above command
    - See also `example_scenarios/batch_scenario_execution.sh` for further usage
4. See the scripts under example_scenarios/ for generating data for a batch of different scenario configurations and for integrating an object detector (latter is experimental)


## Troubleshooting

- It is recommended to restart the CARLA server after each run to avoid any problems


## Modification

1. To create additional scenario templates refer to [CARLA ScenarioRunner](https://github.com/carla-simulator/scenario_runner/)
2. Custom ego vehicle behavior can be defined by extending/overriding CustomAgent or Basic Agent in, e.g., `custom_agent.py`
3. To modify the data recording and environment setup (e.g. weather) look into `scenario_generation_and_recording.py`


## License

ScenarioRunner specific code is distributed under MIT License.

CARLA specific code is distributed under MIT License.

CARLA specific assets are distributed under CC-BY License.

The ad-rss-lib library compiled and linked by the RSS Integration build variant introduces LGPL-2.1-only License.

Note that UE4 itself follows its own license terms.







