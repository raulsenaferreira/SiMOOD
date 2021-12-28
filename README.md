# Repository of the paper "Safety enveloping of neural architectures"


## Usage

## Setup

1. Install CARLA. The scenario generator has been tested with version 0.9.11
2. Clone this repository into the PythonAPI directory of your CARLA installation
3. Create a new virtual environment and install dependencies with `pip install -r requirements`. It has been tested with Python 3.7 and 3.8
4. Install CARLA's python package via the provided .egg file `easy_install ../carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg`. If you use another Python version maybe you have to build the package yourself from CARLA's sources
5. Execute this line to correctly set your python path and test if you can execute the scenario_executor script (if it show some options on your terminal it means that is OK): 
PYTHONPATH=PYTHONPATH:/opt/carla-simulator/PythonAPI/carla/ ../collab_laas/VENV/bin/python src/scenario_executor.py -h
6. execute ./run_experiments.sh for starting the experiments (you can edit this file if you want to change paths for your application)


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







