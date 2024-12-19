# privacy-scheduling-tools

A set of scripts for privacy attacks on schedules and utility-aware privacy protection.

## Overview of the project

The project contains scripts for the Privacy-and-Utility-Preservation Problem in the context of the TWCT and MAKE scheduling problem, which can be found in `privacyschedulingtools.total_weighted_completion_time`.

`entity` contains the entities for the jobs, schedules, outcomes and adversaries. The (parallel) schedule entity can be used to generate schedules from given jobs or to randomly generate schedules given the domains for schedule features. 
Different types of domains can be used. The `parallel_adversary` instantiates the adversary for parallel TWCT schedules, whereas `parallel_make_adversary` instantiates the adversary for parallel MAKE schedules.

`pup` contains everything regarding the privacy-and-utility preservation (PUP) problem, in particular the PUP `solver` entities along with `callback` and `utility` functions and some helper functions in `util`.

The directory `callback` contains the callback functions. 
The callback function is called when a solution to the inverse scheduling attack is found. Currently, there are four callback functions, all with the same inverse scheduling attack: 
* `distribution` keeps track of the decision variable (weights) distributions
* `solution_size` keeps track of the solution size
* `solution_size_w_original` keeps track of the solution size and whether the original weight vector has been found yet.
* `solution_collector` collects all solutions i.e. weight vectors that solve the ISP (up to a maximum count if given)

In `solver`, you can find the solvers for the PUP. They all are breadth-first search approaches operating on a search space induced by a set of perturbation functions applied to the original schedule, e.g., swapping jobs or changing the processing times.
For each neighbor, it is checked whether the schedule is a solution to the PUP. 
For the parallel setting, two solvers are most relevant:
* `pup_parallel` is a PUP solver for parallel TWCT schedules that uses a selection of perturbation functions to define the neighborhood.
* `pup_parallel_make` is the pup_parallel solver adapted to the MAKE problem.

`utility_functions` contains helper functions to measure the utility of a schedule, like the total weighted completion time or average wait time.
`util` contains helper functions for the general PUP setting.

`pup_parallel_scripts` contains the scripts for the experimental setup that considers scheduling jobs on parallel (identical) machines.
It contains the following scripts that were used in the experimental evaluation:
* `pup_parallel_synthetic-data.py` is used to run the experiments for synthetic data.
* `pup_parallel_real-data.py` is used to run the experiments for the real-world data.
* `plot_experiment_results.py` is used to plot the results of the experiments.
* `ExperimentRunner.py` is the "engine" for running experiments, i.e., it defines how the experiments are executed and how the results are collected

# How to set up the project

On macOS and Linux, to set up the project you have to:
1. Get Python, written and tested with Python 3.9.5
2. Create a virtual environment with `python3 -m venv env`
3. Activate the virtual environment with `source env/bin/activate`
4. Install needed packages with `python3 -m pip install -r requirements.txt`
5. Set pythonpath with `export PYTHONPATH=.`
 
**Relevant packages**

1. `ortools` to solve constraint satisfaction problems
2. `xxhash` for the hashing of schedules to avoid duplicate work

# Run experiments

## ...with synthetic data
To use the script <code>pup_parallel_synthetic-data.py</code>,
you may supply three arguments:
- path to JSON file with experimental setup (optional, otherwise default values are used)
- path to .schdls file with scheduling problems (optional, otherwise random schedules complying with the parameters given in the setup file will be generated and output path must be provided)
- path to output directory for generated schedules (optional for generation of schedules, otherwise input data must be provided)

For example:
`python3 pup_parallel_scripts/pup_parallel_synthetic-data.py pup_parallel_scripts/synthetic_data/make_problem_cmax_utility_setup_test.json pup_parallel_scripts/synthetic_data/schedules/08122024_155845_m4_n5-20_p5-20_w0-9_r10_schedules.schdls`

## ... with real-world data
To use the script <code>pup_parallel_real-data.py</code>,
you may supply three arguments:
- path to JSON file containing job data or directory with JSON files containing job data
- path to JSON file with experimental setup
- path to output file for generated schedules (optional, default: `pup_parallel_scripts/real_data/schedules.schdls`)

### Input format for real-world data
The real-world input data includes a list of scheduling problems that should be solved w.r.t. a given objective:
  - schedule id
  - number of available (identical) machines
  - list of jobs characterized by
    - job id
    - weight
    - processing time

Each scheduling problem will be solved according to the specific scheduling problem.
The input data must be provided as a JSON file of the following format:

    [
         {
          "schedule_id": ...,
          "available_machines": ...,
          "jobs": [
              {
                  "job_id": ...,
                  "weight": ...,
                  "processing_time": ...
              },
                ...
          ]
         },
         ...
     ]

See <code>pup_parallel_scripts/test_data/test_job_data.json</code> as an example.


## Parameters for experimental setup
- "trials": number of runs/schedules (will be set to the number of schedules generated from the job data)
- "time_limit": time (in s) until PUP runs into timeout
  - default: 300
- "num_workers": number of workers to parallelize the execution of the experiments (not the PUP itself)
- "source_path": path to where the output of the experiments should be stored (e.g. <code>real_data/experiment_results</code>)
- "transformations": list of transformations to consider 
  - options: ["PROCESSING_TIMES", "SWAPPING_JOBS", "MOVE", "MOVE_PROC", "SWAP_PROC", "ALT_MOVE_PROC", "ALT_MOVE", "SWAP_ALL", "SWAP_ALL_PROC", "SWAP_MACHINE", "SWAP_ORDER"]
  - selection:  ["PROCESSING_TIMES", "MOVE", "MOVE_PROC", "SWAP_ALL", "SWAP_ALL_PROC"]
- "utility_functions": utility functions that are applied during PUP
  - options: ["calculate_avg_wait_time", "calculate_twct", "calculate_cmax", "calculate_tct", "calculate_avg_wait_time_with_release_dates"]
  - selection: ["calculate_avg_wait_time", "calculate_twct"]
- "privacy_thresholds": privacy thresholds applied during PUP
  - default: [0.01, 0.5],
- "utility_thresholds": utility thresholds applied during PUP
  - default: [0.05, 0.2]
- "scheduling_parameters": provides min and max values of the weight domain and processing time domain, respectively (both integer)
  - if not provided, the parameters are extracted from the data set (esp. in case of real world data)
- "schedule_type": 
  - options: ["make", "wspt"]
  - default: "wspt"

See <code>pup_parallel_scripts/test_data/test_setup.json</code> for an example.

# Experimental results
The experimental results for synthetic data can be found in `pup_parallel_scripts/Results` along with the schedules (`.schdls`file) and the setup files used.

The plots are generated using the script `pup_paralell_scripts/plot_experiment_results.py` and can be found in `pup_parallel_scripts/plots`. 

# Contributors
* Ali Kaan Tutak
* Alexandra Tichauer
* Maike Basmer
* Stephan Fahrenkrog-Petersen
* Matthias Weidlich
* Arik Senderovich