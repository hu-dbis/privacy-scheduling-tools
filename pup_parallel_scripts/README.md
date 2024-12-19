# How to Run Experiments

## Which script to use...
### ...with synthetic data
To use the script <code>pup_parallel_real-data.py</code>,
you need to supply three arguments:
- path to file with job data
- path to file with experimental setup
- path to output file for generated schedules

### ... with real-world data
To use the script <code>pup_parallel_real-data.py</code>,
you need to supply three arguments:
- path to file with job data
- path to file with experimental setup
- path to output file for generated schedules (optional, default: <code>pup_parallel_scripts/real_data/schedules.schdls</code>)

## Input Data
The input data includes a list of scheduling problems that should be solved w.r.t. a given objective:
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
  - default: 100
- "num_workers": number of workers to parallelize the execution of the experiments (not the PUP itself)
- "source_path": path to where the output of the experiments should be stored (e.g. <code>real_data/experiment_results</code>)
- "transformations": list of transformations to consider 
  - options: ["PROCESSING_TIMES", "SWAPPING_JOBS", "MOVE", "MOVE_PROC", "SWAP_PROC", "ALT_MOVE_PROC", "ALT_MOVE", "SWAP_ALL", "SWAP_ALL_PROC", "SWAP_MACHINE", "SWAP_ORDER"]
  - selection:  ["PROCESSING_TIMES", "MOVE", "MOVE_PROC", "SWAP_ALL", "SWAP_ALL_PROC"]
- "utility_functions": utility functions that are applied during PUP
  - options: ["calculate_avg_wait_time", "calculate_twct", "calculate_cmax", calculate_tct"]
  - selection: ["calculate_avg_wait_time", "calculate_twct"]
- "privacy_thresholds": privacy thresholds applied during PUP
  - default: [0.01, 0.5],
- "utility_thresholds": utility thresholds applied during PUP
  - default: [0.05, 0.2]
- "scheduling_parameters": provides min and max values of the weight domain and processing time domain, respectively (both integer)
  - if not provided, the parameters are extracted from the data set 
- "schedule_type": 
  - options: ["make", "twct"]
  - default: 

See <code>pup_parallel_scripts/test_data/test_setup.json</code> for an example.