import argparse
import json
import os
import pickle
from bisect import bisect
from datetime import datetime

import pandas as pd

from privacyschedulingtools.total_weighted_completion_time.entity.domain import IntegerDomain
from privacyschedulingtools.total_weighted_completion_time.entity.job import Job
from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import ParallelScheduleFactory
from privacyschedulingtools.total_weighted_completion_time.entity.scheduling_parameters import \
    ParallelSchedulingParameters
from pup_parallel_scripts.ExperimentRunner import ExperimentRunnerFactory, ResultCollectorFactory

"""
    Start through command line:
    python3 pup_parallel_real_data.py <job_data> <setup> --output <output_file>
    where <job_data> is a JSON file including the collection of jobs to schedule
    and <setup> is a JSON file including the setup for the experiment
    and <output_file> is the path to the output file for saving the schedule
"""

floor2chairs = {
    6: 28,
    7: 33,
    8: 28,
    9: 28,
    10: 26,
    11: 15
}

floor2rooms = {
    6: 10,
    7: 18,
    8: 19,
    9: 22,
    10: 19,
    11: 19
}


def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Creates schedules from job data')

    # Adding arguments
    parser.add_argument('job_data', nargs="?", type=str, default='pup_parallel_scripts/real_data/job_data', help='File to job data as JSON file')
    parser.add_argument('setup', nargs="?", type=str, default='pup_parallel_scripts/real_data/setup.json', help="Path to setup JSON file")
    parser.add_argument('--output', '-o', type=str, default='pup_parallel_scripts/real_data/schedules.schdls',
                        help='Output file path')

    # Parse the arguments
    args = parser.parse_args()

    # Accessing the parsed arguments
    input_file = args.job_data
    setup_file = args.setup
    output_file = args.output

    # compute schedules for each collection of jobs

    # if jobs_to_schedule is a directory, iterate through files, else collect from file
    if os.path.isdir(input_file):
        schedules = collect_schedules_from_dir(input_file, setup_file)
    else:
        schedules = collect_schedules_from_file(input_file, setup_file)

    # save jobs to schedules file
    with open(output_file, 'wb') as file:
        pickle.dump(schedules, file)

    exp_runner = ExperimentRunnerFactory.create_from_setup_file(setup_file)
    exp_runner.run_experiment(schedules, output_file)
    res_collector = ResultCollectorFactory.create_from_setup_file(exp_runner.setup_file)
    res_collector.collect_results(schedules)


def collect_schedules_from_file(input_file, setup_file):
    with open(input_file, 'r') as file:
        jobs_to_schedule = json.load(file)
    # Path to JSON file of the format:
    preprocess(jobs_to_schedule)
    domains = get_domains(jobs_to_schedule, setup_file)
    schedules = []

    for batch in jobs_to_schedule:
        machine_count = batch["available_machines"]
        jobs = []

        for job in batch["jobs"]:
            jobs.append(Job(job["job_id"], job["processing_time"], job["weight"]))

        params = ParallelSchedulingParameters(job_count=len(jobs),
                                              machine_count=machine_count,
                                              processing_time_domain=domains[1],
                                              weight_domain=domains[0])
        schedule = ParallelScheduleFactory.generate_schedule_from_jobs(jobs, machine_count, params)
        schedules.append((batch["schedule_id"], schedule))

    return schedules


def preprocess(jobs_to_schedule):
    for batch in jobs_to_schedule:
        for job in batch["jobs"]:
            job["processing_time"] = preprocess_duration(job["processing_time"])
            job["weight"] = preprocess_weight(job["weight"])


def preprocess_duration(duration):
    # TODO: implement to adapt the durations
    return duration


def preprocess_weight(weight):
    # TODO: implement to adapt the weights
    return weight


def get_domains(schedules, setup_file) -> (IntegerDomain, IntegerDomain):
    with open(setup_file, 'r') as file:
        setup = json.load(file)

    if "scheduling_parameters" not in setup or setup["scheduling_parameters"] is None:
        return extract_domains(schedules)
    else:
        weight_domain = setup["scheduling_parameters"]["weight_domain"]
        processing_time_domain = setup["scheduling_parameters"]["processing_time_domain"]
        return IntegerDomain(weight_domain[0], weight_domain[1]), IntegerDomain(processing_time_domain[0],
                                                                                processing_time_domain[1])


def extract_domains(schedules):
    weights = {
        "min": None,
        "max": None
    }

    durations = {
        "min": None,
        "max": None
    }

    for batch in schedules:
        for job in batch["jobs"]:
            if job["weight"] < weights["min"] or weights["min"] is None:
                weights["min"] = job["weight"]
            if job["weight"] > weights["max"] or weights["max"] is None:
                weights["max"] = job["weight"]
            if job["processing_time"] < durations["min"] or durations["min"] is None:
                durations["min"] = job["processing_time"]
            if job["processing_time"] > durations["max"] or durations["max"] is None:
                durations["max"] = job["processing_time"]

    return IntegerDomain(weights["min"], weights["max"]), IntegerDomain(durations["min"], durations["max"])


def collect_schedules_from_dir(directory, setup_file):
    # iterate through files, each CSV file represents the batch to be scheduled for one day:
    # schedule id: floor-day-month-year
    #
    # read for each job:
    #   job id (incremental counter)
    #   processing time: scheduled_duration_min
    #   weight: UNIX timestamp -> ranked -> binned ranking = weight
    #   create job objects -> schedule jobs

    schedules = []  # List to store schedules generated from extracted jobs

    # Here, we require domains to be provided in the setup file
    with open(setup_file, 'r') as file:
        setup = json.load(file)

    if setup is None or setup["scheduling_parameters"] is None:
        raise Exception("Setup file must include scheduling parameters")

    # Assuming integer domains to be given
    # TODO: implement for "discrete" domain?

    weight_domain = IntegerDomain(setup["scheduling_parameters"]["weight_domain"][0],
                                  setup["scheduling_parameters"]["weight_domain"][1])
    processing_time_domain = IntegerDomain(setup["scheduling_parameters"]["processing_time_domain"][0],
                                           setup["scheduling_parameters"]["processing_time_domain"][1])

    # Iterate through CSV files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path, header=0)

            id_counter = 0  # Incremental job counter
            schedule_id = None
            floor = None
            jobs = []

            df = df.reset_index()
            # Generate weights from timestamps by binning them s.t. they are within weight domain
            #min_scheduled_time = df["scheduled_time"].min()
            #max_scheduled_time = df["scheduled_time"].max()
            # Generate weights from UNIX timestamps
            #df['weight'] = df["scheduled_time"].apply(
            #    lambda x: transform_to_weight(x, min_scheduled_time, max_scheduled_time, weight_domain.get_max()))

            # Generate weights from timestamps by ranking them and then binning the ranks s.t. within weight domain
            df['weight'] = df['scheduled_time'].rank(method='dense', ascending=True).astype(int)
            min_rank = df["weight"].min()
            max_rank = df["weight"].max()
            df['weight'] = df["weight"].apply(
                lambda x: transform_to_weight(x, min_rank, max_rank, weight_domain.get_max()))

            for index, row in df.iterrows():
                # Extracting fields (day, month, year, floor) to create schedule ID
                if schedule_id is None:
                    schedule_id = extract_schedule_id(row['arrival_date'], row['floor_id'])
                if floor is None:
                    floor = int(row['floor_id'])

                jobs.append(Job(id_counter, int(row['scheduled_duration_min']), row['weight']))

                # Increment job id counter
                id_counter += 1

            params = ParallelSchedulingParameters(job_count=len(jobs),
                                                  machine_count=floor2chairs[floor],
                                                  processing_time_domain=processing_time_domain,
                                                  weight_domain=weight_domain)
            schedule = ParallelScheduleFactory.generate_schedule_from_jobs(jobs, floor2chairs[floor], params)
            schedules.append((schedule_id, schedule))

    return schedules


def extract_schedule_id(arrival_date_field, floor):
    # TODO: please adjust date format here if necessary
    arrival_date = datetime.strptime(arrival_date_field.split(" ")[0], '%d-%b-%y').date()
    schedule_id = f"{arrival_date.isoformat()}-{floor}"
    return schedule_id


def transform_to_weight(value, min_t, max_t, scaling_factor=10):
    #return (value - min_t) / (max_t - min_t) * (scaling_factor - 1) + 1
    # divide range of values (e.g. scheduled_time or ranks) into scaling_factor bins
    # and return to which one value belongs
    bins = [min_t + i * (max_t - min_t) / scaling_factor for i in range(1, scaling_factor)]
    return scaling_factor - bisect(bins, value)  # highest weight assigned to lowest timestamps


if __name__ == "__main__":
    main()
