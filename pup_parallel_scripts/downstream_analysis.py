import argparse
import csv
import os
import pickle
import re
from statistics import mean
from scipy.stats import wasserstein_distance
#from dit.divergences import earth_movers_distance

from privacyschedulingtools.total_weighted_completion_time.pup.utility_functions.parallel_machine_utilities import \
    get_waiting_time, get_processing_time, get_completion_time, \
    get_tasks_per_machine, compute_utility_difference, get_values, get_machine_completion_times, \
    compute_element_wise_distance

#     avg / max number of tasks per machine
#     avg / max job completion time
#     avg / max waiting time
#     avg / max processing time
#     avg / max machine completion time


property_functions = {
    "waiting_time": get_waiting_time,
    "processing_time": get_processing_time,
    "job_completion_time": get_completion_time,
    "tasks": get_tasks_per_machine,
    "machine_completion_times": get_machine_completion_times
}

aggregate_functions = {
    "avg": mean,
    "max": max,
    "emd": wasserstein_distance
}

location2path = {
    "07312023_104612/calculate_avg_wait_time": "pup_parallel_scripts/Results/calculate_avg_wait_time",
    "08092023_182230/calculate_avg_wait_time": "pup_parallel_scripts/Results/calculate_avg_wait_time_alt",
    "07312023_104612/calculate_twct": "pup_parallel_scripts/Results/calculate_twct",
    "08082023_180209/calculate_twct": "pup_parallel_scripts/Results/calculate_twct_alt"
}


def collect_downstream_analysis_data(directory_path, property_name, aggregate_name):
    utility_functions = get_utility_functions(directory_path)

    if utility_functions:
        for utility in utility_functions:
            directory_name = os.path.basename(os.path.normpath(directory_path))
            u_path = os.path.join(directory_path, utility)
            output_dir = location2path[directory_name + "/" + utility]
            #output_path = os.path.join(u_path, f"{aggregate_name}_{property_name}.csv")
            output_path = os.path.join(output_dir, f"{aggregate_name}_{property_name}.csv")

            fieldnames = ["schedule_id", "utility_function", "utility_threshold", "privacy_threshold",
                          "transformation", "outcome", "utility_loss", "privacy_loss",
                          "downstream_analysis", "dt_original", "dt_anonymized", "dt_difference"]

            with open(output_path, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)

                # Write the header
                writer.writeheader()

                for root, dirs, files in os.walk(u_path):
                    # Check if the directory has no subdirectories (so only contains result files)
                    if not dirs:
                        pattern = r"u0\.\d{1,3}_p0\.\d{1,3}/.*"
                        match = re.search(pattern, root)

                        if match:
                            privacy_threshold, transformation, utility_threshold = extract_pup_params(match)
                        else:
                            print("No match found.")
                            continue

                        # Writing rows one by one
                        for file in files:
                            if not file.endswith("timeout") and not file.endswith("exhausted"):
                                write_data(property_name, aggregate_name, file, privacy_threshold, root,
                                           transformation, utility, utility_threshold, writer)


def write_data(property_name, aggregate_name, file, privacy_threshold, root, transformation, utility,
               utility_threshold, writer):
    schedule_id = file.split("_")[0]
    result = load_pickled_file(os.path.join(root, file))
    original_schedule = result.original_schedule
    anonymized_schedule = result.privatized_schedule
    outcome = result.outcome
    utility_loss = result.utility_loss
    privacy_loss = result.privacy_loss
    dt_anonymized, dt_original, dt_difference = compute_downstream_performance(property_name, aggregate_name,
                                                                               original_schedule, anonymized_schedule)
    csv_data = {
        "outcome": str(outcome).replace("Outcome.", ""),
        "utility_loss": utility_loss,
        "utility_threshold": utility_threshold,
        "privacy_loss": privacy_loss,
        "privacy_threshold": privacy_threshold,
        "downstream_analysis": f"{aggregate_name}_{property_name}",
        "dt_original": dt_original,
        "dt_anonymized": dt_anonymized,
        "dt_difference": dt_difference,
        "schedule_id": schedule_id,
        "utility_function": utility,
        "transformation": transformation
    }
    # write stats to csv file
    writer.writerow(csv_data)


def compute_downstream_performance(property_name, aggregate_name, original_schedule, anonymized_schedule):
    property_function = property_functions[property_name]

    if aggregate_name in ["Euclidian", "Manhattan", "Chebyshev"]:
        machine_lvl = property_name == "tasks" or property_name == "machine_completion_times"

        if property_name == "tasks":
            property_function = lambda m, s: len(m)
        elif property_name == "machine_completion_times":
            property_function = lambda m, s: m[-1].start_time + s.jobs[m[-1].id].processing_time if len(m) > 0 else 0

        return None, None, compute_element_wise_distance(original_schedule, anonymized_schedule, property_function, p=aggregate_name,
                                      machine_lvl=machine_lvl)

    if aggregate_name in ["MAE"]:
        machine_lvl = property_name == "tasks" or property_name == "machine_completion_times"

        if property_name == "tasks":
            property_function = lambda m, s: len(m)
        elif property_name == "machine_completion_times":
            property_function = lambda m, s: m[-1].start_time + s.jobs[m[-1].id].processing_time if len(m) > 0 else 0

        abs_err = compute_element_wise_distance(original_schedule, anonymized_schedule, property_function, p="Manhattan",
                                      machine_lvl=machine_lvl)
        mae = abs_err / len(original_schedule.jobs)

        return None, None, mae
    

    if property_name == "tasks" or property_name == "machine_completion_times":
        orig_values = property_function(original_schedule)
        anon_values = property_function(anonymized_schedule)
    elif property_name in ["job_completion_time", "waiting_time", "processing_time"]:
        orig_values = get_values(original_schedule, property_function)
        anon_values = get_values(anonymized_schedule, property_function)
    else:
        raise ValueError(f"Property {property_name} not supported.")

    if aggregate_name == "avg" or aggregate_name == "max":
        aggr_function = aggregate_functions[aggregate_name]
        dt_original = aggr_function(orig_values)
        dt_anonymized = aggr_function(anon_values)
        dt_difference = compute_utility_difference(dt_original, dt_anonymized)
        return dt_original, dt_anonymized, dt_difference
    elif aggregate_name in ["emd"]:
        dt_difference = wasserstein_distance(orig_values, anon_values)
        return None, None, dt_difference
    elif aggregate_name in ["emd*"]:
        # orig_count = Counter(orig_values)
        # anon_count = Counter(anon_values)
        #
        # o_values = []
        # a_values = []
        # o_weights= []
        # a_weights = []
        # 
        # o_total = sum(orig_count.values())
        # a_total = sum(anon_count.values())
        #
        # for elem, count in orig_count.items():
        #     o_values.append(elem)
        #     o_weights.append(count/o_total)
        #
        # for elem, count in anon_count.items():
        #     a_values.append(elem)
        #     a_weights.append(count/a_total)
        #
        # orig_weights, _ = np.histogram(orig_values, bins=max(orig_values)+1, density=True)
        # anon_weights, _ = np.histogram(anon_values, bins=max(anon_values)+1, density=True)
        #
        # orig_range = np.arange(max(orig_values)+1)
        # anon_range = np.arange(max(anon_values)+1)

        dt_difference = wasserstein_distance(orig_values, anon_values)
        #dt_difference = wasserstein_distance(o_values, a_values, o_weights, a_weights)
        # dt_difference2 = wasserstein_distance(orig_range, anon_range, orig_weights, anon_weights)
        return None, None, dt_difference
    else:
        raise ValueError(f"Aggregate {aggregate_name} not supported.")


def extract_pup_params(match):
    extracted_part = match.group()
    params = extracted_part.split("/")
    transformation = params[1]
    up_params = [p[1:] for p in params[0].split("_")]
    utility_threshold = up_params[0]
    privacy_threshold = up_params[1]
    return privacy_threshold, transformation, utility_threshold


def get_utility_functions(directory_path):
    utility_functions = []
    try:
        files = os.listdir(directory_path)

        for file in files:
            if os.path.isdir(os.path.join(directory_path, file)):
                print(file)
                utility_functions.append(file)
    except:
        FileNotFoundError
    return utility_functions


def load_pickled_file(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


if __name__ == "__main__":
    # Command-line argument parsing
    #parser = argparse.ArgumentParser(
    #    description='Compute downstream analysis measures of original and anonymized schedules')
    #parser.add_argument('directory', type=str, help='Path to the directory')
    # parser.add_argument('property', type=str, help='The property to compute downstream analysis for (waiting_time, processing_time, completion_time)')
    # parser.add_argument('aggregate', type=str, help='The aggregate to compute downstream analysis for (avg, max, distr)')
    #args = parser.parse_args()

    # Check if the provided directory exists
    #if not os.path.exists(args.directory):
    #    print(f"The directory {args.directory} does not exist.")
    #    exit()

    # Call the function to copy files based on schedules
    # collect_downstream_analysis_data(args.directory, args.property, args.aggregate)

    # properties = ["waiting_time", "processing_time", "job_completion_time", "tasks", "machine_completion_times"]
    # aggregates = ["avg", "max", "emd"]
    PATH_TO_SCHEDULES = "/Users/maikebasmer/Projects/schedule_privacy_results/anonymized_schedules"
    for directory_name in ["07312023_104612", "08092023_182230", "08082023_180209"]:
        directory = os.path.join(PATH_TO_SCHEDULES, directory_name)
        properties = ["waiting_time", "processing_time", "job_completion_time", "machine_completion_times", "tasks"]
        #aggregates = ["Euclidian", "Manhattan", "Chebyshev", "avg", "max", "emd", "MAE"]
        aggregates = ["emd*"]
        for prop in properties:
            for aggregate in aggregates:
                collect_downstream_analysis_data(directory, prop, aggregate)
