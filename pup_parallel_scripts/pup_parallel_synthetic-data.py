import argparse
from pup_parallel_scripts.ExperimentRunner import ExperimentRunnerFactory, ResultCollectorFactory

"""
    Start through command line:
    python3 pup_parallel_synthetic_data.py <job_data> <setup> --output <output_file>
    where <job_data> is a JSON file including the collection of jobs to schedule
    and <setup> is a JSON file including the setup for the experiment
    and <output_file> is the path to the output file for saving the schedule
"""

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Creates schedules from job data')

    # Adding arguments
    parser.add_argument('setup', nargs="?", type=str, help="Path to setup JSON file")
    parser.add_argument('schedules', nargs="?", type=str, help='Path to file containing scheduling problems')
    parser.add_argument('--output', '-o', type=str, help='Output file path for schedules')

    # Parse the arguments
    args = parser.parse_args()

    # Accessing the parsed arguments
    input_file = args.schedules
    setup_file = args.setup
    output_file = args.output

    assert setup_file is not None
    if output_file is None and input_file is None:
        print("No schedules and no output path given")
        return

    # Executing experiment
    # TODO: make sure in setup that correct transformations are used.
    print("Creating Experiment \n")
    exp_runner = ExperimentRunnerFactory.create_from_setup_file(setup_file)

    if input_file is None:
        print("Create schedules and run experiments... \n")
        exp_runner.generate_schedules_and_run_experiment(output_file)
        print(f"Finished experiment... \n")
    else:
        print("Load schedules and run experiments... \n")
        exp_runner.load_schedules_and_run_experiment(input_file)
        print(f"Finished experiment... \n")

    print("Collecting Results... \n")
    res_collector = ResultCollectorFactory.create_from_setup_file(exp_runner.setup_file)
    res_collector.load_schedules_and_collect_results()
    print(f"Done...")


if __name__ == "__main__":
    main()
