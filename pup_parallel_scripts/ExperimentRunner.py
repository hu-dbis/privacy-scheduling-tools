import concurrent
import csv
import itertools
import json
import os
import pickle
import typing
from copy import deepcopy
from datetime import datetime

from pebble import ProcessPool

from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import ParallelSchedule
from privacyschedulingtools.total_weighted_completion_time.entity.result import ParallelResult, Outcome
from privacyschedulingtools.total_weighted_completion_time.pup.solver.pup_parallel import PupParallel
from privacyschedulingtools.total_weighted_completion_time.pup.solver.pup_parallel_make import \
    PupParallelMake
from privacyschedulingtools.total_weighted_completion_time.pup.util.transformations import \
    Transformation
from privacyschedulingtools.total_weighted_completion_time.pup.util.file_utilities import pickle_data, \
    load_pickled_data, write_json_data
from privacyschedulingtools.total_weighted_completion_time.pup.util.random_schedule_generation import \
    RandomParameterGenerator
from privacyschedulingtools.total_weighted_completion_time.pup.util.stopping_criteria import found_empty_and_nonempty
from privacyschedulingtools.total_weighted_completion_time.pup.utility_functions import parallel_machine_utilities
from privacyschedulingtools.total_weighted_completion_time.pup.utility_functions.parallel_machine_utilities import \
    calculate_avg_wait_time, calculate_twct, calculate_cmax, calculate_avg_wait_time_with_release_dates

"""
    Record the outcome of the PUP applied to synthetically generated schedules using the specified distortion mechanism 

    Evaluation parameters:
        time_limit: Time (in s) available to explore solution space, Timeout if time_limit has passed
        stopping_criterion: Specifies condition upon which result collection is stopped
        num_worker: Number of threads that are used

    PUP parameters:
        transformations: distortion mechanism that is to be used
        utility_thresholds: values for maximally allowed utility loss that are tested  
        utility_functions: functions to assess quality of resulting schedule
        privacy_thresholds: values for maximally allowed privacy loss that are tested  
"""


class ExperimentRunnerFactory:
    @staticmethod
    def create_from_setup_file(setup_file):
        with open(setup_file, "r") as s_file:
            setup = json.load(s_file)

        setup["transformations"] = [Transformation[t] for t in setup["transformations"]]
        setup["utility_functions"] = [getattr(parallel_machine_utilities, f) for f in setup["utility_functions"]]
        #setup.pop("scheduling_parameters")

        return ExperimentRunner(**setup)

# TODO: add make/twct option to setup file -> based on flag,
# SOURCE_PATH = "output_transformations/comparison_transformations"
class ExperimentRunner:
    def __init__(self,
                 trials: int = 1000,
                 time_limit: int = 300,
                 num_workers: int = 1,
                 source_path: str = None,
                 transformations=None,
                 utility_functions=None,
                 utility_thresholds=None,
                 privacy_thresholds=None,
                 stopping_criterion=found_empty_and_nonempty,
                 schedule_type="wspt",
                 scheduling_parameters=None):
        self.trials = trials
        self.time_limit = time_limit
        self.num_workers = num_workers
        self.source_path = source_path
        self.date_prefix = f"{datetime.now().strftime('%m%d%Y')}_{datetime.now().strftime('%H%M%S')}"
        self.base_path = os.path.join(source_path, self.date_prefix)
        os.mkdir(self.base_path)

        self.privacy_thresholds = [0.01, 0.1, 0.5] if privacy_thresholds is None \
            else privacy_thresholds  # allowed checker‚ success probability -> 1%, 10%, 50%
        self.utility_thresholds = [0.005, 0.01, 0.02] if utility_thresholds is None else utility_thresholds
        self.utility_functions = [calculate_cmax, # calculate_twct,
                                  calculate_avg_wait_time_with_release_dates] if utility_functions is None else utility_functions
        self.transformations = [Transformation.SWAP_ALL, Transformation.ALT_MOVE, Transformation.ALT_MOVE_PROC,
                                Transformation.SWAP_ALL_PROC] if transformations is None else transformations
        self.stopping_criterion = stopping_criterion
        self.setup_file = None
        self.schedule_type = schedule_type
        self.scheduling_parameters = scheduling_parameters

    def generate_schedules_from_setup(self, output_path):
        generator = RandomParameterGenerator(trials=self.trials,
                                             processing_time_range=tuple(self.scheduling_parameters["processing_time_domain"]),
                                             weight_range=self.scheduling_parameters["weight_domain"],
                                             schedule_type=self.schedule_type)

        generator.generate_random_parameter_list()
        file_name = generator.create_file_name()
        generator.save_schedule_stats(os.path.join(output_path,
                                                   file_name + "_schedule_params.csv"))
        schedules = generator.generate_schedules_from_parameters()
        path_to_schedules = os.path.join(output_path,
                              file_name + "_schedules.schdls")
        generator.save_schedules(path_to_schedules)
        return schedules, path_to_schedules

    def generate_schedules_and_run_experiment(self, output_path):
        schedules, path_to_schedules = self.generate_schedules_from_setup(output_path)
        self.run_experiment(schedules, path_to_schedules)

    def load_schedules_and_run_experiment(self,
                                          schedules_path):
        assert schedules_path is not None
        schedules = load_pickled_data(schedules_path)

        self.run_experiment(schedules, schedules_path)

    def run_experiment(self,
                       schedules, schedules_path=None):

        assert schedules is not None

        self.setup_file = os.path.join(self.base_path, f"{self.date_prefix}_experimental_setup.json")
        self.trials = len(schedules)

        # setup file that can be used to collect the results and reproduce the experiment
        write_json_data(self.setup_file,
                        {
                            "schedules": schedules_path,
                            "utility_functions": [u.__name__ for u in self.utility_functions],
                            "source_path": self.source_path,
                            "base_path": self.base_path,
                            "date": self.date_prefix,
                            "workers": self.num_workers,
                            "trials": self.trials,
                            "time_limit": self.time_limit,
                            "utility_thresholds": self.utility_thresholds,
                            "privacy_thresholds": self.privacy_thresholds,
                            "transformations": [str(t).replace("Transformation.", "") for t in self.transformations],
                        })

        for utility in self.utility_functions:
            dir_path = os.path.join(self.base_path, utility.__name__)
            os.mkdir(dir_path)

            #self._test_run(schedules, utility)
            self._run_in_parallel_with_timeout_and_callback(schedules, utility)

        return

    def _test_run(self, schedules: typing.List[typing.Tuple[int, ParallelSchedule]], utility):
        utility_path = os.path.join(self.base_path, utility.__name__)

        for ut, pt in itertools.product(self.utility_thresholds, self.privacy_thresholds):
            threshold_path = os.path.join(utility_path, f"u{ut}_p{pt}")
            os.mkdir(threshold_path)

            for t in self.transformations:
                path = os.path.join(threshold_path,
                                    str(t).replace('Transformation.', '')  # transformation
                                    )
                os.mkdir(path)

            param_list = list(
                itertools.product(schedules, self.transformations, [utility], [ut], [pt], [self.base_path]))

            for params in param_list:
                self._run_for_schedule_w_callback(params)

    def _run_in_parallel_with_timeout_and_callback(self,
                                                   schedules: typing.List[typing.Tuple[int, ParallelSchedule]],
                                                   utility):
        utility_path = os.path.join(self.base_path, utility.__name__)

        for ut, pt in itertools.product(self.utility_thresholds, self.privacy_thresholds):
            threshold_path = os.path.join(utility_path, f"u{ut}_p{pt}")
            os.mkdir(threshold_path)

            for t in self.transformations:
                path = os.path.join(threshold_path,
                                    str(t).replace('Transformation.', '')  # transformation
                                    )
                os.mkdir(path)

            param_list = list(
                itertools.product(schedules, self.transformations, [utility], [ut], [pt], [self.base_path]))
            with ProcessPool(max_workers=self.num_workers) as pool:
                # Distribute the parameter sets evenly across the cores
                future = pool.map(self._run_for_schedule_w_callback, param_list, timeout=self.time_limit)
                result = future.result()

                i = 0
                while True:
                    try:
                        result.next()
                    except StopIteration:
                        break
                    except concurrent.futures.TimeoutError:
                        sol = ParallelResult(outcome=Outcome.TIMEOUT, dismissed=False,
                                             original_schedule=param_list[i][0][1],
                                             time_found=self.time_limit)
                        file_name = f"{param_list[i][0][0]}_timeout"
                        instance_path = self.get_directory_path(param_list[i])
                        pickle_data(os.path.join(instance_path, file_name), sol)
                    i += 1

    def _run_for_schedule_w_callback(self, params):
        schedule = deepcopy(params[0][1])
        param_id = params[0][0]
        transformation = params[1]
        utility_function = params[2]
        utility_threshold = params[3]
        privacy_threshold = params[4]

        instance_path = self.get_directory_path(params)
        found_for_outcome = {Outcome.FOUND: False, Outcome.SIZE_ZERO: False}

        def callback(sol, idx):
            if sol.outcome == Outcome.SIZE_ZERO and not sol.dismissed and not found_for_outcome[Outcome.SIZE_ZERO]:
                file_name = f"{param_id}_{idx}_empty"
                found_for_outcome[Outcome.SIZE_ZERO] = True
                pickle_data(os.path.join(instance_path, file_name), sol)
            elif sol.outcome == Outcome.FOUND and not sol.dismissed and not found_for_outcome[Outcome.FOUND]:
                file_name = f"{param_id}_{idx}_non-empty"
                found_for_outcome[Outcome.FOUND] = True
                pickle_data(os.path.join(instance_path, file_name), sol)

        if self.schedule_type == "make":
            searcher = PupParallelMake(schedule, privacy_threshold, utility_threshold,
                                   utility_function=utility_function,
                                   prune_utility=False,
                                   schedule_parameters=schedule.params,
                                   callback=callback)
        else:
            searcher = PupParallel(schedule, privacy_threshold, utility_threshold,
                                   utility_function=utility_function,
                                   prune_utility=False,
                                   schedule_parameters=schedule.params,
                                   callback=callback)
        result = searcher.start_search(transformation=transformation, stopping_criterion=self.stopping_criterion)

        if not (found_for_outcome[Outcome.SIZE_ZERO] and found_for_outcome[Outcome.FOUND]):
            sol = ParallelResult(outcome=Outcome.NOT_FOUND, dismissed=False, original_schedule=schedule,
                                 time_found=result["time"])
            file_name = f"{param_id}_exhausted"
            pickle_data(os.path.join(instance_path, file_name), sol)

        return result

    def get_directory_path(self, params):
        # params: 0,0 - param_id, 1 - transformation, 2 - utility function, 3 - ul, 4 - pl, 5 - base path
        path = os.path.join(params[5],  # base path
                            str(params[2].__name__),  # utility function
                            f"u{params[3]}_p{params[4]}",
                            str(params[1]).replace('Transformation.', '')  # transformation
                            )
        return path


class ResultCollectorFactory:
    @staticmethod
    def create_from_setup_file(setup_file):
        with open(setup_file, "r") as s_file:
            setup = json.load(s_file)
        return ResultCollector(trials=setup["trials"], time_limit=setup["time_limit"], base_path=setup["base_path"],
                               date_prefix=setup["date"], transformations=setup["transformations"],
                               utility_functions=setup["utility_functions"], utility_thresholds=setup["utility_thresholds"],
                               privacy_thresholds=setup["privacy_thresholds"], schedules_path=setup["schedules"])

class ResultCollector:
    def __init__(self,
                 trials: int,
                 time_limit: int,
                 base_path: str,
                 date_prefix: str,
                 transformations: typing.List[str],
                 utility_functions: typing.List[str],
                 utility_thresholds: typing.List[float],
                 privacy_thresholds: typing.List[float],
                 schedules_path: str
                 ):
        self.trials = trials
        self.time_limit = time_limit
        self.base_path = base_path
        self.date_prefix = date_prefix

        self.privacy_thresholds = privacy_thresholds
        self.utility_thresholds = utility_thresholds
        self.utility_functions = utility_functions
        self.transformations = transformations
        self.schedules_path = schedules_path

    def load_schedules_and_collect_results(self):
        schedules = load_pickled_data(self.schedules_path)
        self.collect_results(schedules)

    def collect_results(self, schedules):
        header_outcomes = ["id", "utility_function", "utility_threshold", "privacy_threshold", "transformation",
                           "outcome"]
        # ADAPT ACCORDING TO EXPERIMENT - TODO: make more modular
        #header_measures = ["id", "utility_function", "utility_threshold", "privacy_threshold", "transformation",
        #                   "outcome_type", "solution_no", "original_twct", "original_avgw", "perturbed_twct",
        #                   "perturbed_avgw", "privacy_loss", "time_found"]

        header_measures = ["id", "utility_function", "utility_threshold", "privacy_threshold", "transformation",
                           "outcome_type", "solution_no", "original_cmax", "original_avgw", "perturbed_cmax",
                           "perturbed_avgw", "privacy_loss", "time_found"]

        for utility_function in self.utility_functions:
            uf_path = os.path.join(self.base_path, utility_function)
            outcomes_path = self._create_file_path(uf_path, self.date_prefix + "_outcomes", "csv")
            measures_path = self._create_file_path(uf_path, self.date_prefix + "_measures", "csv")

            with open(outcomes_path, "w", newline='') as outcomes, open(measures_path, "w", newline='') as measures:
                o_writer = csv.DictWriter(outcomes, fieldnames=header_outcomes)
                o_writer.writeheader()

                m_writer = csv.DictWriter(measures, fieldnames=header_measures)
                m_writer.writeheader()

                for ut, pt, t in itertools.product(self.utility_thresholds, self.privacy_thresholds,
                                                   self.transformations):
                    instance_path = os.path.join(uf_path, f"u{ut}_p{pt}", t)

                    # 0: empty, 1: non-empty, 2: timeout, 3: exhausted
                    # cache to use later to determine outcomes (empty, non-empty, both, timeout, exhausted)
                    o_cache = {pid: [0] * 4 for pid, _ in schedules}

                    if os.path.exists(instance_path) and os.path.isdir(instance_path):
                        directory = os.fsencode(instance_path)
                        dir_contents = os.listdir(directory)
                        if dir_contents:
                            for file in dir_contents:
                                file_name = os.fsdecode(file)
                                file_path = os.path.join(instance_path, file_name)
                                if os.path.isfile(file_path):
                                    self._handle_result_file(file_name, file_path, m_writer,
                                                              utility_function, ut, pt, t, o_cache)
                                else:
                                    print(f"{file_path} is not a file.")
                                    print(f"Corresponding dir_path: {instance_path}")
                                    continue

                        self._write_outcome(schedules, o_cache, o_writer, utility_function, ut, pt, t)
                    else:
                        print(f"No directory given for parameters {instance_path}")
                        continue

    def _handle_result_file(self, file_name, file_path, m_writer, utility_function, ut, pt, t, o_cache):
        try:
            with open(file_path, "rb") as f:
                fparts = file_name.split("_")
                pid = int(fparts[0])

                if fparts[1].startswith('s'):
                    sol_no = fparts[1].replace("s", "")
                    outcome_type = fparts[2]
                else:
                    sol_no = None
                    outcome_type = fparts[1]

                solution: ParallelResult = pickle.load(f)

                self._write_measures(m_writer,
                                     outcome_type,
                                     pid,
                                     utility_function,
                                     ut, pt, t, sol_no,
                                     solution)
                self._record_outcome(pid, outcome_type, o_cache)

        except EOFError:
            print(f"File {file_name} is empty / cannot be opened.")
            print(f"Corresponding dir_path: {file_path}")

    def _write_measures(self,
                        m_writer,
                        outcome_type,
                        pid,
                        utility_function,
                        ut, pt, t, sol_no,
                        solution):
        if outcome_type == "empty" or outcome_type == "non-empty":
            m_writer.writerow({
                "id": pid,
                "utility_function": utility_function,
                "utility_threshold": ut,
                "privacy_threshold": pt,
                "transformation": t,
                "outcome_type": outcome_type,
                "solution_no": sol_no,
                #"original_twct": calculate_twct(solution.original_schedule),
                #"perturbed_twct": calculate_twct(solution.privatized_schedule),
                "original_cmax": calculate_cmax(solution.original_schedule),
                "perturbed_cmax": calculate_cmax(solution.privatized_schedule),
                "original_avgw": calculate_avg_wait_time_with_release_dates(
                    solution.original_schedule),
                "perturbed_avgw": calculate_avg_wait_time_with_release_dates(
                    solution.privatized_schedule),
                "privacy_loss": solution.privacy_loss,
                "time_found": solution.time_found
            })
        else:
            m_writer.writerow({
                "id": pid,
                "utility_function": utility_function,
                "utility_threshold": ut,
                "privacy_threshold": pt,
                "transformation": t,
                "outcome_type": outcome_type,
                "solution_no": sol_no,
                #"original_twct": calculate_twct(solution.original_schedule),
                #"perturbed_twct": None,
                "original_cmax": calculate_cmax(solution.original_schedule),
                "perturbed_cmax": None,
                "original_avgw": calculate_avg_wait_time_with_release_dates(
                    solution.original_schedule),
                "perturbed_avgw": None,
                "privacy_loss": None,
                "time_found": solution.time_found
            })

    def _record_outcome(self, pid, outcome_type, o_cache):
        if outcome_type == "empty":
            o_cache[pid][0] = 1
        elif outcome_type == "non-empty":
            o_cache[pid][1] = 1
        elif outcome_type == "timeout":
            o_cache[pid][2] = 1
        elif outcome_type == "exhausted":
            o_cache[pid][3] = 1

    def _get_outcome_for_row(self, o_row):
        if o_row[0] and not o_row[1] and (o_row[2] or o_row[3]):
            ot = "empty"
        elif not o_row[0] and o_row[1] and (o_row[2] or o_row[3]):
            ot = "non-empty"
        elif not o_row[0] and not o_row[1] and o_row[2] and not o_row[3]:
            ot = "timeout"
        elif o_row[0] and o_row[1] and not o_row[2] and not o_row[3]:
            ot = "both"
        elif not o_row[0] and not o_row[1] and not o_row[2] and o_row[3]:
            ot = "exhausted"
        else:
            ot = "ERROR"

        return ot
    def _write_outcome(self,
                       schedules,
                       o_cache,
                       o_writer,
                       utility_function,
                       utility_threshold,
                       privacy_threshold,
                       transformation):

        for pid, schedule in schedules:
            o_row = o_cache[pid]
            ot = self._get_outcome_for_row(o_row)

            if ot == "ERROR":
                print(f"Error, wrong outcome for schedule with id {pid}")
            else:
                o_writer.writerow({
                    "id": pid,
                    "utility_function": utility_function,
                    "utility_threshold": utility_threshold,  # utility threshold
                    "privacy_threshold": privacy_threshold,  # privacy threshold
                    "transformation": transformation,
                    "outcome": ot
                })

    def _create_file_path(self, path, file_name, file_format):
        filepath = os.path.join(path, f"{file_name}.{file_format}")
        return filepath