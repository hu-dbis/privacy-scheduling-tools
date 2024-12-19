from math import sqrt
from statistics import mean
from typing import Callable, Union

import scipy

from privacyschedulingtools.total_weighted_completion_time.entity.job import DatedJob, Job
from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import ParallelSchedule, \
    ScheduledJob
from privacyschedulingtools.total_weighted_completion_time.pup.utility_functions.objective_functions import \
    w_completion_time, completion_time, p_calculate


def calculate_twct(schedule: ParallelSchedule):
    return p_calculate(schedule, w_completion_time)


def calculate_tct(schedule: ParallelSchedule):
    return p_calculate(schedule, completion_time)


def calculate_cmax(schedule: ParallelSchedule) -> float:
    completion_times = [machine[-1].start_time + schedule.jobs[machine[-1].id].processing_time for machine in
                        schedule.allocation if len(machine) > 0]
    cmax = max(completion_times) if completion_times else 0
    return cmax


def calculate_emd(values1,
                  values2) -> float:
    return scipy.stats.wasserstein_distance(values1, values2)


def get_values(schedule, get_property: Callable):
    return [get_property(schedule.jobs[j.id], j) for machine in schedule.allocation for j in machine]


def get_waiting_time(job: Union[DatedJob, Job], scheduled_job: ScheduledJob):
    release_date = job.release_date if job is DatedJob else 0
    return scheduled_job.start_time - release_date


def get_completion_time(job: Union[DatedJob, Job], scheduled_job: ScheduledJob):
    return scheduled_job.start_time + job.processing_time


def get_processing_time(job: Union[DatedJob, Job], scheduled_job: ScheduledJob):
    return job.processing_time


def get_tasks_per_machine(schedule: ParallelSchedule):
    return [len(machine) for machine in schedule.allocation]


def get_machine_completion_times(schedule: ParallelSchedule):
    return [machine[-1].start_time + schedule.jobs[machine[-1].id].processing_time if len(machine) > 0 else 0 for
            machine in schedule.allocation]


def compute_utility_difference(original_utility, anonymized_utility):
    return abs(original_utility - anonymized_utility)


def compute_element_wise_distance(original_schedule, anonymized_schedule, property_function: Callable,
                                  p: str = "Euclidian", machine_lvl: bool = False):
    # Match jobs by id and compute element-wise Manhattan / Euclidian / Chebyshev distance between properties
    # TODO: normalize by #jobs * max property - min property ?
    assert p in ["Euclidian", "Manhattan", "Chebyshev"]
    if machine_lvl:
        original_values = [(i, property_function(machine, original_schedule)) for i, machine in enumerate(original_schedule.allocation)]
        anonymized_values = {i: property_function(machine, anonymized_schedule) for i, machine in enumerate(anonymized_schedule.allocation)}
    else:
        original_values = [(j.id, property_function(original_schedule.jobs[j.id], j)) for machine in
                           original_schedule.allocation for j in machine]
        anonymized_values = {j.id: property_function(anonymized_schedule.jobs[j.id], j) for machine in
                             anonymized_schedule.allocation for j in machine}

    if p == "Euclidian":
        return sqrt(sum([abs(original_value - anonymized_values[idx])**2 for idx, original_value in original_values]))
    elif p == "Chebyshev":
        return max([abs(original_value - anonymized_values[idx]) for idx, original_value in original_values])
    elif p == "Manhattan":
        return sum([abs(original_value - anonymized_values[idx]) for idx, original_value in original_values])
    else:
        raise ValueError(f"Distance {p} not supported.")


def calculate_avg_wait_time(schedule: ParallelSchedule) -> float:
    waiting_times = get_values(schedule, get_waiting_time)
    return mean(waiting_times)

def calculate_avg_wait_time_with_release_dates(schedule: ParallelSchedule) -> float:
    waiting_times = [j.start_time - schedule.jobs[j.id].weight for machine in schedule.allocation for j in machine]
    return mean(waiting_times)
