from typing import Callable

from privacyschedulingtools.total_weighted_completion_time.entity.job import Job, DatedJob
from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import ParallelSchedule
from privacyschedulingtools.total_weighted_completion_time.entity.schedule import Schedule


def s_calculate(schedule: Schedule, function: Callable) -> int:
    utility = 0
    current_time = 0
    for job_id in schedule.schedule_order:
        job = schedule.jobs[job_id]
        current_time += job.processing_time
        utility += function(job, current_time)
    return utility


def p_calculate(schedule: ParallelSchedule, function) -> int:
    utility = 0
    for machine in schedule.allocation:
        for j in machine:
            job = schedule.jobs[j.id]
            c_t = j.start_time + job.processing_time
            utility += function(job, c_t)
    return utility


def w_completion_time(job: Job, current_time: int) -> int:
    return job.weight * current_time


def completion_time(job: Job, current_time: int) -> int:
    return current_time


def flow_time(job: DatedJob, current_time: int) -> int:
    return current_time - job.release_date


def w_flow_time(job: DatedJob, current_time: int) -> int:
    return job.weight * flow_time(job, current_time)


def lateness(job: DatedJob, current_time: int) -> int:
    return current_time - job.due_date


def w_lateness(job: DatedJob, current_time: int) -> int:
    return job.weight * lateness(job, current_time)


def tardiness(job: DatedJob, current_time: int) -> int:
    return max(lateness(job, current_time), 0)


def w_tardiness(job: DatedJob, current_time: int) -> int:
    return job.weight * tardiness(job, current_time)


