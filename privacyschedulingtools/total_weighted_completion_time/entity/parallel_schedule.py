import random
import numpy as np
from copy import deepcopy
from itertools import combinations

from typing import List, Dict, Union
from privacyschedulingtools.total_weighted_completion_time.entity.scheduling_parameters import \
    ParallelSchedulingParameters
from privacyschedulingtools.total_weighted_completion_time.pup.util.auto_string import auto_str

from privacyschedulingtools.total_weighted_completion_time.entity.job import Job


class ScheduledJob:
    def __init__(self, idx: int, start_time: int):
        self.id: int = idx
        self.start_time: int = start_time

    def __eq__(self, other):
        if not isinstance(other, ScheduledJob):
            return False
        if not self.id == other.id:
            return False
        if not self.start_time == other.start_time:
            return False

        return True

    def __str__(self):
        return f"({self.id}: {self.start_time})"


class ParallelSchedule:

    def __init__(self,
                 jobs: Dict[int, Job],
                 allocation: List,
                 params: ParallelSchedulingParameters):

        self.jobs = jobs
        self.params = params
        self.allocation = allocation
        self.schedule_order = self._compute_schedule_order_by_machine()

    def update_schedule_order(self):
        self.schedule_order = self._compute_schedule_order_by_machine()

    def _compute_schedule_order(self):
        jobs_with_start_times = [job for machine in self.allocation for job in machine]
        jobs_with_start_times.sort(
            key=lambda x: (x.start_time, - self.jobs[x.id].weight / self.jobs[x.id].processing_time))
        return [j.id for j in jobs_with_start_times]

    def _compute_schedule_order_by_machine(self):
        jobs_with_start_times = [(i, job) for i, machine in enumerate(self.allocation) for job in machine]
        jobs_with_start_times.sort(
            key=lambda x: (x[1].start_time, x[0]))
        return [j.id for i, j in jobs_with_start_times]

    def _compute_schedule_order_linear(self):
        return [job.id for machine in self.allocation for job in machine]

    def __str__(self):
        output = ""
        for i, machine in enumerate(self.allocation):
            machine_str = [x.__str__() for x in machine]
            job_str = ", ".join(machine_str)
            machine_str = f"M{i}: {job_str}\n"
            output += machine_str

        return output

    def schedule_to_tuple(self):
        return tuple([tuple([(j.start_time, self.jobs[j.id].tuple()) for j in machine]) for machine in self.allocation])

    def __hash__(self):
        return hash(self.schedule_to_tuple())

    def __eq__(self, other):
        if not isinstance(other, ParallelSchedule):
            return False
        # check number of machines
        if not len(self.allocation) == len(other.allocation):
            return False
        if not self.schedule_order == other.schedule_order:
            return False
        for j in self.schedule_order:
            if not self.jobs.get(j) == other.jobs.get(j):
                return False
        for i, machine in enumerate(self.allocation):
            # check number of jobs on each machine
            if not len(machine) == len(other.allocation[i]):
                return False
            # check if same job scheduled at same time
            for j, scheduled_job in enumerate(machine):
                if not scheduled_job == other.allocation[i][j]:
                    return False

        return True

    def compare_published_version(self, other):
        if not isinstance(other, ParallelSchedule):
            return False
        # check number of machines
        if not len(self.allocation) == len(other.allocation):
            return False
        # if not self.schedule_order == other.schedule_order:
        #    return False
        # for idx in self.schedule_order:
        for idx, job in self.jobs.items():
            if idx not in other.jobs.keys():
                return False
            else:
                if job.processing_time != other.jobs[idx].processing_time:
                    return False
            # if not (self.jobs[idx].processing_time == other.jobs[idx].processing_time):
            #    return False
        for i, machine in enumerate(self.allocation):
            # check number of jobs on each machine
            if not len(machine) == len(other.allocation[i]):
                return False
            # check if same job scheduled at same time
            for j, scheduled_job in enumerate(machine):
                if not scheduled_job == other.allocation[i][j]:
                    return False

        return True

    @staticmethod
    def is_valid_twct_schedule(schedule) -> bool:
        if not isinstance(schedule, ParallelSchedule):
            return False

        machine_lengths = [len(m) for m in schedule.allocation]
        last_jobs = [m[-1] for m in schedule.allocation if m]

        if min(machine_lengths) == 0 and max(machine_lengths) > 1:
            return False

        for j1, j2 in combinations(last_jobs, 2):
            if j1.start_time > j2.start_time + schedule.jobs[j2.id].processing_time or \
                    j2.start_time > j1.start_time + schedule.jobs[j1.id].processing_time:
                return False

        return True

    @staticmethod
    def is_valid_make_schedule(schedule, original_jobs: dict[int, Job]) -> bool:
        if not isinstance(schedule, ParallelSchedule):
            return False

        machine_lengths = [len(m) for m in schedule.allocation]

        # EST check
        for machine in schedule.allocation:
            for j, scheduled_job in enumerate(machine):
                if scheduled_job.start_time < original_jobs[scheduled_job.id].weight:
                    return False
                else:
                    prev = machine[j-1].start_time + schedule.jobs[machine[j-1].id].processing_time if j > 0 else 0
                    if scheduled_job.start_time != prev and scheduled_job.start_time != original_jobs[scheduled_job.id].weight:
                        return False

        return True

    def to_json_representation(self):
        representation = []

        for i, machine in enumerate(self.allocation):
            for scheduled_job in machine:
                scheduled_job_representation = dict(
                    Task=f"J{scheduled_job.id}",
                    Start=scheduled_job.start_time,
                    Finish=scheduled_job.start_time + self.jobs[scheduled_job.id].processing_time,
                    Resource=f"M{i}",
                    Weight=self.jobs[scheduled_job.id].weight
                )

                representation.append(scheduled_job_representation)

        return representation


@auto_str
class ParallelScheduleFactory:

    def __init__(self, scheduling_parameters: ParallelSchedulingParameters):
        self.params = scheduling_parameters

    def from_job_tuple_list(self, job_tuple_lists, schedule_type="wspt"):
        """
        :param job_tuple_lists: a list of lists of job-tuples (id, processing time, weight) expressing the allocation
        to jobs to machines
        :return: a schedule object
        """
        jobs: Dict[int, Job] = {}
        allocation = []

        for job_tuple_list in job_tuple_lists:
            machine = []
            possible_start_time = 0
            for job_tuple in job_tuple_list:
                job_id = job_tuple[0]
                jobs[job_id] = Job(id=job_id, processing_time=job_tuple[1], weight=job_tuple[2])
                start_time = max(possible_start_time, job_tuple[2]) if schedule_type == "make" else possible_start_time
                machine.append(ScheduledJob(job_id, start_time))
                possible_start_time = start_time + job_tuple[1] # processing_time
            allocation.append(machine)

        return ParallelSchedule(jobs, allocation, self.params)

    def generate_random_schedule(self):
        jobs: Dict[int, Job] = {}
        allocation: List[List[ScheduledJob]] = [[] for _ in range(self.params.machine_count)]
        start_times = [0] * self.params.machine_count

        for generated_id in range(self.params.job_count):
            processing_time = self.params.processing_time_domain.get_random()
            weight = self.params.weight_domain.get_random()
            jobs[generated_id] = Job(generated_id, processing_time, weight)

            m = random.randrange(0, self.params.machine_count)
            allocation[m].append(ScheduledJob(generated_id, start_times[m]))
            start_times[m] += processing_time

        return ParallelSchedule(jobs, allocation, self.params)

    # generate a random schedule and reorder according to the chosen dispatching rule
    def generate_random_schedule_with_dispatching_rule(self, schedule_type="wspt"):
        generated_schedule = self.generate_random_schedule()

        jobs = list(generated_schedule.jobs.values())

        if schedule_type == "make":
            generated_schedule.allocation, generated_schedule.schedule_order = self._schedule_jobs_by_make(jobs,
                                                                                                       self.params.machine_count)
        else:
            generated_schedule.allocation, generated_schedule.schedule_order = self._schedule_jobs_by_wspt(jobs,
                                                                                                           self.params.machine_count)

        # reassign the job id's for a prettier starting position [0,1,...,n-1]
        # for processing_position, job in enumerate(jobs):
        #    job.id = processing_position
        #    generated_schedule.jobs[processing_position] = job

        # generated_schedule.update_schedule_order()

        return generated_schedule

    def generate_schedule_with_weights(self, schedule: ParallelSchedule, weights: Dict[int, Union[int, float]],
                                       schedule_type="wspt"):
        new_schedule = deepcopy(schedule)
        jobs = list(new_schedule.jobs.values())
        for job in jobs:
            job.weight = weights[job.id]

        if schedule_type == "make":
            new_schedule.allocation, new_schedule.schedule_order = self._schedule_jobs_by_make(jobs,
                                                                                               self.params.machine_count)
        else:
            new_schedule.allocation, new_schedule.schedule_order = self._schedule_jobs_by_wspt(jobs,
                                                                                           self.params.machine_count)

        return new_schedule

    @staticmethod
    def generate_schedule_from_jobs(jobs, machine_count, parameters, schedule_type="wspt"):
        job_dict = {j.id: j for j in jobs}
        if schedule_type == "make":
            allocation, schedule_order = ParallelScheduleFactory._schedule_jobs_by_make(jobs, machine_count)
        else:
            allocation, schedule_order = ParallelScheduleFactory._schedule_jobs_by_wspt(jobs, machine_count)

        schedule = ParallelSchedule(job_dict, allocation, parameters)
        schedule.schedule_order = schedule_order
        return schedule

    @staticmethod
    def _schedule_jobs_by_wspt(jobs, machine_count):
        machines = [[] for _ in range(machine_count)]

        current_makespan = [0] * machine_count
        sorted_jobs = sorted(jobs, key=(lambda job: job.weight / job.processing_time), reverse=True)
        schedule_order = []

        for i in range(len(jobs)):
            # gets id of next free machine -> the one with minimum cmax so far
            free_machine = current_makespan.index(min(current_makespan))
            next_job = sorted_jobs[i]

            scheduled_job = ScheduledJob(next_job.id, current_makespan[free_machine])

            machines[free_machine].append(scheduled_job)
            current_makespan[free_machine] += next_job.processing_time
            schedule_order.append(next_job.id)

        return machines, schedule_order

    @staticmethod
    def _schedule_jobs_by_make(jobs, machine_count):
        machines = [[] for _ in range(machine_count)]

        current_makespan = [0] * machine_count
        sorted_jobs = sorted(jobs, key=(lambda job: job.processing_time), reverse=True)
        schedule_order = []

        while sorted_jobs:
            # gets id of next free machine -> the one with minimum cmax so far
            min_cmax = min(current_makespan)
            free_machine = current_makespan.index(min_cmax)
            next_job = sorted_jobs[0]
            next_job_idx = 0

            # get next job : either the one with next release date or one that has been released
            # and has max duration of released ones
            for i, job in enumerate(sorted_jobs):
                if job.weight <= min_cmax:
                    # job has already been published
                    # takes the one with max duration because jobs pre-sorted
                    next_job = job
                    next_job_idx = i
                    break
                elif job.weight < next_job.weight:
                    # job not yet published but which may be released next (weight interpreted as release date)
                    # because of < while going through sorted list: also take the one with highest duration when jobs with same lowest release date
                    next_job = job
                    next_job_idx = i

            sorted_jobs.pop(next_job_idx)
            scheduled_job = ScheduledJob(next_job.id, max(next_job.weight, min_cmax))

            machines[free_machine].append(scheduled_job)
            current_makespan[free_machine] = scheduled_job.start_time + next_job.processing_time
            schedule_order.append(next_job.id)

        return machines, schedule_order
