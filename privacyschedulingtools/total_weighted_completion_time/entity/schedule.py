import typing

from privacyschedulingtools.total_weighted_completion_time.entity.scheduling_parameters import SchedulingParameters
from privacyschedulingtools.total_weighted_completion_time.pup.util.auto_string import auto_str

from numpy import ndarray

from privacyschedulingtools.total_weighted_completion_time.entity.job import Job
import numpy as np


class Schedule:

    def __init__(self,
                 jobs: typing.Dict[int, Job],
                 schedule_order: ndarray,
                 params: SchedulingParameters):

        self.jobs = jobs
        self.schedule_order = schedule_order
        self.params = params

    def __str__(self):
        jobs = [self.jobs[id].__str__() for id in self.schedule_order]
        return ", ".join(jobs)

    def __eq__(self, other):
        if not isinstance(other, Schedule):
            return False
        if not self.schedule_order == other.schedule_order:
            return False
        for j in self.schedule_order:
            if not self.jobs.get(j) == other.jobs.get(j):
                return False

        return True

    def ordered_job_tuple(self):
        return tuple([self.jobs[i].tuple() for i in self.schedule_order])

    def __hash__(self):
        return hash(self.ordered_job_tuple())

    def weight_vector_equals(self, weight_vector):
        for id, job in self.jobs.items():
            if job.weight != weight_vector[id]:
                return False
        return True


@auto_str
class ScheduleFactory:

    def __init__(self, scheduling_parameters: SchedulingParameters):
        self.params = scheduling_parameters

    def from_job_tuple_list(self, job_tuple_list):
        """
        :param job_tuple_list: a list of job-tuples (id, processing time, weight) expressing the order of jobs
        :return: a schedule object
        """
        jobs: typing.Dict[int, Job] = {}
        schedule_order = []
        for job_tuple in job_tuple_list:
            job_id = job_tuple[0]
            jobs[job_id] = Job(id=job_id, processing_time=job_tuple[1], weight=job_tuple[2])
            schedule_order.append(job_id)
        so: ndarray = np.array(schedule_order, dtype=np.int32)

        return Schedule(jobs, so, self.params)

    def generate_random_schedule(self):
        jobs: typing.Dict[int, Job] = {}
        schedule_order: ndarray = np.array([], dtype=np.int32)
        for generated_id in range(self.params.job_count):
            processing_time = self.params.processing_time_domain.get_random()
            weight = self.params.weight_domain.get_random()
            jobs[generated_id] = Job(generated_id, processing_time, weight)
            np.append(schedule_order, generated_id)

        return Schedule(jobs, schedule_order, self.params)

    # generate a random schedule and reorder according to the WSPT-rule
    def generate_random_optimized_schedule(self):

        generated_schedule = self.generate_random_schedule()

        jobs = list(generated_schedule.jobs.values())
        # optimize the twct (usage of the weighted shortest processing time first rule)
        jobs.sort(key=lambda j: j.weight / j.processing_time, reverse=True)

        # reassign the job id's for a prettier starting position [0,1,...,n-1]
        for processing_position, job in enumerate(jobs):
            job.id = processing_position
            generated_schedule.jobs[processing_position] = job

        generated_schedule.schedule_order: ndarray = np.array([i.id for i in jobs], dtype=np.int32)
        return generated_schedule
