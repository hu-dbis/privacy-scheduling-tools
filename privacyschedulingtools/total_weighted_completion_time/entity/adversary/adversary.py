from typing import Union

import numpy as np
from more_itertools import pairwise
from ortools.sat.python import cp_model

from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import ParallelSchedule
from privacyschedulingtools.total_weighted_completion_time.entity.schedule import Schedule
from privacyschedulingtools.total_weighted_completion_time.entity.scheduling_parameters import SchedulingParameters, \
    ParallelSchedulingParameters
from privacyschedulingtools.total_weighted_completion_time.pup.callback.result_size_and_w_original_callback import \
    ResultSizeAndWOriginalCallback
from privacyschedulingtools.total_weighted_completion_time.pup.util.auto_string import auto_str


@auto_str
class Adversary():

    def execute_attack(self, schedule: Schedule):
        pass


class WSPTAdversary(Adversary):

    def __init__(self, scheduling_parameters: Union[ParallelSchedulingParameters, SchedulingParameters]):
        self.processing_time_domain = scheduling_parameters.processing_time_domain
        self.weight_domain = scheduling_parameters.weight_domain
        self.domain_max = self.processing_time_domain.get_max() * self.weight_domain.get_max() + 1

    # this solves the model counting problem for the inverse scheduling attack
    # the inverse scheduling attack leverages the weighted shortest processing time first (WSPT) rule by setting up a CSP
    # the callback is called for each found solution to the inverse scheduling problem
    # the callback keeps track of the solution count and whether the original weight vector has been seen as a solution
    def execute_attack(self, schedule: Union[Schedule, ParallelSchedule], solution_counting_stop=None):

        model = cp_model.CpModel()
        decision_variables = {}  # weights
        for job_index in schedule.schedule_order:
            decision_variables[job_index] = model.NewIntVar(self.weight_domain.get_min(),
                                                            self.weight_domain.get_max(), 'w%i' % job_index)

        # WSPT first rule added for each neighboring job pair as constraints
        for job_left_index, job_right_index in pairwise(schedule.schedule_order):
            left = model.NewIntVar(1, self.domain_max, '')
            right = model.NewIntVar(1, self.domain_max, '')
            model.AddMultiplicationEquality(left, [decision_variables[job_left_index],
                                                   schedule.jobs[job_right_index].processing_time])
            model.AddMultiplicationEquality(right, ([decision_variables[job_right_index],
                                                     schedule.jobs[job_left_index].processing_time]))
            model.Add((left >= right))

        callback = ResultSizeAndWOriginalCallback(schedule, decision_variables, solution_counting_stop)

        solver = cp_model.CpSolver()
        # solver.parameters.max_time_in_seconds = 10.0
        status = solver.SearchForAllSolutions(model, callback)
        return callback.get_results()


class RandomGuessingAdversary(Adversary):

    def __init__(self, scheduling_parameters: SchedulingParameters, min_solution_count=1, max_solution_count=10):
        self.processing_time_domain = scheduling_parameters.processing_time_domain
        self.weight_domain = scheduling_parameters.weight_domain
        self.min_solution_count = min_solution_count
        self.max_solution_count = max_solution_count

    def execute_attack(self, schedule: Schedule):
        solution_count = np.random.randint(self.min_solution_count, self.max_solution_count + 1)
        solutions = [self.generate_random_weight_vector(len(schedule.jobs)) for i in range(solution_count)]
        original_weight_vector_seen = any([schedule.weight_vector_equals(w) for w in solutions])
        result = {"solution_count": solution_count,
                  "original_weight_vector_seen": original_weight_vector_seen,
                  "solutions": solutions}
        return result

    def generate_random_weight_vector(self, job_count):
        weights = np.random.randint(self.weight_domain.get_min(),
                                    self.weight_domain.get_max() + 1,
                                    job_count)
        random_weight_vector = {i: weights[i] for i in range(job_count)}
        return random_weight_vector
