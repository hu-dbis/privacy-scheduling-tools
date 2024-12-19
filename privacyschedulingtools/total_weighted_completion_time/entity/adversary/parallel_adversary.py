import time

from more_itertools import pairwise
from ortools.sat.python import cp_model

from privacyschedulingtools.total_weighted_completion_time.entity.adversary.adversary import Adversary
from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import ParallelSchedule
from privacyschedulingtools.total_weighted_completion_time.entity.scheduling_parameters import \
    ParallelSchedulingParameters
from privacyschedulingtools.total_weighted_completion_time.pup.callback.result_size_and_w_original_callback import \
    ResultSizeAndWOriginalCallback


class ParallelWSPTAdversary(Adversary):
    # TODO: reconsider handling of solver and constraints during attack

    def __init__(self, scheduling_parameters: ParallelSchedulingParameters):
        self.processing_time_domain = scheduling_parameters.processing_time_domain
        self.weight_domain = scheduling_parameters.weight_domain
        self.domain_max = self.processing_time_domain.get_max() * self.weight_domain.get_max() + 1

    # this solves the model counting problem for the inverse scheduling attack
    # the inverse scheduling attack leverages the weighted shortest processing time first (WSPT) rule by setting up a CSP
    # the callback is called for each found solution to the inverse scheduling problem
    # the callback keeps track of the solution count and whether the original weight vector has been seen as a solution
    def execute_attack(self, schedule: ParallelSchedule, solution_counting_stop=None):

        model = cp_model.CpModel()
        decision_variables = {}  # weights
        for job_index in schedule.jobs.keys():
            decision_variables[job_index] = model.NewIntVar(self.weight_domain.get_min(),
                                                            self.weight_domain.get_max(), 'w%i' % job_index)

        # WSPT first rule added for each neighboring job pair as constraints
        scheduled_jobs = [j for machine in schedule.allocation for j in machine]
        jobs_by_start = {}

        for j in scheduled_jobs:
            if j.start_time in jobs_by_start:
                jobs_by_start[j.start_time].append(j.id)
            else:
                jobs_by_start[j.start_time] = [j.id]

        jobs_by_start = sorted(jobs_by_start.items())

        for jobs_left, jobs_right in pairwise(jobs_by_start):
            for job_left_index in jobs_left[1]:
                for job_right_index in jobs_right[1]:
                    left = model.NewIntVar(1, self.domain_max, '')
                    right = model.NewIntVar(1, self.domain_max, '')
                    model.AddMultiplicationEquality(left, [decision_variables[job_left_index],
                                                           schedule.jobs[job_right_index].processing_time])
                    model.AddMultiplicationEquality(right, ([decision_variables[job_right_index],
                                                             schedule.jobs[job_left_index].processing_time]))
                    model.Add((left >= right))

        callback = ResultSizeAndWOriginalCallback(schedule, decision_variables, solution_counting_stop)

        solver = cp_model.CpSolver()
        limit = 60.0
        solver.parameters.max_time_in_seconds = limit

        st = time.time()
        status = solver.SearchForAllSolutions(model, callback)
        et = time.time() - st

        results = callback.get_results()
        results['status'] = cp_model.UNKNOWN if et > limit else status
        return results


class ParallelLocalWSPTAdversary(Adversary):
    # TODO: reconsider local WSPT attack (necessary (?) optimality condition for P- - w_jC_j)

    def __init__(self, scheduling_parameters: ParallelSchedulingParameters):
        self.processing_time_domain = scheduling_parameters.processing_time_domain
        self.weight_domain = scheduling_parameters.weight_domain
        self.domain_max = self.processing_time_domain.get_max() * self.weight_domain.get_max() + 1

    # this solves the model counting problem for the inverse scheduling attack
    # the inverse scheduling attack leverages the weighted shortest processing time first (WSPT) rule by setting up a CSP
    # the callback is called for each found solution to the inverse scheduling problem
    # the callback keeps track of the solution count and whether the original weight vector has been seen as a solution
    def execute_attack(self, schedule: ParallelSchedule, solution_counting_stop=None):

        model = cp_model.CpModel()
        decision_variables = {}  # weights
        for job_index in schedule.jobs.keys():
            decision_variables[job_index] = model.NewIntVar(self.weight_domain.get_min(),
                                                            self.weight_domain.get_max(), 'w%i' % job_index)

        # WSPT first rule added for each neighboring job pair as constraints
        # Checks for local WSPT constraints (only ensure within machine but not across)
        for machine in schedule.allocation:
            for job_left, job_right in pairwise(machine):
                left = model.NewIntVar(1, self.domain_max, '')
                right = model.NewIntVar(1, self.domain_max, '')
                model.AddMultiplicationEquality(left, [decision_variables[job_left.id],
                                                       schedule.jobs[job_right.id].processing_time])
                model.AddMultiplicationEquality(right, ([decision_variables[job_right.id],
                                                         schedule.jobs[job_left.id].processing_time]))
                model.Add((left >= right))

        callback = ResultSizeAndWOriginalCallback(schedule, decision_variables, solution_counting_stop)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0
        status = solver.SearchForAllSolutions(model, callback)
        results = callback.get_results()
        results['status'] = status
        return results
