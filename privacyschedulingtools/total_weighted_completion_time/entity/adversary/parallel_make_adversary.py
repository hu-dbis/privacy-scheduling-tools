import time

from more_itertools import pairwise
from ortools.sat.python import cp_model

from privacyschedulingtools.total_weighted_completion_time.entity.adversary.adversary import Adversary
from privacyschedulingtools.total_weighted_completion_time.entity.domain import IntegerDomain
from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import ParallelSchedule
from privacyschedulingtools.total_weighted_completion_time.entity.scheduling_parameters import \
    ParallelSchedulingParameters
from privacyschedulingtools.total_weighted_completion_time.pup.callback.result_size_and_w_original_callback import \
    ResultSizeAndWOriginalCallback


class ParallelMakeAdversary(Adversary):

    def __init__(self, scheduling_parameters: ParallelSchedulingParameters):
        self.processing_time_domain = scheduling_parameters.processing_time_domain
        self.release_domain_max = scheduling_parameters.weight_domain.get_max()
        self.release_domain = scheduling_parameters.weight_domain

    # this solves the model counting problem for the inverse scheduling attack
    # the inverse scheduling attack sets up a CSP for MAKE
    # the callback is called for each found solution to the inverse scheduling problem
    # the callback keeps track of the solution count and whether the original weight vector has been seen as a solution
    def execute_attack(self, schedule: ParallelSchedule, solution_counting_stop=None):
        model = cp_model.CpModel()
        decision_variables = {}  # release dates
        for job_index in schedule.jobs.keys():
            decision_variables[job_index] = model.NewIntVar(self.release_domain.get_min(),
                                                            self.release_domain.get_max(), 'r%i' % job_index)

        number_machines = len(schedule.allocation)
        #print(f"Number of machines: {number_machines}")

        # Sort scheduled jobs by start times
        scheduled_jobs = [j for machine in schedule.allocation for j in machine]
        jobs_by_start = {}
        start_of_job = {}

        for j in scheduled_jobs:
            if j.start_time in jobs_by_start:
                jobs_by_start[j.start_time].append(j.id)
            else:
                jobs_by_start[j.start_time] = [j.id]

            start_of_job[j.id] = j.start_time

        jobs_by_start = sorted(jobs_by_start.items())

        # 1) add release time constraints based on starting time of the very same job
        # r_i <= pi_i
        #print("1) add r_i <= pi_i")
        for job_index in schedule.jobs.keys():
            pi_i = start_of_job[job_index]
            #print(f"ADD CONSTRAINT: {decision_variables[job_index]} <= {pi_i}")
            model.Add((decision_variables[job_index] <= pi_i))

        # 2) add release time constraints based on closest predecessor job k with smaller processing time
        # r_i > pi_k
        #print("2) add r_i > pi_k")
        # 3) add release time constraints based on idle times
        # r_i = pi_i
        #print("3) add r_i = pi_i")
        #print(jobs_by_start)

        for job_index in schedule.jobs.keys():
            predecessor_jobs_by_start = [time_tuple for time_tuple in jobs_by_start if time_tuple[0] < start_of_job[job_index]]
            predecessor_jobs_by_start = sorted(predecessor_jobs_by_start, reverse=True)
            #print(f"job {job_index}, "
            #      f"start {start_of_job[job_index]}, "
            #      f"predecessors {predecessor_jobs_by_start}")
            p_i = schedule.jobs[job_index].processing_time
            found_predecessor_for_constraint = False
            for (t,js) in predecessor_jobs_by_start:
                # Check whether there are predecessors that have a smaller processing time than the current job
                if not found_predecessor_for_constraint:
                    if [j for j in js if schedule.jobs[j].processing_time < p_i]:
                        #print(f"job {job_index}, "
                        #      f"start {start_of_job[job_index]}, "
                        #      f"earlier jobs with smaller processing time {[j for j in js if schedule.jobs[j].processing_time < p_i]} "
                        #      f"at time {t}")
                        #print(f"ADD CONSTRAINT: {decision_variables[job_index]} > {t}")
                        model.Add((decision_variables[job_index] > t))
                        #break
                        found_predecessor_for_constraint = True

                # Check whether a machine was idling before the current job started execution
                # First determine running jobs, i.e., that started earlier and are still executed

           #     running_jobs = [j for j in js if start_of_job[j] + schedule.jobs[j].processing_time >= start_of_job[job_index]]
            running_jobs = [j for start, js in predecessor_jobs_by_start for j in js if start + schedule.jobs[j].processing_time >= start_of_job[job_index]]

           # print(f"job {job_index}, "
           #       f"start {start_of_job[job_index]}, "
           #       f"running jobs {running_jobs} "
           #       f"at time {start_of_job[job_index]}")
            if len(running_jobs) < number_machines:
                #print(f"ADD CONSTRAINT: {decision_variables[job_index]} == {start_of_job[job_index]}")
                model.Add((decision_variables[job_index] == start_of_job[job_index]))

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
