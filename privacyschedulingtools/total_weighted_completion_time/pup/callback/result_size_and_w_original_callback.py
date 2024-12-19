from typing import Union

from ortools.sat.python import cp_model

from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import ParallelSchedule
from privacyschedulingtools.total_weighted_completion_time.entity.schedule import Schedule


#  keeps track of the solution count and whether the original weight vector was found to be in the solution set
#  stops the solver once all (enough) solutions have been found
class ResultSizeAndWOriginalCallback(cp_model.CpSolverSolutionCallback):

    def __init__(self, schedule: Union[Schedule, ParallelSchedule], decision_variables, desired_solution_count = None):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.decision_variables = decision_variables
        self.solution_count = 0
        self.original_weight_vector_seen = False
        self.original_weights = {}
        self.solutions = []
        for job in schedule.jobs.values():
            self.original_weights[job.id] = job.weight
        self.desired_solution_count = desired_solution_count

    def on_solution_callback(self):
        self.solution_count += 1
        current_solution_weights = dict()
        # key should be job id
        for key, value in self.decision_variables.items():
            current_solution_weights[key] = self.Value(value)
        self.solutions.append(current_solution_weights)
        if self.original_weights == current_solution_weights:
            self.original_weight_vector_seen = True
        if self.desired_solution_count is not None and self.solution_count == self.desired_solution_count:
            self.StopSearch()

    def get_solution_count(self):
        return self.solution_count

    def was_original_weight_vector_seen(self):
        return self.original_weight_vector_seen

    def get_results(self):
        result = {"solution_count": self.solution_count,
                  "original_weight_vector_seen": self.original_weight_vector_seen,
                  "solutions": self.solutions}
        return result

