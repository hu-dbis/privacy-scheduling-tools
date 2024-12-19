from collections import Counter

from ortools.sat.python import cp_model


#  keeps track of the solution count and the distribution for each weight decision variable
class DistributionCallback(cp_model.CpSolverSolutionCallback):

    def __init__(self, decision_variables, desired_solution_count=None):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.decision_variables = decision_variables
        self.distributions = {}
        self.solution_count = 0
        self.desired_solution_count = desired_solution_count
        for key, value in decision_variables.items():
            self.distributions[value] = Counter()

    def on_solution_callback(self):
        for key, value in self.decision_variables.items():
            self.distributions[value][self.Value(value)] += 1
        self.solution_count += 1
        if self.desired_solution_count is not None and self.solution_count == self.desired_solution_count:
            self.StopSearch()

    def get_distributions(self):
        return self.distributions
