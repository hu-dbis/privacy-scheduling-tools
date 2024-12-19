from ortools.sat.python import cp_model

#  collects all solutions i.e. weight vectors that solve the ISP (up to a maximum count if given)
class SolutionCollectorCallback(cp_model.CpSolverSolutionCallback):

    def __init__(self, decision_variables, desired_solution_count= None):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.decision_variables = decision_variables
        self.solutions = []
        self.desired_solution_count = desired_solution_count

    def on_solution_callback(self):
        current_solution_weights = [self.Value(value) for value in self.decision_variables]
        self.solutions.append(current_solution_weights)
        if self.desired_solution_count is not None and len(self.solutions) == self.desired_solution_count:
            self.StopSearch()

    def get_solution_count(self):
        return len(self.solutions)

    def get_results(self):
        return self.solutions
