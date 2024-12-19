from ortools.sat.python import cp_model


#  keeps track of the solution count and stops the solver once enough solutions have been found
class SolutionSizeCallback(cp_model.CpSolverSolutionCallback):

    def __init__(self, desired_solution_count=None):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.solution_count = 0
        self.desired_solution_count = desired_solution_count

    def on_solution_callback(self):
        self.solution_count += 1
        if self.desired_solution_count is not None and self.solution_count == self.desired_solution_count:
            self.StopSearch()

    def get_solution_count(self):
        return self.solution_count
