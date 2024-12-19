import copy
import math

from privacyschedulingtools.total_weighted_completion_time.entity.adversary.adversary import WSPTAdversary
from privacyschedulingtools.total_weighted_completion_time.entity.result import Outcome
from privacyschedulingtools.total_weighted_completion_time.entity.schedule import Schedule
from privacyschedulingtools.total_weighted_completion_time.pup.util.duplicate_detection import DuplicateDetection
from privacyschedulingtools.total_weighted_completion_time.pup.utility_functions import single_machine_utilities


# to setup the PUP and solve it
# solver based on bfs, swapping neighboring jobs
# privacy and utility thresholds are specified as decimals (0.01 -> 1%)
# utility loss is measured by the percentage change in the twct
class PupTwctMultipleOutcomes:

    def __init__(self, schedule: Schedule, privacy_threshold: float, utility_threshold: float):
        self.original_schedule = schedule
        self.privacy_threshold = privacy_threshold
        self.utility_threshold = utility_threshold

        self.visited_states = DuplicateDetection()
        self.original_utility = single_machine_utilities.calculate_twct(schedule)

        self.depth = 0
        self.result = {}

        self.adversary = WSPTAdversary(schedule.params)

    def satisfies_utility_constraint(self, schedule: Schedule) -> bool:
        utility = single_machine_utilities.calculate_twct(schedule)

        percentage_change = (abs(self.original_utility - utility)) / self.original_utility
        return percentage_change <= self.utility_threshold

    def update_result(self, outcome: Outcome, schedule):
        self.result[outcome] = {"depth": self.depth, "schedule": copy.deepcopy(schedule)}

    def check_pup(self, schedule: Schedule):
        # utility_functions check is here not necessary, it was already checked
        if self.privacy_threshold == 0:
            necessary_solution_count = 0
        else:
            necessary_solution_count = int(math.ceil(1 / self.privacy_threshold))

        if Outcome.SIZE_ZERO not in self.result:  # can't limit the model counting here
            result_privacy = self.adversary.execute_attack(schedule)
        else:
            result_privacy = self.adversary.execute_attack(schedule, necessary_solution_count)

        solution_count = result_privacy['solution_count']
        original_vector_seen = result_privacy['original_weight_vector_seen']

        if solution_count == 0 and Outcome.SIZE_ZERO not in self.result:
            self.update_result(Outcome.SIZE_ZERO, schedule)
        if necessary_solution_count <= solution_count and Outcome.SOLUTION_COUNT_LARGE_ENOUGH not in self.result:
            self.update_result(Outcome.SOLUTION_COUNT_LARGE_ENOUGH, schedule)
        if not original_vector_seen and solution_count != 0 and Outcome.TRUE_W_NOT_IN_SET not in self.result:
            self.update_result(Outcome.TRUE_W_NOT_IN_SET, schedule)

    def start_search(self):
        self.__search()
        if len(self.result) == 0:
            self.result[Outcome.NOT_FOUND] = {"depth": self.depth}
        return self.result

    def __search(self):

        current_depth = [self.original_schedule]
        next_depth = []

        self.visited_states.was_visited(self.original_schedule)
        while len(current_depth) > 0:
            while len(current_depth) > 0:
                schedule = current_depth.pop()
                # part of the PUP is done here to not add the neighbors of a schedule that is already beyond the utility loss threshold,
                # the utility loss can't be improved with more changes to the schedule without visiting undiscovered states (schedules).
                # this monotony is due to the start point (sorted by twct), bfs and neighborhood function (swapping neighboring jobs)
                if not self.satisfies_utility_constraint(schedule):
                    continue

                self.check_pup(schedule)

                # all outcomes have been found
                if len(self.result) == 3:
                    return

                for i in range(0, len(schedule.schedule_order) - 1):
                    neighbor_schedule: Schedule = copy.deepcopy(schedule)
                    neighbor_schedule.schedule_order[i], neighbor_schedule.schedule_order[i + 1] = \
                        neighbor_schedule.schedule_order[i + 1], neighbor_schedule.schedule_order[i]
                    if not self.visited_states.was_visited(neighbor_schedule):
                        next_depth.append(neighbor_schedule)

            current_depth = next_depth
            next_depth = []
            self.depth += 1
