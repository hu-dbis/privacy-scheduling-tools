import copy

from privacyschedulingtools.total_weighted_completion_time.entity.adversary.adversary import WSPTAdversary
from privacyschedulingtools.total_weighted_completion_time.entity.schedule import Schedule
from privacyschedulingtools.total_weighted_completion_time.pup.util.duplicate_detection import DuplicateDetection
from privacyschedulingtools.total_weighted_completion_time.pup.utility_functions import single_machine_utilities


# to setup the PUP and solve it
# solver based on bfs, swapping neighboring jobs
# privacy and utility thresholds are specified as decimals (0.01 -> 1%)
# utility loss is measured by the percentage change in the average wait time


class PupAverageWaitTime:
    def __init__(self, schedule: Schedule, privacy_threshold: float, utility_threshold: float):
        self.original_schedule = schedule
        self.privacy_threshold = privacy_threshold
        self.utility_threshold = utility_threshold

        self.visited_states = DuplicateDetection()
        self.original_utility = single_machine_utilities.calculate_avg_wait_time(self.original_schedule)
        self.adversary = WSPTAdversary(schedule.params)
        self.result = {}

    def start_search(self):
        self.__search()
        return self.result

    def satisfies_utility_constraint(self, schedule: Schedule) -> bool:
        utility = single_machine_utilities.calculate_avg_wait_time(schedule)

        percentage_change = (abs(self.original_utility - utility)) / self.original_utility
        return percentage_change <= self.utility_threshold

    def check_pup(self, schedule: Schedule) -> bool:
        utility_satisfied = self.satisfies_utility_constraint(schedule)

        if utility_satisfied:
            if self.privacy_threshold == 0:
                necessary_solution_count = 0
            else:
                necessary_solution_count = 1 / self.privacy_threshold
            result_privacy = self.adversary.execute_attack(schedule, necessary_solution_count)
            if result_privacy['solution_count'] == necessary_solution_count or result_privacy['original_weight_vector_seen'] is False:
                self.result['schedule'] = schedule
                self.result['solution_found'] = True
                return True
        return False

    def __search(self):
        queue = [self.original_schedule]
        self.visited_states.was_visited(self.original_schedule)
        while 0 < len(queue):
            schedule = queue.pop(0)
            solution_found = self.check_pup(schedule)
            if solution_found:
                return
            for i in range(0, len(schedule.schedule_order) - 1):
                neighbor_schedule: Schedule = copy.deepcopy(schedule)
                neighbor_schedule.schedule_order[i], neighbor_schedule.schedule_order[i + 1] = neighbor_schedule.schedule_order[i + 1], neighbor_schedule.schedule_order[i]
                if not self.visited_states.was_visited(neighbor_schedule):
                    queue.append(neighbor_schedule)
        self.result['solution_found'] = False
