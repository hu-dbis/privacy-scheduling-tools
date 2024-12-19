import time
from typing import Callable

from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import ParallelSchedule
from privacyschedulingtools.total_weighted_completion_time.entity.scheduling_parameters import \
    ParallelSchedulingParameters
from privacyschedulingtools.total_weighted_completion_time.pup.solver.pup_parallel import PupParallel, Transformation
from privacyschedulingtools.total_weighted_completion_time.pup.utility_functions import parallel_machine_utilities


class PupParallelSBU(PupParallel):
    def __init__(self, schedule: ParallelSchedule, privacy_threshold: float, utility_threshold: float,
                 utility_function: Callable = parallel_machine_utilities.calculate_twct,
                 schedule_parameters: ParallelSchedulingParameters = None, prune_utility: bool = False,
                 minimize_utility_loss=True, timeout=None, callback: Callable = None):
        super().__init__(schedule, privacy_threshold, utility_threshold, utility_function, schedule_parameters,
                         prune_utility, timeout, callback)
        self.minimize_utility_loss = minimize_utility_loss

    def start_search(self, transformation: Transformation, stopping_criterion=lambda x: False):
        ''' DFS that moves towards utility threshold (assumption: high utility loss means lower privacy loss)
            but eventually explores (all or only part of?) schedules within utility range

        :param stopping_criterion:
        :return:
        '''
        assert stopping_criterion is not None

        stack = [self.original_schedule]
        self.visited_states.was_visited_parallel(self.original_schedule)
        start_time = time.time()

        while stack:
            schedule = stack.pop()
            data = self.check_pup(schedule, time.time() - start_time)
            if data is not None:
                self._record_solution(data["schedule"], data["outcome"], data["privacy_loss"],
                                      data["pl_distance_per_job"], data["utility_loss"], data["dismissed"],
                                      data["solution_time"])

            if stopping_criterion(self.result):
                break

            neighbor_schedules = self._add_move_neighbors(schedule)
            neighbor_schedules.extend(self._add_processing_time_neighbors(schedule))

            neighbor_schedules.sort(key=self.calculate_utility_loss, reverse=self.minimize_utility_loss)
            neighbor_schedules = [s for s in neighbor_schedules if self.satisfies_utility_constraint(s)]

            stack.extend(neighbor_schedules)

        time_difference = time.time() - start_time
        self.result["time"] = time_difference

        return self.result
