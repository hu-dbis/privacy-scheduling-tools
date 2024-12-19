import time
from typing import Callable

from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import ParallelSchedule
from privacyschedulingtools.total_weighted_completion_time.entity.result import Outcome
from privacyschedulingtools.total_weighted_completion_time.entity.scheduling_parameters import \
    ParallelSchedulingParameters
from privacyschedulingtools.total_weighted_completion_time.pup.solver.pup_parallel import PupParallel, Transformation
from privacyschedulingtools.total_weighted_completion_time.pup.utility_functions import parallel_machine_utilities


class DepthLimitedPupParallel(PupParallel):
    def __init__(self, schedule: ParallelSchedule, privacy_threshold: float, utility_threshold: float,
                 utility_function: Callable = parallel_machine_utilities.calculate_twct,
                 schedule_parameters: ParallelSchedulingParameters = None, prune_utility: bool = False,
                 breadth_first: bool = True, limit_depth: int = None, timeout=None, callback: Callable = None):
        super().__init__(schedule, privacy_threshold, utility_threshold, utility_function, schedule_parameters,
                         prune_utility, timeout, callback)

        self.breadth_first = breadth_first
        self.limit_depth = limit_depth
        self.max_depth = 0

    def start_search(self, transformation: Transformation, stopping_criterion=lambda x: False):
        if self.breadth_first:
            self._search_systematic_bfs(transformation, stopping_criterion)
        else:
            self._search_systematic_dfs(transformation, stopping_criterion)

        # TODO: remove comment for "normal" use
        # if self.prune_neighbors and len(self.result["solutions"]) == 0:
        #    self.prune_neighbors = False
        #    self.__search_systematic(transformation, stopping_criterion)

        return self.result

    def _record_solution(self, schedule: ParallelSchedule, outcome: Outcome, privacy_loss: float,
                         pl_distance_per_job, utility_loss: float, dismissed: bool, solution_time=None, depth=None,
                         **kwargs):
        solution = self._build_solution_record(schedule, outcome, privacy_loss,
                                               pl_distance_per_job, utility_loss, dismissed,
                                               solution_time, depth)
        if dismissed:
            self.result["dismissed"].append(solution)
            idx = f"d{len(self.result['dismissed'])}"
        else:
            self.result["solutions"].append(solution)
            idx = f"s{len(self.result['solutions'])}"

        if self.callback is not None:
            self.callback(solution, idx)

    def _build_solution_record(self, schedule: ParallelSchedule, outcome: Outcome, privacy_loss: float,
                               pl_distance_per_job, utility_loss: float, dismissed: bool, solution_time=None,
                               depth=None, **kwargs):
        solution = super()._build_solution_record(schedule, outcome, privacy_loss, pl_distance_per_job, utility_loss,
                                                  dismissed, solution_time)
        solution.depth = depth
        return solution

    def _search_systematic_bfs(self, transformation: Transformation, stopping_criterion=lambda x: False):
        assert stopping_criterion is not None

        current_depth = 0
        queue = [{
            "depth": current_depth,
            "schedule": self.original_schedule
        }]
        self.visited_states.was_visited_parallel(self.original_schedule)
        start_time = time.time()

        while 0 < len(queue):
            item = queue.pop(0)
            current_depth = item["depth"]

            if self.limit_depth is not None and current_depth > self.limit_depth:
                break

            neighbors = self.__handle_next_item(item, transformation, stopping_criterion, start_time)

            if neighbors is None:
                break

            for neighbor in neighbors:
                queue.append({
                    "depth": current_depth + 1,
                    "schedule": neighbor
                })

        self.result["time"] = time.time() - start_time

    def _search_systematic_dfs(self, transformation: Transformation, stopping_criterion=lambda x: False):
        assert stopping_criterion is not None

        current_depth = 0
        stack = [{
            "depth": current_depth,
            "schedule": self.original_schedule
        }]
        self.visited_states.was_visited_parallel(self.original_schedule)
        start_time = time.time()

        while 0 < len(stack):
            item = stack.pop()
            current_depth = item["depth"]
            neighbors = self.__handle_next_item(item, transformation, stopping_criterion, start_time)

            if neighbors is None:
                break

            for neighbor in neighbors:
                if self.limit_depth is not None and current_depth < self.limit_depth:
                    stack.append({
                        "depth": current_depth + 1,
                        "schedule": neighbor
                    })

        self.result["time"] = time.time() - start_time

    def __handle_next_item(self, item, transformation, stopping_criterion, start_time):
        current_depth = item["depth"]
        self.max_depth = current_depth if current_depth > self.max_depth else self.max_depth
        schedule = item["schedule"]
        data = self.check_pup(schedule, time.time() - start_time)
        if data is not None:
            self._record_solution(data["schedule"], data["outcome"], data["privacy_loss"],
                                 data["pl_distance_per_job"], data["utility_loss"], data["dismissed"],
                                 data["solution_time"],
                                 current_depth)

        if stopping_criterion(self.result):
            return None

        neighbors = []

        if transformation == Transformation.PROCESSING_TIMES:
            neighbors = self._add_processing_time_neighbors(schedule, limit=2)
        elif transformation == Transformation.AVAILABILITY:
            neighbors = self._add_availability_neighbors(schedule)
        elif transformation == Transformation.SWAPPING_JOBS:
            neighbors = self._add_swap_neighbors(schedule)
        elif transformation == Transformation.DELETING_JOBS:
            neighbors = self._add_delete_neighbors(schedule)
        elif transformation == Transformation.ADDING_JOBS:
            pass
        elif transformation == Transformation.MOVE:
            neighbors = self._add_move_neighbors(schedule)
        elif transformation == Transformation.MOVE_PROC:
            neighbors = self._add_move_neighbors(schedule)
            neighbors.extend(self._add_processing_time_neighbors(schedule))

        return neighbors
