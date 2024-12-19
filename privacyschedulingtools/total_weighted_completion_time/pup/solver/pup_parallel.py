import copy
import time
import typing
from enum import Enum
from typing import List, Callable
from ortools.sat.python import cp_model

from privacyschedulingtools.total_weighted_completion_time.entity.adversary.parallel_adversary import \
    ParallelWSPTAdversary
from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import ParallelSchedule, \
    ParallelScheduleFactory
from privacyschedulingtools.total_weighted_completion_time.entity.result import Outcome, ParallelResult
from privacyschedulingtools.total_weighted_completion_time.entity.scheduling_parameters import \
    ParallelSchedulingParameters
from privacyschedulingtools.total_weighted_completion_time.pup.privacy_loss import distance_based_per_job
from privacyschedulingtools.total_weighted_completion_time.pup.util.duplicate_detection import DuplicateDetection
from privacyschedulingtools.total_weighted_completion_time.pup.util.transformations import Transformation
from privacyschedulingtools.total_weighted_completion_time.pup.utility_functions import parallel_machine_utilities


class Mode(Enum):
    SYSTEMATIC = 0
    HEURISTIC = 1
    RANDOM = 2


# TODO: Refactor neighborhood functions
class PupParallel:
    """
        PUP solver for parallel TWCT schedules, based on BFS and selected perturbation functions

        Parameters: schedule:               schedule for which the PUP is to be solved privacy_threshold:
        maximum allowed privacy loss (specified as decimals (0.01 -> 1%)) utility_loss:           maximum allowed
        utility loss (specified as decimals (0.01 -> 1%)) utility_function:       function that is used to calculate
        utility loss (utility loss is measured by the percentage change in utility) schedule_parameters:
        parameters of the scheduling environment of schedule prune_utility:          flag whether to not explore
        neighborhood of schedules exceeding utility loss further timeout:                timeout for PUP callback:
                   used to record solutions while continuing the search

    """

    def __init__(self, schedule: ParallelSchedule, privacy_threshold: float, utility_threshold: float,
                 utility_function: Callable = parallel_machine_utilities.calculate_twct,
                 schedule_parameters: ParallelSchedulingParameters = None, prune_utility: bool = False, timeout=None,
                 callback: Callable = None):
        self.original_schedule = schedule
        self.privacy_threshold = privacy_threshold
        self.utility_threshold = utility_threshold
        self.utility_function = utility_function
        self.adversary = ParallelWSPTAdversary(schedule.params)
        self.factory = ParallelScheduleFactory(schedule_parameters) if schedule_parameters else None
        self.prune_neighbors = prune_utility
        self.timeout = timeout
        self.callback = callback

        self.visited_states = DuplicateDetection()
        self.original_utility = utility_function(self.original_schedule)
        self.result = {"original": self.original_schedule, "solutions": [], "dismissed": []}
        self.start_time = -1

    def start_search(self, transformation: Transformation, stopping_criterion=None):
        """
            Search to solve the PUP

            Parameters:
                transformation:     Perturbation function to be used during the search
                stopping_criterion: when to stop the search
        """
        # stopping_criterion could be None (until visited all states),
        # or len(self.result["solutions"]) == 1 (stop when solution is found), set time limit etc?
        self.start_time = time.time()
        self._search_systematic(transformation, stopping_criterion)

        if self.prune_neighbors and len(self.result["solutions"]) == 0:
            self.prune_neighbors = False
            self.visited_states = DuplicateDetection()
            self._search_systematic(transformation, stopping_criterion)

        return self.result

    def calculate_utility_loss(self, schedule: ParallelSchedule) -> float:
        """
            Calculates utility loss of a given schedule by percentage change of the utility
            compared to the original schedule
        """
        utility = self.utility_function(schedule)

        percentage_change = 0 if self.original_utility == 0 else (abs(self.original_utility - utility)) / self.original_utility
            # we can assume this to be valid because completion times of zero should never happen due to p > 0 and
            # avg waiting times of 0 should only happen due to having less jobs than available machines
        # print(f"Utility changed by {percentage_change} from original utility {self.original_utility} to {utility}")

        return percentage_change

    def satisfies_utility_constraint(self, arg: typing.Union[float, ParallelSchedule]):
        """
            Checks utility constraint (by use of utility_threshold) for a given schedule or utility value
        """
        if isinstance(arg, ParallelSchedule):
            return self.calculate_utility_loss(arg) <= self.utility_threshold
        elif isinstance(arg, float):
            return arg <= self.utility_threshold
        else:
            return False

    def calculate_privacy_loss(self, schedule: ParallelSchedule, attack_result) -> (typing.List[float], int):
        """
            Calculates privacy loss per job for a given schedule and the result of the simulated attack on the schedule
            Incorporates check f(w) = s before considering an attack result as a valid candidate in the privacy loss
        """
        if self.factory:
            # Check for each candidate weight w, if f(w) produces given schedule
            # print(f"candidate size before check: {len(attack_result['solutions'])}")
            candidates = [c for c in attack_result['solutions'] if
                          schedule.compare_published_version(
                              self.factory.generate_schedule_with_weights(schedule, c))]
            # print(f"candidate size after check: {len(candidates)}")
        else:
            candidates = attack_result['solutions']
            # print(f"candidate size without check: {len(candidates)}")

        for i in range(len(candidates)):
            candidates[i] = {idx: w for (idx, w) in candidates[i].items() if idx < len(self.original_schedule.jobs)}
        pl_distance_per_job = distance_based_per_job(self.original_schedule, candidates)

        return pl_distance_per_job, len(candidates)

    def satisfies_privacy_constraint(self, privacy_loss: int):
        return privacy_loss <= self.privacy_threshold

    def check_pup(self, schedule: ParallelSchedule, run_time: float = None, **kwargs):
        """
            Checks if a schedule satisfies the PUP conditions by checking utility constraint, and then using the
            max privacy loss of a job to check the privacy constraint
        """
        st = time.time()
        if ParallelSchedule.is_valid_twct_schedule(schedule):
            utility_loss = self.calculate_utility_loss(schedule)

            if self.satisfies_utility_constraint(utility_loss):
                attack_result = self.adversary.execute_attack(schedule)
                pl_distance_per_job, candidate_size = self.calculate_privacy_loss(schedule, attack_result)

                # print("Privacy loss per job: ", pl_distance_per_job)
                privacy_loss = max(pl_distance_per_job)
                if self.satisfies_privacy_constraint(privacy_loss):
                    outcome = Outcome.FOUND if candidate_size else Outcome.SIZE_ZERO
                    # and self.timeout is not None and run_time < self.timeout
                    solution_time = run_time + (time.time() - st) \
                        if run_time is not None \
                        else None
                    dismissed = False if attack_result["status"] != cp_model.UNKNOWN else True

                    return {
                        "schedule": schedule,
                        "dismissed": dismissed,
                        "outcome": outcome,
                        "utility_loss": utility_loss,
                        "privacy_loss": privacy_loss,
                        "pl_distance_per_job": pl_distance_per_job,
                        "solution_time": solution_time
                    }

        return None

    def _record_solution(self, schedule: ParallelSchedule, outcome: Outcome, privacy_loss: float,
                         pl_distance_per_job, utility_loss: float, dismissed: bool, solution_time=None, **kwargs):
        """
             Records a solution for the PUP
                A schedule is recorded as dismissed if the attack during the privacy loss computation
                    did not finish due to a timeout,
                otherwise it is recorded as a valid solution
        """

        solution = self._build_solution_record(schedule, outcome, privacy_loss,
                                               pl_distance_per_job, utility_loss, dismissed,
                                               solution_time)
        if dismissed:
            self.result["dismissed"].append(solution)
            idx = f"d{len(self.result['dismissed'])}"
        else:
            self.result["solutions"].append(solution)
            idx = f"s{len(self.result['solutions'])}"

        if self.callback is not None:
            self.callback(solution, idx)

    def _build_solution_record(self, schedule: ParallelSchedule, outcome: Outcome, privacy_loss: float,
                               pl_distance_per_job, utility_loss: float, dismissed: bool, solution_time=None, **kwargs):
        return ParallelResult(
            outcome=outcome,
            original_schedule=self.original_schedule,
            privatized_schedule=schedule,
            privacy_loss=privacy_loss,
            privacy_loss_per_job=pl_distance_per_job,
            utility_loss=utility_loss,
            time_found=solution_time,
            dismissed=dismissed)

    def _search_systematic(self, transformation: Transformation, stopping_criterion=lambda x: False):
        assert stopping_criterion is not None

        queue = [self.original_schedule]
        self.visited_states.was_visited_parallel(self.original_schedule)

        while 0 < len(queue):
            schedule = queue.pop(0)
            data = self.check_pup(schedule, time.time() - self.start_time)
            if data is not None:
                self._record_solution(data["schedule"], data["outcome"], data["privacy_loss"],
                                      data["pl_distance_per_job"], data["utility_loss"], data["dismissed"],
                                      data["solution_time"])

            if stopping_criterion(self.result):
                break

            if transformation == Transformation.PROCESSING_TIMES:
                queue.extend(self._add_processing_time_neighbors(schedule))
            elif transformation == Transformation.AVAILABILITY:
                queue.extend(self._add_availability_neighbors(schedule))
            elif transformation == Transformation.SWAPPING_JOBS:
                queue.extend(self._add_swap_neighbors(schedule))
            elif transformation == Transformation.DELETING_JOBS:
                queue.extend(self._add_delete_neighbors(schedule))
            elif transformation == Transformation.ADDING_JOBS:
                pass
            elif transformation == Transformation.MOVE:
                queue.extend(self._add_move_neighbors(schedule))
            elif transformation == Transformation.MOVE_PROC:
                queue.extend(self._add_move_neighbors(schedule))
                queue.extend(self._add_processing_time_neighbors(schedule))
            elif transformation == Transformation.SWAP_PROC:
                queue.extend(self._add_swap_neighbors(schedule))
                queue.extend(self._add_processing_time_neighbors(schedule))
            elif transformation == Transformation.ALT_MOVE_PROC:
                queue.extend(self._add_move_neighbors_alt(schedule))
                queue.extend(self._add_processing_time_neighbors(schedule))
            elif transformation == Transformation.ALT_MOVE:
                queue.extend(self._add_move_neighbors_alt(schedule))
            elif transformation == Transformation.SWAP_ALL_PROC:
                queue.extend(self._add_swap_all(schedule))
                queue.extend(self._add_processing_time_neighbors(schedule))
            elif transformation == Transformation.SWAP_ALL:
                queue.extend(self._add_swap_all(schedule))

        self.result["time"] = time.time() - self.start_time

    def __adapt_following_start_times(self, machine: List, start_index: int, end_index: int, sign: int, summand: int):
        for follows in range(start_index, end_index):
            machine[follows].start_time += sign * summand

    def _add_delete_neighbors(self, schedule):
        queue = []
        for i, machine in enumerate(schedule.allocation):
            for j in range(len(machine)):
                neighbor_schedule: ParallelSchedule = copy.deepcopy(schedule)
                j_id = neighbor_schedule.allocation[i][j].id
                j_processing_time = schedule.jobs[j_id].processing_time
                neighbor_schedule.allocation[i].pop(j)
                del neighbor_schedule.jobs[j_id]

                for follows in range(j, len(machine) - 1):
                    neighbor_schedule.allocation[i][follows].start_time -= j_processing_time

                neighbor_schedule.update_schedule_order()

                if not self.visited_states.was_visited_parallel(neighbor_schedule) \
                        and self.satisfies_utility_constraint(neighbor_schedule):
                    queue.append(neighbor_schedule)

        return queue

    def _add_availability_neighbors(self, schedule):
        queue = []
        for i, machine in enumerate(schedule.allocation):
            for j in range(len(machine)):
                neighbor_schedule: ParallelSchedule = copy.deepcopy(schedule)

                for delayed in range(j, len(machine)):
                    neighbor_schedule.allocation[i][delayed].start_time += 1

                neighbor_schedule.update_schedule_order()

                # adapt was_visited (to use start times)
                if not self.visited_states.was_visited_parallel(neighbor_schedule) \
                        and self.satisfies_utility_constraint(neighbor_schedule):
                    queue.append(neighbor_schedule)

        return queue

    # TODO: add general swapping neighbors
    def _swap(self, schedule, position1, position2):
        """
            Parameters:
                schedule    schedule where jobs are to be swapped
                first       tuple describing machine and position of first job
                second      tuple describing machine and position of second job
        """
        neighbor_schedule: ParallelSchedule = copy.deepcopy(schedule)

        job1 = schedule.allocation[position1[0]][position1[1]]
        job2 = schedule.allocation[position2[0]][position2[1]]

        p1 = schedule.jobs[job1.id].processing_time
        p2 = schedule.jobs[job2.id].processing_time

        # scheduled on the same machine
        if position1[0] == position2[0]:
            m = position1[0]
            if position1[1] < position2[1]:
                first = job1
                second = job2
                pos1 = position1[1]
                pos2 = position2[1]
            else:
                first = job2
                second = job1
                pos1 = position2[1]
                pos2 = position1[1]

            neighbor_schedule.allocation[m][pos1].id = second.id
            neighbor_schedule.allocation[m][pos2].id = first.id

            for j in range(pos1 + 1, pos2 + 1):
                left_neighbor = neighbor_schedule.allocation[m][j - 1]
                neighbor_schedule.allocation[m][j].start_time = left_neighbor.start_time + \
                                                                neighbor_schedule.jobs[left_neighbor.id].processing_time
        else:
            # first gets start time of second job on corresponding machine,
            # and start times of following jobs need to be updated
            # same goes for second job
            neighbor_schedule.allocation[position2[0]][position2[1]].id = job1.id
            neighbor_schedule.allocation[position1[0]][position1[1]].id = job2.id

            # TODO check direction
            p_diff = p1 - p2

            for j in range(position1[1] + 1, len(neighbor_schedule.allocation[position1[0]])):
                neighbor_schedule.allocation[position1[0]][j].start_time -= p_diff

            for j in range(position2[1] + 1, len(neighbor_schedule.allocation[position2[0]])):
                neighbor_schedule.allocation[position2[0]][j].start_time += p_diff

        neighbor_schedule.update_schedule_order()

        return neighbor_schedule

    def _add_swap_all(self, schedule):
        """
        Swaps each job with each other job to produce neighboring schedules
        """
        queue = []
        for m1, machine1 in enumerate(schedule.allocation):
            for j1, job1 in enumerate(machine1):
                for i in range(m1, len(schedule.allocation)):
                    start = j1 + 1 if i == m1 else 0

                    for j in range(start, len(schedule.allocation[i])):
                        neighbor_schedule = self._swap(schedule, (m1, j1), (i, j))

                        if not self.visited_states.was_visited_parallel(neighbor_schedule) \
                                and (not self.prune_neighbors or self.satisfies_utility_constraint(neighbor_schedule)):
                            queue.append(neighbor_schedule)
        return queue

    def _add_swap_machine_neighbors(self, schedule):
        # only does right swaps
        queue = []
        for m, machine in enumerate(schedule.allocation):
            for j, job in enumerate(machine):
                if j != len(machine) - 1:
                    neighbor_schedule = self._swap(schedule, (m, j), (m, j + 1))

                    if not self.visited_states.was_visited_parallel(neighbor_schedule) \
                            and (not self.prune_neighbors or self.satisfies_utility_constraint(neighbor_schedule)):
                        queue.append(neighbor_schedule)
                else:
                    if m != len(schedule.allocation) - 1 and schedule.allocation[m + 1][0] is not None:
                        neighbor_schedule = self._swap(schedule, (m, j), (m + 1, 0))

                        if not self.visited_states.was_visited_parallel(neighbor_schedule) \
                                and (not self.prune_neighbors or self.satisfies_utility_constraint(neighbor_schedule)):
                            queue.append(neighbor_schedule)

        return queue

    def _add_swap_time_neighbors(self, schedule):
        queue = []
        for i in range(len(schedule.schedule_order) - 1):
            first_idx = schedule.schedule_order[i]
            second_idx = schedule.schedule_order[i + 1]

            first = None
            second = None
            for m, machine in enumerate(schedule.allocation):
                for j, scheduled_job in enumerate(machine):
                    if scheduled_job.id == first_idx:
                        first = (m, j)
                    elif scheduled_job.id == second_idx:
                        second = (m, j)

                    if first is not None and second is not None:
                        break  # TODO: break in first outer loop as well?

            neighbor_schedule = self._swap(schedule, first, second)

            if not self.visited_states.was_visited_parallel(neighbor_schedule) \
                    and (not self.prune_neighbors or self.satisfies_utility_constraint(neighbor_schedule)):
                queue.append(neighbor_schedule)

        return queue

    def _add_swap_neighbors(self, schedule):
        queue = []
        for i in range(len(schedule.schedule_order) - 1):
            neighbor_schedule: ParallelSchedule = copy.deepcopy(schedule)
            first_idx = neighbor_schedule.schedule_order[i]
            second_idx = neighbor_schedule.schedule_order[i + 1]
            neighbor_schedule.schedule_order[i] = second_idx
            neighbor_schedule.schedule_order[i + 1] = first_idx

            first_tuple = None
            second_tuple = None
            for m, machine in enumerate(neighbor_schedule.allocation):
                for j, scheduled_job in enumerate(machine):
                    if scheduled_job.id == first_idx:
                        first_tuple = (m, j, scheduled_job)
                    elif scheduled_job.id == second_idx:
                        second_tuple = (m, j, scheduled_job)

                    if first_tuple is not None and second_tuple is not None:
                        break

            assert first_tuple is not None and second_tuple is not None

            # change start times of jobs in schedule
            f_machine, f_position, f_scheduled_job = first_tuple
            s_machine, s_position, s_scheduled_job = second_tuple

            f_processing_time = neighbor_schedule.jobs[f_scheduled_job.id].processing_time
            s_processing_time = neighbor_schedule.jobs[s_scheduled_job.id].processing_time

            if f_machine == s_machine:
                # second job gets start time of first
                temp_f_idx = f_scheduled_job.id
                neighbor_schedule.allocation[f_machine][f_position].id = s_scheduled_job.id

                # first job gets start time of orignal start time + processing time of second
                neighbor_schedule.allocation[f_machine][
                    s_position].start_time = f_scheduled_job.start_time + s_processing_time
                neighbor_schedule.allocation[f_machine][s_position].id = temp_f_idx
            else:
                # first gets start time of second job on corresponding machine,
                # and start times of following jobs need to be updated
                # same goes for second job

                temp_s_idx = s_scheduled_job.id
                neighbor_schedule.allocation[s_machine][s_position].id = f_scheduled_job.id
                neighbor_schedule.allocation[f_machine][f_position].id = temp_s_idx
                p_difference = f_processing_time - s_processing_time

                for j in range(f_position + 1, len(neighbor_schedule.allocation[f_machine])):
                    neighbor_schedule.allocation[f_machine][j].start_time -= p_difference

                for j in range(s_position + 1, len(neighbor_schedule.allocation[s_machine])):
                    neighbor_schedule.allocation[s_machine][j].start_time += p_difference

            neighbor_schedule.update_schedule_order()

            if not self.visited_states.was_visited_parallel(neighbor_schedule) \
                    and (not self.prune_neighbors or self.satisfies_utility_constraint(neighbor_schedule)):
                queue.append(neighbor_schedule)

        return queue

    # TODO: add processing time swapping
    def _add_processing_time_neighbors(self, schedule: ParallelSchedule, limit: int = None):
        queue = []
        for i, machine in enumerate(schedule.allocation):
            for j, scheduled_job in enumerate(machine):
                neighbor_schedule_minus: ParallelSchedule = copy.deepcopy(schedule)
                neighbor_schedule_plus: ParallelSchedule = copy.deepcopy(schedule)
                original_p = self.original_schedule.jobs[scheduled_job.id].processing_time

                min_limit = schedule.params.processing_time_domain.get_min() if limit is None \
                    else max(schedule.params.processing_time_domain.get_min(), original_p - limit)
                max_limit = schedule.params.processing_time_domain.get_max() if limit is None \
                    else min(original_p + limit, schedule.params.processing_time_domain.get_max())

                if neighbor_schedule_plus.jobs[scheduled_job.id].processing_time + 1 <= max_limit:
                    neighbor_schedule_plus.jobs[scheduled_job.id].processing_time += 1

                    for following in range(j + 1, len(machine)):
                        neighbor_schedule_plus.allocation[i][following].start_time += 1

                    neighbor_schedule_plus.update_schedule_order()

                    if not self.visited_states.was_visited_parallel(neighbor_schedule_plus) \
                            and (not self.prune_neighbors or self.satisfies_utility_constraint(neighbor_schedule_plus)):
                        queue.append(neighbor_schedule_plus)

                if neighbor_schedule_minus.jobs[scheduled_job.id].processing_time - 1 >= min_limit:
                    neighbor_schedule_minus.jobs[scheduled_job.id].processing_time -= 1

                    for following in range(j + 1, len(machine)):
                        neighbor_schedule_minus.allocation[i][following].start_time -= 1

                    neighbor_schedule_minus.update_schedule_order()

                    if not self.visited_states.was_visited_parallel(neighbor_schedule_minus) \
                            and (
                            not self.prune_neighbors or self.satisfies_utility_constraint(neighbor_schedule_minus)):
                        queue.append(neighbor_schedule_minus)

        return queue

    def _add_move_neighbors(self, schedule):
        queue = []
        for i in range(schedule.params.machine_count):
            size_old_machine = len(schedule.allocation[i])
            for j in range(size_old_machine):
                for move2machine in range(schedule.params.machine_count):
                    for move2position in range(len(schedule.allocation[move2machine]) + 1):
                        if not (move2machine == i and (move2position == j or move2position == j + 1)):
                            neighbor_schedule: ParallelSchedule = copy.deepcopy(schedule)
                            new_machine = neighbor_schedule.allocation[move2machine]

                            # remove job from old machine and adapt start times of following jobs accordingly
                            scheduled_job = neighbor_schedule.allocation[i].pop(j)
                            job_data = neighbor_schedule.jobs[scheduled_job.id]

                            # size of machine is decreased by 1 due to popping job
                            for following in range(j, size_old_machine - 1):
                                neighbor_schedule.allocation[i][following].start_time -= job_data.processing_time

                            # insert job into new machine
                            # and adapt start times of moving job and following jobs accordingly
                            if move2machine == i and move2position > j:
                                move2position -= 1

                            if move2position != len(new_machine):
                                scheduled_job.start_time = new_machine[move2position].start_time
                                for following in range(move2position, len(new_machine)):
                                    new_machine[following].start_time += job_data.processing_time
                            else:
                                if len(new_machine) != 0:
                                    preceding = new_machine[len(new_machine) - 1]
                                    scheduled_job.start_time = preceding.start_time + neighbor_schedule.jobs[
                                        preceding.id].processing_time
                                else:
                                    scheduled_job.start_time = 0

                            new_machine.insert(move2position, scheduled_job)
                            neighbor_schedule.update_schedule_order()

                            if (not self.prune_neighbors or self.satisfies_utility_constraint(neighbor_schedule)) \
                                    and not self.visited_states.was_visited_parallel(neighbor_schedule):
                                queue.append(neighbor_schedule)

        return queue

    def _add_move_neighbors_alt(self, schedule):
        queue = []
        for i, machine in enumerate(schedule.allocation):
            for j, _ in enumerate(machine):
                left_schedule = self.__move_left(schedule, i, j)

                if left_schedule is not None and \
                        (not self.prune_neighbors or self.satisfies_utility_constraint(left_schedule)) and \
                        not self.visited_states.was_visited_parallel(left_schedule):
                    queue.append(left_schedule)

                right_schedule = self.__move_right(schedule, i, j)

                if right_schedule is not None \
                        and (not self.prune_neighbors or self.satisfies_utility_constraint(right_schedule)) \
                        and not self.visited_states.was_visited_parallel(right_schedule):
                    queue.append(right_schedule)

        return queue

    def __move_right(self, schedule: ParallelSchedule, i: int, j: int) -> typing.Union[ParallelSchedule, None]:
        # move to the right
        right_schedule: ParallelSchedule = copy.deepcopy(schedule)
        original_size = len(right_schedule.allocation[i])
        job = right_schedule.allocation[i].pop(j)
        job_data = right_schedule.jobs[job.id]

        if j == original_size - 1:
            # last job -> gets moved to next machine
            if i != len(right_schedule.allocation) - 1:
                next_machine = right_schedule.allocation[i + 1]
                job.start_time = 0

                # adapt start times of following jobs on next machine
                for following in range(len(schedule.allocation[i + 1])):
                    right_schedule.allocation[i + 1][following].start_time += job_data.processing_time

                next_machine.insert(0, job)

            else:
                return None
        else:
            # job just swaps position with right neighbor
            right_job = right_schedule.allocation[i][j]
            right_job.start_time = job.start_time
            job.start_time += right_schedule.jobs[right_job.id].processing_time
            right_schedule.allocation[i].insert(j + 1, job)

        right_schedule.update_schedule_order()
        return right_schedule

    def __move_left(self, schedule: ParallelSchedule, i: int, j: int) -> typing.Union[ParallelSchedule, None]:
        left_schedule: ParallelSchedule = copy.deepcopy(schedule)
        job = left_schedule.allocation[i].pop(j)
        job_data = left_schedule.jobs[job.id]

        if j == 0:
            # job is the first job and gets moved to previous machine
            if i != 0:
                prev_machine = left_schedule.allocation[i - 1]
                if prev_machine:
                    last_job = prev_machine[-1]
                    job.start_time = last_job.start_time + left_schedule.jobs[last_job.id].processing_time
                else:
                    job.start_time = 0

                prev_machine.append(job)

                for following in range(j, len(left_schedule.allocation[i])):
                    left_schedule.allocation[i][following].start_time -= job_data.processing_time
            else:
                return None
        else:
            # job just swaps position with left neighbor
            left_job = left_schedule.allocation[i][j - 1]
            job.start_time = left_job.start_time
            left_job.start_time += job_data.processing_time
            left_schedule.allocation[i].insert(j - 1, job)

        left_schedule.update_schedule_order()
        return left_schedule
