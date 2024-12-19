import copy
import time
import typing
from enum import Enum
from typing import List, Callable
from ortools.sat.python import cp_model

from privacyschedulingtools.total_weighted_completion_time.entity.adversary.parallel_make_adversary import \
    ParallelMakeAdversary
from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import ParallelSchedule, \
    ParallelScheduleFactory
from privacyschedulingtools.total_weighted_completion_time.entity.result import Outcome, ParallelResult
from privacyschedulingtools.total_weighted_completion_time.entity.scheduling_parameters import \
    ParallelSchedulingParameters
from privacyschedulingtools.total_weighted_completion_time.pup.privacy_loss import distance_based_per_job
from privacyschedulingtools.total_weighted_completion_time.pup.util.transformations import Transformation
from privacyschedulingtools.total_weighted_completion_time.pup.util.duplicate_detection import DuplicateDetection
from privacyschedulingtools.total_weighted_completion_time.pup.utility_functions import parallel_machine_utilities

class Mode(Enum):
    SYSTEMATIC = 0
    HEURISTIC = 1
    RANDOM = 2


# TODO: Refactor neighborhood functions
# TODO: Refactor to PUPSolverFactory that instantiates PUP solver accordingly?
class PupParallelMake:
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
        self.adversary = ParallelMakeAdversary(schedule.params)
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

        percentage_change = 0 if self.original_utility == 0 else (
                                                                     abs(self.original_utility - utility)) / self.original_utility
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
                              self.factory.generate_schedule_with_weights(schedule, c, "make"))]
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
        if ParallelSchedule.is_valid_make_schedule(schedule, schedule.jobs):
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

        schedule = copy.deepcopy(self.original_schedule)
        queue = [schedule]
        self.visited_states.was_visited_parallel(schedule)

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
            elif transformation == Transformation.MOVE:
                queue.extend(self._add_move_neighbors(schedule))
            elif transformation == Transformation.MOVE_PROC:
                queue.extend(self._add_move_neighbors(schedule))
                queue.extend(self._add_processing_time_neighbors(schedule))
            elif transformation == Transformation.SWAP_ALL_PROC:
                queue.extend(self._add_swap_all(schedule))
                queue.extend(self._add_processing_time_neighbors(schedule))
            elif transformation == Transformation.SWAP_ALL:
                queue.extend(self._add_swap_all(schedule))
            elif transformation == Transformation.SWAP_ALL_REL:
                queue.extend(self._add_swap_all(schedule))
                queue.extend(self._add_release_time_neighbors(schedule))
            elif transformation == Transformation.MOVE_REL:
                queue.extend(self._add_move_neighbors(schedule))
                queue.extend(self._add_release_time_neighbors(schedule))
            elif transformation == Transformation.RELEASE_TIMES:
                queue.extend(self._add_swap_all(schedule))
            elif transformation == Transformation.MRP:
                queue.extend(self._add_move_neighbors(schedule))
                queue.extend(self._add_release_time_neighbors(schedule))
                queue.extend(self._add_processing_time_neighbors(schedule))
            elif transformation == Transformation.SRP:
                queue.extend(self._add_move_neighbors(schedule))
                queue.extend(self._add_release_time_neighbors(schedule))
                queue.extend(self._add_processing_time_neighbors(schedule))

        self.result["time"] = time.time() - self.start_time

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

            #possible_start = self._get_possible_start_time(neighbor_schedule, m, pos1)
            self._keep_valid_start_times(neighbor_schedule, m, pos1)
        else:
            # first gets start time of second job on corresponding machine,
            # and start times of following jobs need to be updated
            # same goes for second job
            neighbor_schedule.allocation[position2[0]][position2[1]].id = job1.id
            neighbor_schedule.allocation[position1[0]][position1[1]].id = job2.id

            #possible_start = self._get_possible_start_time(neighbor_schedule, position2[0], position2[1])
            self._keep_valid_start_times(neighbor_schedule, position2[0], position2[1])

            #possible_start = self._get_possible_start_time(neighbor_schedule, position1[0], position1[1])
            self._keep_valid_start_times(neighbor_schedule, position1[0], position1[1])

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

                    self._keep_valid_start_times(neighbor_schedule_plus, i, j + 1)
                    neighbor_schedule_plus.update_schedule_order()

                    if not self.visited_states.was_visited_parallel(neighbor_schedule_plus) \
                            and (not self.prune_neighbors or self.satisfies_utility_constraint(neighbor_schedule_plus)):
                        queue.append(neighbor_schedule_plus)

                if neighbor_schedule_minus.jobs[scheduled_job.id].processing_time - 1 >= min_limit:
                    neighbor_schedule_minus.jobs[scheduled_job.id].processing_time -= 1

                    # only when reducing processing times, we may bring a job forward before its release date...
                    # so when that is the case, we keep the original start times that are compliant with release dates
                    # assuming input schedule is valid
                    self._keep_valid_start_times(neighbor_schedule_minus, i, j + 1)
                    neighbor_schedule_minus.update_schedule_order()

                    if not self.visited_states.was_visited_parallel(neighbor_schedule_minus) \
                            and (
                            not self.prune_neighbors or self.satisfies_utility_constraint(neighbor_schedule_minus)):
                        queue.append(neighbor_schedule_minus)

        return queue

    def _add_release_time_neighbors(self, schedule: ParallelSchedule, limit: int = None):
        # TODO:
        #   - Adapt s.t. when new release time is assigned, that job is moved slightly forward/backward on machine if possible
        #   - Likewise for following jobs
        #   - Adapt in other functions that not original schedule is used but the weight vector of the schedule in question

        queue = []
        for i, machine in enumerate(schedule.allocation):
            for j, scheduled_job in enumerate(machine):
                neighbor_schedule_minus: ParallelSchedule = copy.deepcopy(schedule)
                neighbor_schedule_plus: ParallelSchedule = copy.deepcopy(schedule)
                original_r = self.original_schedule.jobs[scheduled_job.id].weight

                min_limit = schedule.params.processing_time_domain.get_min() if limit is None \
                    else max(schedule.params.processing_time_domain.get_min(), original_r - limit)
                max_limit = schedule.params.processing_time_domain.get_max() if limit is None \
                    else min(original_r + limit, schedule.params.processing_time_domain.get_max())

                if neighbor_schedule_plus.jobs[scheduled_job.id].weight + 1 <= max_limit:
                    neighbor_schedule_plus.jobs[scheduled_job.id].weight += 1

                    # TODO: adapt because now we need to consider job as well
                    self._keep_valid_start_times(neighbor_schedule_plus, i, j)
                    neighbor_schedule_plus.update_schedule_order()

                    # TODO: check whether weights are considered
                    if not self.visited_states.was_visited_parallel(neighbor_schedule_plus) \
                            and (not self.prune_neighbors or self.satisfies_utility_constraint(neighbor_schedule_plus)):
                        queue.append(neighbor_schedule_plus)

                if neighbor_schedule_minus.jobs[scheduled_job.id].weight - 1 >= min_limit:
                    neighbor_schedule_minus.jobs[scheduled_job.id].weight -= 1

                    # TODO adapt
                    self._keep_valid_start_times(neighbor_schedule_minus, i, j)
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

                            # remove job from old machine
                            # size of machine is decreased by 1 due to popping job
                            scheduled_job = neighbor_schedule.allocation[i].pop(j)
                            job_data = neighbor_schedule.jobs[scheduled_job.id]

                            # adapt start times of following jobs accordingly
                            # check for release dates, if new start time would be lower than release date,
                            # move to earliest possible start time and adapt following accordingly
                            # possible_start = self._get_possible_start_time(neighbor_schedule, i, j)
                            self._keep_valid_start_times(neighbor_schedule, i, j)

                            # insert job into new machine
                            # and adapt start times of moving job and following jobs accordingly
                            if move2machine == i and move2position > j:
                                move2position -= 1

                            if move2position != len(new_machine):
                                # ensure validity of make schedule by ensuring compliance with release dates
                                job_release_date = schedule.jobs[scheduled_job.id].weight
                                if move2position > 0:
                                    prev_job = new_machine[move2position - 1]
                                    possible_start = prev_job.start_time + neighbor_schedule.jobs[
                                        prev_job.id].processing_time
                                else:
                                    possible_start = 0

                                scheduled_job.start_time = max(job_release_date, possible_start)

                                # new job not yet inserted, that is why starting at same move2position
                                possible_start = scheduled_job.start_time + job_data.processing_time
                                self._keep_valid_start_times_helper(neighbor_schedule, move2machine, move2position,
                                                                    possible_start)
                            else:
                                if len(new_machine) != 0:
                                    preceding = new_machine[len(new_machine) - 1]
                                    possible_start = preceding.start_time + neighbor_schedule.jobs[
                                        preceding.id].processing_time
                                else:
                                    possible_start = 0

                                scheduled_job.start_time = max(
                                    possible_start,
                                    schedule.jobs[scheduled_job.id].weight)

                            new_machine.insert(move2position, scheduled_job)
                            neighbor_schedule.update_schedule_order()

                            if (not self.prune_neighbors or self.satisfies_utility_constraint(neighbor_schedule)) \
                                    and not self.visited_states.was_visited_parallel(neighbor_schedule):
                                queue.append(neighbor_schedule)

        return queue

    def _get_possible_start_time(self, neighbor_schedule, machine, position):
        if position > 0:
            prev_job = neighbor_schedule.allocation[machine][position - 1]
            possible_start = prev_job.start_time + neighbor_schedule.jobs[prev_job.id].processing_time
        else:
            possible_start = 0
        return possible_start

    def _keep_valid_start_times(self, new_schedule, m_idx, j_idx):
        possible_start = self._get_possible_start_time(new_schedule, m_idx, j_idx)
        self._keep_valid_start_times_helper(new_schedule, m_idx, j_idx, possible_start)

    def _keep_valid_start_times_helper(self, new_schedule, m_idx, j_idx, possible_start):
        machine = new_schedule.allocation[m_idx]
        for job in machine[j_idx:]:
            release_date = new_schedule.jobs[job.id].weight
            duration = new_schedule.jobs[job.id].processing_time

            job.start_time = max(release_date, possible_start)
            possible_start = job.start_time + duration
