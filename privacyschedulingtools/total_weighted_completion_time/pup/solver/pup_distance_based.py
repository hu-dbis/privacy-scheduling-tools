import copy

from privacyschedulingtools.total_weighted_completion_time.entity.adversary.adversary import Adversary, WSPTAdversary
from privacyschedulingtools.total_weighted_completion_time.entity.result import Outcome, Result
from privacyschedulingtools.total_weighted_completion_time.entity.schedule import Schedule
from privacyschedulingtools.total_weighted_completion_time.pup import privacy_loss
from privacyschedulingtools.total_weighted_completion_time.pup.util.duplicate_detection import DuplicateDetection
from privacyschedulingtools.total_weighted_completion_time.pup.utility_functions import single_machine_utilities


# to setup the PUP and solve it
# solver based on bfs, swapping neighboring jobs
# privacy and utility thresholds are specified as decimals (0.01 -> 1%)
# utility loss is measured by the percentage change in the twct
class PupDistanceBased:

    def __init__(self, schedule: Schedule,
                 privacy_threshold: float,
                 utility_threshold: float,
                 adversary : Adversary = None,
                 collect_all_results = True):

        self.original_schedule = schedule
        self.privacy_threshold = privacy_threshold
        self.utility_threshold = utility_threshold
        self.required_results = 2 if collect_all_results else 1

        if adversary:
            self.adversary = adversary
        else:
            self.adversary = WSPTAdversary(schedule.params)

        self.visited_states = DuplicateDetection()
        self.original_utility = single_machine_utilities.calculate_twct(schedule)

        self.depth = 0
        self.result = []
        self.seen_outcomes = []

    def satisfies_utility_constraint(self, schedule: Schedule) -> bool:
        return self.get_utility_loss(schedule) <= self.utility_threshold

    def get_utility_loss(self, schedule:Schedule) -> float:
        utility = single_machine_utilities.calculate_twct(schedule)
        utility_loss = (abs(self.original_utility - utility)) / self.original_utility
        return utility_loss

    def check_pup(self, schedule: Schedule):
        # utility_functions check is here not necessary, it was already checked
        attack_result = self.adversary.execute_attack(schedule)

        if(attack_result["solution_count"] == 0):
            if not any(r.outcome == Outcome.SIZE_ZERO for r in self.result):
                self.result.append(Result(
                    outcome=Outcome.SIZE_ZERO,
                    original_schedule=self.original_schedule,
                    privatized_schedule=schedule,
                    privacy_loss=0,
                    utility_loss=self.get_utility_loss(schedule)
                ))
            return

        pl_per_job = privacy_loss.distance_based_per_job(schedule, attack_result["solutions"])
        pl_overall = max(pl_per_job)

        if (pl_overall <= self.privacy_threshold):
            if not any(r.outcome == Outcome.FOUND for r in self.result):
                self.result.append(Result(
                    outcome=Outcome.FOUND,
                    original_schedule=self.original_schedule,
                    privatized_schedule=schedule,
                    privacy_loss=pl_overall,
                    privacy_loss_per_job=pl_per_job,
                    utility_loss=self.get_utility_loss(schedule)
                    ))


    def start_search(self):
        self.__search()
        if len(self.result) == 0:
            self.result.append(Result(outcome=Outcome.NOT_FOUND))
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

                # two schedules (with different outcomes) have been found
                if len(self.result) >= self.required_results:
                    return

                # add schedules with one more swap
                for i in range(0, len(schedule.schedule_order) - 1):
                    neighbor_schedule: Schedule = copy.deepcopy(schedule)
                    neighbor_schedule.schedule_order[i], neighbor_schedule.schedule_order[i + 1] = \
                        neighbor_schedule.schedule_order[i + 1], neighbor_schedule.schedule_order[i]
                    if not self.visited_states.was_visited(neighbor_schedule):
                        next_depth.append(neighbor_schedule)

            current_depth = next_depth
            next_depth = []
            self.depth += 1
