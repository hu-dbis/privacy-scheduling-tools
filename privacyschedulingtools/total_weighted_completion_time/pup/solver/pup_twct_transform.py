import random
from statistics import mean
from typing import List

from privacyschedulingtools.total_weighted_completion_time.pup.checker.solution_size_w_original import \
    inverse_scheduling_attack
from privacyschedulingtools.total_weighted_completion_time.entity.schedule import Schedule, generate_jobs
from privacyschedulingtools.total_weighted_completion_time.pup.util.duplicate_detection import DuplicateDetection
from privacyschedulingtools.total_weighted_completion_time.utility_functions.total_weighted_completion_time import \
    calculate
import copy

import numpy as np


# to setup the PUP and solve it
# solver based on bfs, swapping neighboring jobs
# privacy and utility thresholds are specified as decimals (0.01 -> 1%)
# utility loss is measured by the percentage change in the twct

class PupTransform:
    def __init__(self, schedule: Schedule, privacy_threshold: float, utility_threshold: float):
        self.original_schedule = schedule
        self.privacy_threshold = privacy_threshold
        self.utility_threshold = utility_threshold

        self.visited_states = DuplicateDetection()
        self.original_utility = calculate(self.original_schedule)
        self.result = {}

    def start_search(self):
        self.__search()
        return self.result

    def satisfies_utility_constraint(self, schedule: Schedule) -> bool:
        utility = calculate(schedule)

        percentage_change = (abs(self.original_utility - utility)) / self.original_utility
        # print(f"Utility changed by {percentage_change} from original utility {self.original_utility} to {utility}")

        return percentage_change <= self.utility_threshold

    def check_pup(self, schedule: Schedule):
        utility_satisfied = self.satisfies_utility_constraint(schedule)

        if utility_satisfied:
            attack_result = inverse_scheduling_attack.execute(schedule)
            candidates = attack_result['solutions']

            for i in range(len(candidates)):
                candidates[i] = {idx: w for (idx, w) in candidates[i].items() if idx < len(self.original_schedule.jobs)}
            pl_distance_per_job = privacy_loss_distance_based_per_job(self.original_schedule, candidates)

            # print("Privacy loss per job: ", pl_distance_per_job)
            if max(pl_distance_per_job) < self.privacy_threshold:
                self.result['schedule'] = schedule
                self.result['solution_found'] = True
                return True
        return False

    def __search(self):
        run_counts = 100
        for i in range(run_counts):
            # print("#### Iteration ####: ", i)
            schedule = self.__add_noise_to_p()
            # print("Perturbed schedule: ", schedule)
            solution_found = self.check_pup(schedule)

            if solution_found:
                return

        self.result['solution_found'] = False

    def __add_jobs(self):
        job_count = 2
        additional_jobs = generate_jobs(job_count, {
            'processing_time': (15, 30),
            'weight': (1, 1)
        })

        schedule = copy.deepcopy(self.original_schedule)

        for i in range(job_count):
            idx = len(self.original_schedule.jobs) + i
            position = random.choice(range(0, len(self.original_schedule.jobs)))
            additional_jobs[i].id = idx
            schedule.jobs[idx] = additional_jobs[i]
            schedule.schedule_order = np.insert(schedule.schedule_order, position, [idx])

        return schedule

    def __add_noise_to_p(self):
        mu, sigma = 0, 0.8
        noise = np.random.normal(mu, sigma, [len(self.original_schedule.jobs)])
        noise = [round(num) for num in noise]
        # print("Noise: ", noise)
        schedule = copy.deepcopy(self.original_schedule)

        change_prob = 0.3
        for i, noise_i in enumerate(noise):
            p_i = schedule.jobs[i].processing_time
            if random.random() <= change_prob:
                schedule.jobs[i].processing_time = min(schedule.processing_time_max,
                                                       max(p_i + noise_i, schedule.processing_time_min))
        return schedule

    def __search_systematic(self):
        queue = [self.original_schedule]
        self.visited_states.was_visited_jobs(self.original_schedule)
        while 0 < len(queue):
            schedule = queue.pop(0)
            solution_found = self.check_pup(schedule)
            if solution_found:
                return
            for i in range(len(schedule.jobs)):
                neighbor_schedule_minus: Schedule = copy.deepcopy(schedule)
                neighbor_schedule_plus: Schedule = copy.deepcopy(schedule)

                if neighbor_schedule_plus.jobs[i].processing_time + 1 in range(self.original_schedule.jobs[i].processing_time - 2, self.original_schedule.jobs[i].processing_time + 3): # < schedule.processing_time_max:
                    neighbor_schedule_plus.jobs[i].processing_time += 1

                    if not self.visited_states.was_visited_jobs(neighbor_schedule_plus) and self.satisfies_utility_constraint(neighbor_schedule_plus):
                        queue.append(neighbor_schedule_plus)

                if neighbor_schedule_minus.jobs[i].processing_time - 1 in range(self.original_schedule.jobs[i].processing_time - 2, self.original_schedule.jobs[i].processing_time + 3): # > schedule.processing_time_min and self.satisfies_utility_constraint(neighbor_schedule_minus):
                    neighbor_schedule_minus.jobs[i].processing_time -= 1

                    if not self.visited_states.was_visited_jobs(neighbor_schedule_minus):
                        queue.append(neighbor_schedule_minus)

        self.result['solution_found'] = False


def distance(x: int, values: List[int]):
    return mean([abs(x - v) for v in values])


def privacy_loss_distance_based_per_job(true_schedule, candidates):
    if (len(candidates) == 0):
        return [0] * len(true_schedule.jobs)

    true_weights = [j.weight for j in true_schedule.jobs.values()]
    candidates = [list(c.values()) for c in candidates]

    per_item_loss = []
    for i in range(len(true_schedule.jobs)):
        i_candidates = [c[i] for c in candidates]

        d = distance(true_weights[i], i_candidates)
        d_norm = distance(true_weights[i], range(true_schedule.weight_min, true_schedule.weight_max + 1));

        per_item_loss.append(1 - d / d_norm)

    return per_item_loss
