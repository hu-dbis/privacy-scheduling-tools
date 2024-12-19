from typing import List, Union

from numpy import mean

from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import ParallelSchedule
from privacyschedulingtools.total_weighted_completion_time.entity.schedule import Schedule


def exact_match(schedule: Schedule, candidates):
    if (len(candidates) == 0):
        return 0

    if (schedule not in candidates):
        return 0

    else:
        return 1 / len(candidates)


def distance_based_per_job(schedule: Union[Schedule, ParallelSchedule], candidates):
    if len(candidates) == 0:
        return [0] * len(schedule.jobs)

    # true_weights = [j.weight for j in schedule.jobs.values()]
    # candidates_values = [list(c.values()) for c in candidates]
    candidates_values = candidates

    per_item_loss = []
    for i, job in schedule.jobs.items():
        i_candidates = [c[i] for c in candidates_values]
        i_true_weight = job.weight

        d = distance(i_true_weight, i_candidates)
        d_norm = schedule.params.weight_domain.expected_distance(i_true_weight);

        per_item_loss.append(1 - d / d_norm)

    return per_item_loss


def distance(x: int, values: List[int]):
    distances = [abs(x - v) for v in values]
    return mean(distances)
