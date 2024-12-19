import copy
from random import sample


def construct_neighbourhood(original_schedule_order, max_swaps, max_size_per_depth=None):
    neighbourhood = [original_schedule_order]
    swaps = 1
    while swaps <= max_swaps:
        new_neighbourhood = []
        for schedule_order in neighbourhood:
            for i in range(0, len(schedule_order) - 1):
                neighbour_schedule_order = copy.deepcopy(schedule_order)
                neighbour_schedule_order[i], neighbour_schedule_order[i + 1] = \
                    neighbour_schedule_order[i + 1], neighbour_schedule_order[i]
                new_neighbourhood.append(neighbour_schedule_order)
        swaps += 1
        neighbourhood = new_neighbourhood \
            if (not max_size_per_depth or len(new_neighbourhood) < max_size_per_depth) \
            else sample(new_neighbourhood, max_size_per_depth)
    return neighbourhood