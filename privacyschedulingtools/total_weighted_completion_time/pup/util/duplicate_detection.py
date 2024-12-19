import pickle

import xxhash as xxhash

from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import ParallelSchedule
from privacyschedulingtools.total_weighted_completion_time.entity.schedule import Schedule


# helper to check whether a schedule has been seen before
# checks the job order
class DuplicateDetection:

    def __init__(self):
        self.hasher = xxhash.xxh64()
        self.visited_hashset = set()

    def was_visited(self, schedule: Schedule):
        self.hasher.update(schedule.schedule_order)
        hash_value = self.hasher.intdigest()
        self.hasher.reset()
        if hash_value not in self.visited_hashset:
            self.visited_hashset.add(hash_value)
            return False
        else:
            return True

    def was_visited_parallel(self, schedule: ParallelSchedule):
        self.hasher.update(schedule.schedule_to_tuple().__str__().encode())
        hash_value = self.hasher.intdigest()
        self.hasher.reset()
        if hash_value not in self.visited_hashset:
            self.visited_hashset.add(hash_value)
            return False
        else:
            return True
