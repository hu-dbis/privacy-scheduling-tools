from privacyschedulingtools.total_weighted_completion_time.entity.domain import IntegerDomain
from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import ParallelScheduleFactory
from privacyschedulingtools.total_weighted_completion_time.entity.scheduling_parameters import \
    ParallelSchedulingParameters
from privacyschedulingtools.total_weighted_completion_time.pup.solver.pup_parallel import PupParallel
from privacyschedulingtools.total_weighted_completion_time.pup.util.duplicate_detection import DuplicateDetection

dd = DuplicateDetection()
job_count = 10
processing_time_domain = (5, 30)
weight_domain = (1, 5)
domains = {
    "processing_time": processing_time_domain,
    "weight": weight_domain
}

machine_count = 2

# generate schedule
params = ParallelSchedulingParameters(
    weight_domain=IntegerDomain(1,5),
    processing_time_domain=IntegerDomain(15,30),
    job_count=10,
    machine_count=2
)
factory = ParallelScheduleFactory(params)
schedule = factory.generate_random_schedule()

assert not dd.was_visited_parallel(schedule)
assert dd.was_visited_parallel(schedule)

privacy_threshold = 0.01  # allowed checker success probability -> 1%
utility_threshold = 0.002  # allowed utility_functions deviation -> .2% change
searcher = PupParallel(schedule, privacy_threshold, utility_threshold)
queue = []

searcher._add_delete_neighbors(schedule)

for q in queue:
    assert not dd.was_visited_parallel(q)
for q in queue:
    assert dd.was_visited_parallel(q)

queue = []
searcher._add_processing_time_neighbors(schedule)

for q in queue:
    assert not dd.was_visited_parallel(q)
for q in queue:
    assert dd.was_visited_parallel(q)

queue = []
searcher._add_availability_neighbors(schedule)

for q in queue:
    assert not dd.was_visited_parallel(q)
for q in queue:
    assert dd.was_visited_parallel(q)

queue = []
searcher._add_swap_neighbors(schedule)

for q in queue:
    assert not dd.was_visited_parallel(q)
for q in queue:
    assert dd.was_visited_parallel(q)

queue = []
searcher._add_move_neighbors(schedule)

for q in queue:
    assert not dd.was_visited_parallel(q)
for q in queue:
    assert dd.was_visited_parallel(q)

