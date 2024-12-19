from privacyschedulingtools.total_weighted_completion_time.entity.domain import IntegerDomain
from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import ParallelSchedule, \
    ParallelScheduleFactory
from privacyschedulingtools.total_weighted_completion_time.entity.result import Outcome
from privacyschedulingtools.total_weighted_completion_time.entity.scheduling_parameters import \
    ParallelSchedulingParameters
from privacyschedulingtools.total_weighted_completion_time.pup.solver.pup_parallel import \
    PupTransformMultiple, Transformation, Mode
from privacyschedulingtools.total_weighted_completion_time.pup.utility_functions.parallel_machine_utilities import \
    calculate_cmax

# set schedule parameters
schedule_params = ParallelSchedulingParameters(
    job_count=10,
    machine_count=2,
    processing_time_domain=IntegerDomain(min_value=15, max_value=30),
    weight_domain=IntegerDomain(min_value=1, max_value=5)
)

# generate schedule
factory = ParallelScheduleFactory(schedule_params)
schedule = factory.generate_random_schedule_with_dispatching_rule()

# setup pup parameters
privacy_threshold = 0.8  # allowed checker success probability -> 1%
utility_threshold = 0.02  # allowed utility_functions deviation -> .2% change

# setup solver
searcher = PupTransformMultiple(schedule, privacy_threshold, utility_threshold, utility_function=calculate_cmax)
result = searcher.start_search(transformation=Transformation.MOVE_PROC, mode=Mode.SYSTEMATIC,
                               stopping_criterion=(lambda x, t: len(x["solutions"]) == 1))

if result["solutions"]:
    original_schedule = []
    for machine in schedule.allocation:
        original_schedule.append([{"start_time": j.start_time, "job": str(schedule.jobs[j.id])} for j in machine])

    perturbed_schedule = []
    solution = result["solutions"][0]
    for machine in solution.privatized_schedule.allocation:
        perturbed_schedule.append(
            [{"start_time": j.start_time, "job": str(solution.privatized_schedule.jobs[j.id])} for j in machine])

    print(original_schedule)
    print(perturbed_schedule)

    print(schedule.schedule_order)
    print(solution.privatized_schedule.schedule_order)

    print(f"privacy loss: {solution.privacy_loss}, utility loss: {solution.utility_loss}")
    print(f"outcome: {solution.outcome}, privacy loss per job: {solution.privacy_loss_per_job}")
else:
    utility_losses = [f.utility_loss for f in result["failures"]]
    privacy_losses = [f.privacy_loss for f in result["failures"]]
    print(f"Utility losses: {utility_losses}")
    print(f"Privacy losses: {privacy_losses}")
    print("No solution found!")
