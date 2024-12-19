import copy
from unittest import TestCase

from privacyschedulingtools.total_weighted_completion_time.entity.domain import IntegerDomain
from privacyschedulingtools.total_weighted_completion_time.entity.job import Job
from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import ParallelSchedule, \
    ParallelScheduleFactory
from privacyschedulingtools.total_weighted_completion_time.entity.scheduling_parameters import \
    ParallelSchedulingParameters
from privacyschedulingtools.total_weighted_completion_time.pup.solver.pup_parallel import PupParallel
from privacyschedulingtools.total_weighted_completion_time.pup.utility_functions import parallel_machine_utilities


def create_schedule():
    params = ParallelSchedulingParameters(
        job_count=3,
        machine_count=2,
        processing_time_domain=IntegerDomain(5, 30),
        weight_domain=IntegerDomain(1, 5)
    )
    factory = ParallelScheduleFactory(params)
    jobs = [[(1, 5, 1), (2, 30, 1)], [(0, 30, 5)]]
    schedule = factory.from_job_tuple_list(jobs)
    return schedule

def create_schedule2():
    params = ParallelSchedulingParameters(
        job_count=5,
        machine_count=2,
        processing_time_domain=IntegerDomain(5, 30),
        weight_domain=IntegerDomain(1, 5)
    )
    factory = ParallelScheduleFactory(params)
    jobs = [[(1, 5, 2), (2, 25, 1), (3, 30, 1)], [(0, 20, 5), (4, 30, 1)]]
    schedule = factory.from_job_tuple_list(jobs)
    return schedule

def create_schedule_single_machine():
    params = ParallelSchedulingParameters(
        job_count=5,
        machine_count=1,
        processing_time_domain=IntegerDomain(5, 30),
        weight_domain=IntegerDomain(1, 5)
    )
    factory = ParallelScheduleFactory(params)
    jobs = [[(0, 5, 1), (1, 5, 1), (2, 5, 5), (3, 5, 1), (4, 5, 1)]]
    schedule = factory.from_job_tuple_list(jobs)
    return schedule

def create_schedule_single_machine2():
    params = ParallelSchedulingParameters(
        job_count=4,
        machine_count=1,
        processing_time_domain=IntegerDomain(5, 30),
        weight_domain=IntegerDomain(1, 5)
    )
    factory = ParallelScheduleFactory(params)
    jobs = [[(0, 5, 4), (1, 5, 2), (2, 10, 5), (3, 15, 1)]]
    schedule = factory.from_job_tuple_list(jobs)
    return schedule

def create_schedule_same_p():
    params = ParallelSchedulingParameters(
        job_count=3,
        machine_count=1,
        processing_time_domain=IntegerDomain(10, 10),
        weight_domain=IntegerDomain(1, 5)
    )

    factory = ParallelScheduleFactory(params)
    jobs = [[(0, 10, 5), (1, 10, 3), (2, 10, 1)]]
    schedule = factory.from_job_tuple_list(jobs)
    return schedule

def create_schedule_empty_machine():
    params = ParallelSchedulingParameters(
        job_count=2,
        machine_count=3,
        processing_time_domain=IntegerDomain(10, 10),
        weight_domain=IntegerDomain(1, 5)
    )

    factory = ParallelScheduleFactory(params)
    jobs = [[(0, 10, 5)], [], [(1, 10, 1)]]
    schedule = factory.from_job_tuple_list(jobs)
    return schedule

def create_schedule_empty_machine2():
    params = ParallelSchedulingParameters(
        job_count=1,
        machine_count=3,
        processing_time_domain=IntegerDomain(10, 10),
        weight_domain=IntegerDomain(1, 5)
    )

    factory = ParallelScheduleFactory(params)
    jobs = [[(0, 10, 5)], [], []]
    schedule = factory.from_job_tuple_list(jobs)
    return schedule

class TransformationTest(TestCase):
    def test_delete_neighbors(self):
        schedule = create_schedule()
        searcher = PupParallel(
            copy.deepcopy(schedule), 0.1, 0.1,
            utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_delete_neighbors(schedule)
        self.assertEqual(len(queue), 1)

        # expected_jobs = [{1: Job(1, 5, 1), 2: Job(2, 30, 1)}, {0: Job(0, 30, 5), 1: Job(1, 5, 1)}, {0: Job(0, 30, 5), 2: Job(2, 30, 1)}]
        expected_jobs = [{1: Job(1, 5, 1), 2: Job(2, 30, 1)}]
        actual_jobs = [s.jobs for s in queue]
        self.assertCountEqual(expected_jobs, actual_jobs)

        # expected_orders = [[2, 0], [1, 0], [1, 2]]
        expected_orders = [[1, 2]]
        actual_orders = [s.schedule_order for s in queue]
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(sorted(expected_orders), sorted(actual_orders))


    def test_availability_neighbors(self):
        schedule = create_schedule()
        searcher = PupParallel(copy.deepcopy(schedule), 0.1, 0.1, utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_availability_neighbors(schedule)
        self.assertEqual(len(queue), 3)

        expected_start_times = [[1, 6, 0], [0, 5, 1], [0, 6, 0]]
        actual_start_times = []
        for s in queue:
            actual_start_times.append([j.start_time for machine in s.allocation for j in machine])
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(sorted(expected_start_times), sorted(actual_start_times))

    def test_swap_neighbors(self):
        schedule = create_schedule()
        searcher = PupParallel(copy.deepcopy(schedule), 0.1, 0.1, utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_swap_neighbors(schedule)
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(len(queue), 2)

        expected_schedules = [
            [[(0, 1), (5, 0)], [(0, 2)]],
            [[(0, 0), (30, 2)], [(0, 1)]]
        ]

        actual_schedules = []
        for s in queue:
            representation = []
            for machine in s.allocation:
                representation.append([(j.start_time, j.id) for j in machine])
            actual_schedules.append(representation)

        self.assertEqual(sorted(actual_schedules), sorted(expected_schedules))

    def test_processing_time_neighbors(self):
        schedule = create_schedule()
        searcher = PupParallel(copy.deepcopy(schedule), 0.1, 0.1, utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_processing_time_neighbors(schedule, limit=2)
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(3, len(queue))

        expected_start_times = [[0, 6, 0], [0, 5, 0], [0, 5, 0]]
        actual_start_times = []
        for s in queue:
            actual_start_times.append([j.start_time for machine in s.allocation for j in machine])
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(sorted(expected_start_times), sorted(actual_start_times))

        expected_processing_times = [[6, 30, 30], [5, 29, 30], [5, 30, 29]]
        actual_processing_times = []
        for s in queue:
            actual_processing_times.append([j.processing_time for j in s.jobs.values()])
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(sorted(expected_processing_times), sorted(actual_processing_times))

    def test_move_neighbors(self):
        schedule = create_schedule()
        searcher = PupParallel(copy.deepcopy(schedule), 0.1, 0.1, utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_move_neighbors(schedule)
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(8, len(queue))

        expected_schedules = [
            [[(0, 0), (30, 1), (35, 2)], []],
            [[(0, 1), (5, 0), (35, 2)], []],
            [[(0, 1), (5, 2), (35, 0)], []],
            [[(0, 2)], [(0, 1), (5, 0)]],
            [[(0, 2)], [(0, 0), (30, 1)]],
            [[(0, 1)], [(0, 0), (30, 2)]],
            [[(0, 1)], [(0, 2), (30, 0)]],
            [[(0, 2), (30, 1)], [(0, 0)]]
        ]
        actual_schedules = []
        for s in queue:
            representation = []
            for machine in s.allocation:
                representation.append([(j.start_time, j.id) for j in machine])
            actual_schedules.append(representation)

        self.assertCountEqual(actual_schedules, expected_schedules)
        self.assertEqual(sorted(actual_schedules), sorted(expected_schedules))


    def test_move_alt_neighbors(self):
        schedule = create_schedule()
        searcher = PupParallel(copy.deepcopy(schedule), 0.1, 0.1, utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_move_neighbors_alt(schedule)
        self.assertEqual(schedule, searcher.original_schedule)
        #self.assertEqual(3, len(queue))

        expected_schedules = [
            [[(0, 1), (5, 2), (35, 0)], []],
            [[(0, 2), (30, 1)], [(0, 0)]],
            [[(0, 1)], [(0, 2), (30, 0)]]
        ]

        actual_schedules = []
        for s in queue:
            representation = []
            for machine in s.allocation:
                representation.append([(j.start_time, j.id) for j in machine])
            actual_schedules.append(representation)

        self.assertEqual(sorted(actual_schedules), sorted(expected_schedules))
        self.assertCountEqual(actual_schedules, expected_schedules)

    def test_move_alt_neighbors_empty_machine(self):
        schedule = create_schedule_empty_machine()
        searcher = PupParallel(copy.deepcopy(schedule), 0.1, 0.1,
                               utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_move_neighbors_alt(schedule)
        self.assertEqual(schedule, searcher.original_schedule)
        # self.assertEqual(3, len(queue))

        expected_schedules = [
            [[], [(0, 0)], [(0, 1)]],
            [[(0, 0)], [(0, 1)], []]
        ]

        actual_schedules = []
        for s in queue:
            representation = []
            for machine in s.allocation:
                representation.append([(j.start_time, j.id) for j in machine])
            actual_schedules.append(representation)

        self.assertEqual(sorted(actual_schedules), sorted(expected_schedules))
        self.assertCountEqual(actual_schedules, expected_schedules)

    def test_move_alt_neighbors_single_machine(self):
        schedule = create_schedule_single_machine()
        searcher = PupParallel(copy.deepcopy(schedule), 0.1, 0.1,
                               utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_move_neighbors_alt(schedule)
        self.assertEqual(schedule, searcher.original_schedule)
        # self.assertEqual(3, len(queue))
        jobs = [[(0, 5, 1), (1, 5, 1), (2, 5, 5), (3)]]
        expected_schedules = [
            [[(0, 1), (5, 0), (10, 2), (15, 3), (20, 4)]],
            [[(0, 0), (5, 2), (10, 1), (15, 3), (20, 4)]],
            [[(0, 0), (5, 1), (10, 3), (15, 2), (20, 4)]],
            [[(0, 0), (5, 1), (10, 2), (15, 4), (20, 3)]]
        ]

        actual_schedules = []
        for s in queue:
            representation = []
            for machine in s.allocation:
                representation.append([(j.start_time, j.id) for j in machine])
            actual_schedules.append(representation)

        self.assertEqual(sorted(actual_schedules), sorted(expected_schedules))
        self.assertCountEqual(actual_schedules, expected_schedules)

    def test_move_neighbors_single_machine(self):
        schedule = create_schedule_single_machine()
        searcher = PupParallel(copy.deepcopy(schedule), 0.1, 0.1,
                               utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_move_neighbors(schedule)
        self.assertEqual(schedule, searcher.original_schedule)
        # self.assertEqual(3, len(queue))
        jobs = [[(0, 5, 1), (1, 30, 1), (2, 30, 5)]]
        expected_schedules = [
            # move 0
            [[(0, 1), (5, 0), (10, 2), (15, 3), (20, 4)]],
            [[(0, 1), (5, 2), (10, 0), (15, 3), (20, 4)]],
            [[(0, 1), (5, 2), (10, 3), (15, 0), (20, 4)]],
            [[(0, 1), (5, 2), (10, 3), (15, 4), (20, 0)]],
            # move 1
            [[(0, 0), (5, 2), (10, 1), (15, 3), (20, 4)]],
            [[(0, 0), (5, 2), (10, 3), (15, 1), (20, 4)]],
            [[(0, 0), (5, 2), (10, 3), (15, 4), (20, 1)]],
            # move 2
            [[(0, 2), (5, 0), (10, 1), (15, 3), (20, 4)]],
            [[(0, 0), (5, 1), (10, 3), (15, 2), (20, 4)]],
            [[(0, 0), (5, 1), (10, 3), (15, 4), (20, 2)]],
            # move 3
            [[(0, 3), (5, 0), (10, 1), (15, 2), (20, 4)]],
            [[(0, 0), (5, 3), (10, 1), (15, 2), (20, 4)]],
            [[(0, 0), (5, 1), (10, 2), (15, 4), (20, 3)]],
            # move 4
            [[(0, 4), (5, 0), (10, 1), (15, 2), (20, 3)]],
            [[(0, 0), (5, 4), (10, 1), (15, 2), (20, 3)]],
            [[(0, 0), (5, 1), (10, 4), (15, 2), (20, 3)]],
        ]

        actual_schedules = []
        for s in queue:
            representation = []
            for machine in s.allocation:
                representation.append([(j.start_time, j.id) for j in machine])
            actual_schedules.append(representation)

        self.assertEqual(sorted(actual_schedules), sorted(expected_schedules))
        self.assertCountEqual(actual_schedules, expected_schedules)

    def test_processing_time_neighbors_edge(self):
        schedule = create_schedule()
        schedule.jobs[0].processing_time = 5
        schedule.jobs[1].processing_time = 30
        if schedule.jobs[2].processing_time == 30 or schedule.jobs[2].processing_time == 5:
            expected = 3
        else:
            expected = 4

        searcher = PupParallel(schedule, 0.1, 0.1, utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_processing_time_neighbors(schedule)
        self.assertEqual(expected, len(queue))

    def test_swapping_same_p(self):
        schedule = create_schedule_same_p()
        searcher = PupParallel(copy.deepcopy(schedule), 0.1, 0.1, utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_swap_neighbors(schedule)
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(len(queue), 2)

        expected_schedules = [
            [[(0, 1), (10, 0), (20, 2)]],
            [[(0, 0), (10, 2), (20, 1)]]
        ]

        actual_schedules = []
        for s in queue:
            representation = []
            for machine in s.allocation:
                representation.append([(j.start_time, j.id) for j in machine])
            actual_schedules.append(representation)

        self.assertEqual(sorted(actual_schedules), sorted(expected_schedules))

    def test_move_same_p(self):
        schedule = create_schedule_same_p()
        searcher = PupParallel(copy.deepcopy(schedule), 0.1, 0.1, utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_move_neighbors(schedule)
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(len(queue), 4)

        jobs = [[(0, 10, 5), (1, 10, 3), (2, 10, 1)]]
        expected_schedules = [
            [[(0, 1), (10, 0), (20, 2)]],
            [[(0, 1), (10, 2), (20, 0)]],
            [[(0, 0), (10, 2), (20, 1)]],
            [[(0, 2), (10, 0), (20, 1)]]
        ]

        actual_schedules = []
        for s in queue:
            representation = []
            for machine in s.allocation:
                representation.append([(j.start_time, j.id) for j in machine])
            actual_schedules.append(representation)

        self.assertEqual(sorted(actual_schedules), sorted(expected_schedules))

    def test_move_empty_onejob(self):
        schedule = create_schedule_empty_machine2()
        searcher = PupParallel(copy.deepcopy(schedule), 0.1, 0.1, utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_move_neighbors(schedule)
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(len(queue), 2)

        expected_schedules = [
            [[], [], [(0, 0)]],
            [[], [(0, 0)], []]]

        actual_schedules = []
        for s in queue:
            representation = []
            for machine in s.allocation:
                representation.append([(j.start_time, j.id) for j in machine])
            actual_schedules.append(representation)

        self.assertEqual(sorted(actual_schedules), sorted(expected_schedules))

    def test_swap_empty_schedule(self):
        pass

    def test_swap(self):
        #jobs = [[(0, 5, 4), (1, 5, 2), (2, 10, 5), (3, 15, 1)]]
        #schedule = [[(0,0), (5, 1), (10, 2), (20,3)]]
        schedule = create_schedule_single_machine2()
        searcher = PupParallel(copy.deepcopy(schedule), 0.1, 0.1,
                               utility_function=parallel_machine_utilities.calculate_cmax)
        s1 = searcher._swap(schedule, (0, 1), (0, 0))
        s2 = searcher._swap(schedule, (0, 0), (0, 1))
        self.assertEqual(schedule, searcher.original_schedule)

        expected_schedule = [[(0, 1), (5, 0), (10, 2), (20, 3)]]

        representation1 = []
        for machine in s1.allocation:
            representation1.append([(j.start_time, j.id) for j in machine])

        representation2 = []
        for machine in s2.allocation:
            representation2.append([(j.start_time, j.id) for j in machine])

        self.assertEqual(representation1, expected_schedule)
        self.assertEqual(representation2, expected_schedule)
        self.assertEqual(representation1, representation2)


    def test_swap_right(self):
        # jobs = [[(0, 5, 4), (1, 5, 2), (2, 10, 5), (3, 15, 1)]]
        # schedule = [[(0,0), (5, 1), (10, 2), (20,3)]]
        schedule = create_schedule_single_machine2()
        searcher = PupParallel(copy.deepcopy(schedule), 0.1, 0.1,
                               utility_function=parallel_machine_utilities.calculate_cmax)
        s = searcher._swap(schedule, (0, 0), (0, 1))
        self.assertEqual(schedule, searcher.original_schedule)

        expected_schedule = [[(0, 1), (5, 0), (10, 2), (20, 3)]]

        representation = []
        for machine in s.allocation:
            representation.append([(j.start_time, j.id) for j in machine])

        self.assertEqual(representation, expected_schedule)

    def test_swap_same_machine_ends(self):
        # jobs = [[(0, 5, 4), (1, 5, 2), (2, 10, 5), (3, 15, 1)]]
        # schedule = [[(0,0), (5, 1), (10, 2), (20,3)]]
        schedule = create_schedule_single_machine2()
        searcher = PupParallel(copy.deepcopy(schedule), 0.1, 0.1,
                               utility_function=parallel_machine_utilities.calculate_cmax)
        s = searcher._swap(schedule, (0, 0), (0, 3))
        self.assertEqual(schedule, searcher.original_schedule)

        expected_schedule = [[(0, 3), (15, 1), (20, 2), (30, 0)]]

        representation = []
        for machine in s.allocation:
            representation.append([(j.start_time, j.id) for j in machine])


        self.assertEqual(representation, expected_schedule)

    def test_swap_same_machine_middle(self):
        # jobs = [[(0, 5, 4), (1, 5, 2), (2, 10, 5), (3, 15, 1)]]
        # schedule = [[(0,0), (5, 1), (10, 2), (20,3)]]
        schedule = create_schedule_single_machine2()
        searcher = PupParallel(copy.deepcopy(schedule), 0.1, 0.1,
                               utility_function=parallel_machine_utilities.calculate_cmax)
        s = searcher._swap(schedule, (0, 1), (0, 2))
        self.assertEqual(schedule, searcher.original_schedule)

        expected_schedule = [[(0, 0), (5, 2), (15, 1), (20, 3)]]

        representation = []
        for machine in s.allocation:
            representation.append([(j.start_time, j.id) for j in machine])

        self.assertEqual(representation, expected_schedule)

    def test_swap_diff_machine(self):
        # jobs = [[(1, 5, 2), (2, 25, 1), (3, 30, 1)], [(0, 20, 5), (4, 30, 1)]]
        # schedule = [[(0, 1), (5, 2), (30, 3)], [(0, 0), (20, 4)]]
        schedule = create_schedule2()
        searcher = PupParallel(copy.deepcopy(schedule), 0.1, 0.1,
                               utility_function=parallel_machine_utilities.calculate_cmax)
        s1 = searcher._swap(schedule, (0, 1), (1, 0))
        s2 = searcher._swap(schedule, (1, 0), (0, 1))
        self.assertEqual(schedule, searcher.original_schedule)

        expected_schedule = [[(0, 1), (5, 0), (25, 3)], [(0, 2), (25, 4)]]

        representation1 = []
        for machine in s1.allocation:
            representation1.append([(j.start_time, j.id) for j in machine])

        representation2 = []
        for machine in s2.allocation:
            representation2.append([(j.start_time, j.id) for j in machine])

        self.assertEqual(representation1, expected_schedule)
        self.assertEqual(representation2, expected_schedule)
        self.assertEqual(representation1, representation2)

    def test_swap_machine_edges(self):
        # jobs = [[(1, 5, 2), (2, 25, 1), (3, 30, 1)], [(0, 20, 5), (4, 30, 1)]]
        # schedule = [[(0, 1), (5, 2), (30, 3)], [(0, 0), (20, 4)]]
        schedule = create_schedule2()
        searcher = PupParallel(copy.deepcopy(schedule), 0.1, 0.1,
                               utility_function=parallel_machine_utilities.calculate_cmax)
        s1 = searcher._swap(schedule, (0, 2), (1, 0))
        s2 = searcher._swap(schedule, (1, 0), (0, 2))
        self.assertEqual(schedule, searcher.original_schedule)

        expected_schedule = [[(0, 1), (5, 2), (30, 0)], [(0, 3), (30, 4)]]

        representation1 = []
        for machine in s1.allocation:
            representation1.append([(j.start_time, j.id) for j in machine])

        representation2 = []
        for machine in s2.allocation:
            representation2.append([(j.start_time, j.id) for j in machine])

        self.assertEqual(representation1, expected_schedule)
        self.assertEqual(representation2, expected_schedule)
        self.assertEqual(representation1, representation2)

    def test_swap_all_neighbors(self):
        schedule = create_schedule()
        searcher = PupParallel(copy.deepcopy(schedule), 0.1, 0.1, utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_swap_all(schedule)
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(len(queue), 3)

        expected_schedules = [
            [[(0, 2), (30, 1)], [(0, 0)]],
            [[(0, 0), (30, 2)], [(0, 1)]],
            [[(0, 1), (5, 0)], [(0, 2)]]
        ]

        actual_schedules = []
        for s in queue:
            representation = []
            for machine in s.allocation:
                representation.append([(j.start_time, j.id) for j in machine])
            actual_schedules.append(representation)

        self.assertEqual(sorted(actual_schedules), sorted(expected_schedules))

    def test_swap_all_neighbors_empty(self):
        #jobs = [[(0, 10, 5)], [], [(1, 10, 1)]]
        schedule = create_schedule_empty_machine()
        searcher = PupParallel(copy.deepcopy(schedule), 0.1, 0.1, utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_swap_all(schedule)
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(len(queue), 1)

        expected_schedules = [
            [[(0, 1)], [], [(0, 0)]]
        ]

        actual_schedules = []
        for s in queue:
            representation = []
            for machine in s.allocation:
                representation.append([(j.start_time, j.id) for j in machine])
            actual_schedules.append(representation)

        self.assertEqual(sorted(actual_schedules), sorted(expected_schedules))

    def test_swap_all_neighbors_empty2(self):
        #jobs = [[(0, 10, 5)], [], [(1, 10, 1)]]
        schedule = create_schedule_empty_machine2()
        searcher = PupParallel(copy.deepcopy(schedule), 0.1, 0.1, utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_swap_all(schedule)
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(len(queue), 0)

        expected_schedules = [
        ]

        actual_schedules = []
        for s in queue:
            representation = []
            for machine in s.allocation:
                representation.append([(j.start_time, j.id) for j in machine])
            actual_schedules.append(representation)

        self.assertEqual(sorted(actual_schedules), sorted(expected_schedules))
