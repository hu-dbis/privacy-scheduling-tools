import copy
from unittest import TestCase

from privacyschedulingtools.total_weighted_completion_time.entity.domain import IntegerDomain
from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import ParallelSchedule, \
    ParallelScheduleFactory
from privacyschedulingtools.total_weighted_completion_time.entity.scheduling_parameters import \
    ParallelSchedulingParameters
from privacyschedulingtools.total_weighted_completion_time.pup.solver.pup_parallel_make import PupParallelMake
from privacyschedulingtools.total_weighted_completion_time.pup.utility_functions import parallel_machine_utilities


def create_schedule():
    params = ParallelSchedulingParameters(
        job_count=3,
        machine_count=2,
        processing_time_domain=IntegerDomain(5, 30),
        weight_domain=IntegerDomain(1, 40)
    )
    factory = ParallelScheduleFactory(params)
    jobs = [[(0, 15, 0), (3, 10, 0)], [(1, 5, 10), (2, 10, 20)]]
    schedule = factory.from_job_tuple_list(jobs, schedule_type="make")
    return schedule

def create_schedule2():
    params = ParallelSchedulingParameters(
        job_count=5,
        machine_count=2,
        processing_time_domain=IntegerDomain(5, 30),
        weight_domain=IntegerDomain(0, 40)
    )
    factory = ParallelScheduleFactory(params)
    jobs = [[(0, 20, 0), (3, 15, 22), (4, 5, 40)], [(1, 15, 0), (2, 10, 10)]]
    schedule = factory.from_job_tuple_list(jobs, schedule_type="make")
    return schedule

def create_schedule_single_machine():
    params = ParallelSchedulingParameters(
        job_count=5,
        machine_count=1,
        processing_time_domain=IntegerDomain(5, 30),
        weight_domain=IntegerDomain(0, 40)
    )
    factory = ParallelScheduleFactory(params)
    jobs = [[(0, 10, 0), (1, 5, 0), (2, 30, 40)]]
    schedule = factory.from_job_tuple_list(jobs, schedule_type="make")
    return schedule

def create_schedule_single_machine2():
    params = ParallelSchedulingParameters(
        job_count=4,
        machine_count=1,
        processing_time_domain=IntegerDomain(5, 30),
        weight_domain=IntegerDomain(1, 5)
    )

    # jobs = [[(2, 10, 0), (0, 6, 5), (1, 5, 0), , (3, 15, 20)]]
    # schedule = [[(0,2), (10, 0), (16, 1), (21,3)]]

    factory = ParallelScheduleFactory(params)
    jobs = [[(2, 10, 0), (0, 6, 10), (1, 5, 0), (3, 30, 20)]]
    schedule = factory.from_job_tuple_list(jobs, schedule_type="make")
    return schedule

def create_schedule_single_machine3():
    params = ParallelSchedulingParameters(
        job_count=4,
        machine_count=1,
        processing_time_domain=IntegerDomain(5, 30),
        weight_domain=IntegerDomain(1, 5)
    )

    factory = ParallelScheduleFactory(params)
    jobs = [[(2, 10, 0), (0, 6, 10), (1, 5, 16), (3, 30, 20)]]
    # schedule = [[(0, 2), (10, 0), (1, 16), (3, 21)]
    schedule = factory.from_job_tuple_list(jobs, schedule_type="make")
    return schedule

def create_schedule_same_p():
    params = ParallelSchedulingParameters(
        job_count=3,
        machine_count=1,
        processing_time_domain=IntegerDomain(10, 10),
        weight_domain=IntegerDomain(0, 5)
    )

    factory = ParallelScheduleFactory(params)
    jobs = [[(0, 10, 0), (1, 10, 0), (2, 10, 0)]]
    schedule = factory.from_job_tuple_list(jobs, schedule_type="make")
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
    schedule = factory.from_job_tuple_list(jobs, schedule_type="make")
    return schedule

def create_schedule_empty_machine2():
    params = ParallelSchedulingParameters(
        job_count=1,
        machine_count=3,
        processing_time_domain=IntegerDomain(10, 10),
        weight_domain=IntegerDomain(0, 5)
    )

    factory = ParallelScheduleFactory(params)
    jobs = [[(0, 10, 5)], [], []]
    schedule = factory.from_job_tuple_list(jobs, schedule_type="make")
    return schedule

class TransformationTest(TestCase):
    def test_processing_time_neighbors(self):
        # DONE
        # jobs = [[(0, 15, 0), (3,10,0)], [(1, 5, 10), (2, 10, 20)]]
        # schedule = [[(0,0)],[(10,1),[(20,2)]]
        schedule = create_schedule()
        searcher = PupParallelMake(copy.deepcopy(schedule), 0.1, 0.1, utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_processing_time_neighbors(schedule, limit=2)
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(7, len(queue))

        expected_start_times = [[0, 14, 10, 20], [0, 16, 10, 20], # job 0 +/- 1
                                [0, 15, 10, 20], [0, 15, 10, 20], # job 3 +/- 1
                                [0, 15, 10, 20],  # job 1 +/- 1
                                [0, 15, 10, 20], [0, 15, 10, 20] # job 2 +/- 1
                                ]
        actual_start_times = []
        for s in queue:
            actual_start_times.append([j.start_time for machine in s.allocation for j in machine])
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(sorted(expected_start_times), sorted(actual_start_times))

        expected_processing_times = [
                                [14, 10, 5, 10], [16, 10, 5, 10], # job 0 +/- 1
                                [15, 9, 5, 10], [15, 11, 5, 10], # job 3 +/- 1
                                [15, 10, 6, 10], # job 1 +/- 1
                                [15, 10, 5, 11], [15, 10, 5, 9]
        ]
        actual_processing_times = []
        for s in queue:
            actual_processing_times.append([j.processing_time for j in s.jobs.values()])
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(sorted(expected_processing_times), sorted(actual_processing_times))

    def test_processing_time_neighbors_single_schedule(self):
        #DONE

        # jobs = [[(2, 10, 0), (0, 6, 10), (1, 5, 0), (3, 30, 20)]]
        # schedule = [[(0,2), (10, 0), (16, 1), (21,3)]]
        schedule = create_schedule_single_machine2()
        searcher = PupParallelMake(copy.deepcopy(schedule), 0.1, 0.1, utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_processing_time_neighbors(schedule, limit=2)
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(6, len(queue))

        expected_start_times = [[0, 11, 17, 22], [0, 10, 16, 21],   # job 2 +/- 1
                                [0, 10, 15, 20], [0, 10, 17, 22],   # job 0 +/- 1
                                [0, 10, 16, 22],                    # job 1 +1, -1 not allowed
                                [0, 10, 16, 21]                     # job 3 -1, +1 not allowed
                                ]
        actual_start_times = []
        for s in queue:
            actual_start_times.append([j.start_time for machine in s.allocation for j in machine])
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(sorted(expected_start_times), sorted(actual_start_times))

        expected_processing_times = [[9,6,5,30], [11,6,5,30], [10,5,5,30], [10,7,5,30], [10,6,6,30], [10,6,5,29]]
        actual_processing_times = []
        for s in queue:
            actual_processing_times.append([j.processing_time for j in s.jobs.values()])
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(sorted(expected_processing_times), sorted(actual_processing_times))

    def test_move_neighbors(self):
        #DONE
        # jobs = [[(0, 15, 0), (3,10,0)], [(1, 5, 10), (2, 10, 20)]]
        # schedule = [[(0,0), (15, 3)],[(10,1),[(20,2)]]
        schedule = create_schedule()
        searcher = PupParallelMake(schedule, 0.1, 0.1, utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_move_neighbors(schedule)
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(14, len(queue))

        expected_schedules = [
            [[(0, 3), (10, 0)], [(10, 1),(20, 2)]],
            [[(0,3)], [(0, 0), (15, 1), (20, 2)]],
            [[(0,3)], [(10, 1), (15, 0), (30, 2)]],
            [[(0,3)], [(10, 1), (20, 2), (30,0)]],
            [[(0, 0)], [(0, 3), (10, 1), (20, 2)]],
            [[(0, 0)], [(10, 1), (15, 3), (25, 2)]],
            [[(0, 0)], [(10, 1), (20, 2), (30, 3)]],
            [[(0, 0), (15, 3)], [(20, 2), (30, 1)]],
            [[(10, 1), (15, 0), (30, 3)], [(20, 2)]],
            [[(0, 0), (15, 1), (20, 3)], [(20, 2)]],
            [[(0, 0), (15, 3), (25, 1)], [(20, 2)]],
            [[(20, 2), (30, 0), (45, 3)], [(10, 1)]],
            [[(0, 0), (20, 2), (30, 3)], [(10, 1)]],
            [[(0, 0), (15, 3), (25, 2)], [(10, 1)]]
        ]
        actual_schedules = []
        for s in queue:
            representation = []
            for machine in s.allocation:
                representation.append([(j.start_time, j.id) for j in machine])
            actual_schedules.append(representation)

        self.assertCountEqual(actual_schedules, expected_schedules)
        self.assertEqual(sorted(actual_schedules), sorted(expected_schedules))

    def test_move_neighbors_single_machine(self):
        #DONE
        # jobs = [[(2, 10, 0), (0, 6, 5), (1, 5, 0), , (3, 15, 20)]]
        # schedule = [[(0,2), (10, 0), (16, 1), (21,3)]]
        schedule = create_schedule_single_machine2()
        searcher = PupParallelMake(schedule, 0.1, 0.1, utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_move_neighbors(schedule)
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(9, len(queue))

        expected_schedules = [
            [[(10, 0), (16, 2), (26, 1), (31, 3)]],
            [[(10, 0), (16, 1), (21, 2), (31, 3)]],
            [[(10, 0), (16, 1), (21, 3), (51, 2)]],
            [[(0, 2), (10, 1), (15, 0), (21, 3)]],
            [[(0, 2), (10, 1), (20, 3), (50, 0)]],
            [[(0, 2), (20, 3), (50, 0), (56, 1)]],
            [[(0, 1), (5, 2), (15, 0), (21, 3)]],
            [[(0, 2), (10, 0), (20, 3), (50, 1)]],
            [[(20, 3), (50, 2), (60, 0), (66, 1)]]
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
        #TODO
        schedule = create_schedule()
        schedule.jobs[0].processing_time = 5
        schedule.jobs[1].processing_time = 30
        if schedule.jobs[2].processing_time == 30 or schedule.jobs[2].processing_time == 5:
            expected = 3
        else:
            expected = 4

        searcher = PupParallelMake(schedule, 0.1, 0.1, utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_processing_time_neighbors(schedule)
        self.assertEqual(expected, len(queue))

    def test_move_same_p(self):
        #DONE
        schedule = create_schedule_same_p()
        searcher = PupParallelMake(copy.deepcopy(schedule), 0.1, 0.1, utility_function=parallel_machine_utilities.calculate_cmax)
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
        # DONE
        schedule = create_schedule_empty_machine2()
        searcher = PupParallelMake(copy.deepcopy(schedule), 0.1, 0.1, utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_move_neighbors(schedule)
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(len(queue), 2)

        expected_schedules = [
            [[], [], [(5, 0)]],
            [[], [(5, 0)], []]]

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
        #DONE
        # jobs = [[(0, 6, 5), (1, 5, 0), (2, 10, 0), (3, 30, 20)]]
        # schedule = [[(0,2), (10, 0), (16, 1), (21,3)]]
        schedule = create_schedule_single_machine2()
        searcher = PupParallelMake(copy.deepcopy(schedule), 0.1, 0.1,
                                   utility_function=parallel_machine_utilities.calculate_cmax)
        s1 = searcher._swap(schedule, (0, 2), (0, 3))
        s2 = searcher._swap(schedule, (0, 3), (0, 2))
        self.assertEqual(schedule, searcher.original_schedule)

        expected_schedule = [[(0,2), (10, 0), (20, 3), (50,1)]]

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
        #DONE
        # jobs = [[(0, 6, 5), (1, 5, 0), (2, 10, 0), (3, 30, 20)]]
        # schedule = [[(0,2), (10, 0), (16, 1), (21,3)]]
        schedule = create_schedule_single_machine2()
        searcher = PupParallelMake(copy.deepcopy(schedule), 0.1, 0.1,
                                   utility_function=parallel_machine_utilities.calculate_cmax)
        s = searcher._swap(schedule, (0, 2), (0, 3))
        self.assertEqual(schedule, searcher.original_schedule)

        expected_schedule = [[(0,2), (10, 0), (20, 3), (50,1)]]

        representation = []
        for machine in s.allocation:
            representation.append([(j.start_time, j.id) for j in machine])

        self.assertEqual(representation, expected_schedule)

    def test_swap_same_machine_ends(self):
        #DONE
        # jobs = [[(0, 6, 5), (1, 5, 0), (2, 10, 0), (3, 30, 20)]]
        # schedule = [[(0,2), (10, 0), (16, 1), (21,3)]]
        schedule = create_schedule_single_machine2()
        searcher = PupParallelMake(copy.deepcopy(schedule), 0.1, 0.1,
                                   utility_function=parallel_machine_utilities.calculate_cmax)
        s = searcher._swap(schedule, (0, 0), (0, 3))
        self.assertEqual(schedule, searcher.original_schedule)

        expected_schedule = [[(20, 3), (50, 0), (56, 1), (61, 2)]]

        representation = []
        for machine in s.allocation:
            representation.append([(j.start_time, j.id) for j in machine])


        self.assertEqual(representation, expected_schedule)

    def test_swap_same_machine_middle(self):
        #DONE
        # jobs = [[(0, 5, 4), (1, 5, 2), (2, 10, 5), (3, 15, 1)]]
        # schedule = [[(0,0), (5, 1), (10, 2), (20,3)]]
        schedule = create_schedule_single_machine2()
        searcher = PupParallelMake(copy.deepcopy(schedule), 0.1, 0.1,
                                   utility_function=parallel_machine_utilities.calculate_cmax)
        s = searcher._swap(schedule, (0, 1), (0, 2))
        self.assertEqual(schedule, searcher.original_schedule)

        expected_schedule = [[(0, 2), (10, 1), (15, 0), (21, 3)]]

        representation = []
        for machine in s.allocation:
            representation.append([(j.start_time, j.id) for j in machine])

        self.assertEqual(representation, expected_schedule)

    def test_swap_diff_machine(self):
        # DONE
        # jobs = [[(0, 20, 0), (3, 15, 22), (4, 5, 40)], [(1, 15, 0), (2, 10, 10)]]
        # schedule = [[(0, 0), (22, 3), (40, 4)], [(0, 1), (15, 2)]]
        schedule = create_schedule2()
        print(schedule)
        searcher = PupParallelMake(copy.deepcopy(schedule), 0.1, 0.1,
                                   utility_function=parallel_machine_utilities.calculate_cmax)
        s1 = searcher._swap(schedule, (0, 1), (1, 0))
        s2 = searcher._swap(schedule, (1, 0), (0, 1))
        self.assertEqual(schedule, searcher.original_schedule)

        expected_schedule = [[(0, 0), (20, 1), (40, 4)], [(22, 3), (37, 2)]]

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
        #DONE
        # jobs = [[(0, 20, 0), (3, 15, 22), (4, 5, 40)], [(1, 15, 0), (2, 10, 10)]]
        # schedule = [[(0, 0), (22, 3), (40, 4)], [(0, 1), (15, 2)]]
        schedule = create_schedule2()
        searcher = PupParallelMake(copy.deepcopy(schedule), 0.1, 0.1,
                                   utility_function=parallel_machine_utilities.calculate_cmax)
        s1 = searcher._swap(schedule, (0, 2), (1, 0))
        s2 = searcher._swap(schedule, (1, 0), (0, 2))
        self.assertEqual(schedule, searcher.original_schedule)

        expected_schedule = [[(0, 0), (22, 3), (37, 1)], [(40, 4), (45, 2)]]

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
        #TODO
        schedule = create_schedule()
        searcher = PupParallelMake(copy.deepcopy(schedule), 0.1, 0.1, utility_function=parallel_machine_utilities.calculate_cmax)
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
        # DONE
        #jobs = [[(0, 10, 5)], [], [(1, 10, 1)]]
        schedule = create_schedule_empty_machine()
        searcher = PupParallelMake(copy.deepcopy(schedule), 0.1, 0.1, utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_swap_all(schedule)
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(len(queue), 1)

        expected_schedules = [
            [[(1, 1)], [], [(5, 0)]]
        ]

        actual_schedules = []
        for s in queue:
            representation = []
            for machine in s.allocation:
                representation.append([(j.start_time, j.id) for j in machine])
            actual_schedules.append(representation)

        self.assertEqual(sorted(actual_schedules), sorted(expected_schedules))

    def test_swap_all_neighbors_empty2(self):
        # DONE
        schedule = create_schedule_empty_machine2()
        searcher = PupParallelMake(copy.deepcopy(schedule), 0.1, 0.1, utility_function=parallel_machine_utilities.calculate_cmax)
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

    def test_release_time_neighbors(self):
        #TODO
        # jobs = [[(0, 15, 0), (3,10,0)], [(1, 5, 10), (2, 10, 20)]]
        # schedule = [[(0,0), (15, 3)],[(10,1),[(20,2)]]
        schedule = create_schedule()
        searcher = PupParallelMake(copy.deepcopy(schedule), 0.1, 0.1, utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_release_time_neighbors(schedule)
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(len(queue), 6)

        expected_schedules = [
            [[(1, 0), (16, 3)], [(10, 1), (20, 2)]],  # job 0 +1
            [[(0, 0), (15, 3)], [(10, 1), (20, 2)]],  # job 3 +1
            [[(0, 0), (15, 3)], [(9, 1), (20, 2)]],  # job 1 -1
            [[(0, 0), (15, 3)], [(11, 1), (20, 2)]],  # job 1 +1
            [[(0, 0), (15, 3)], [(10, 1), (21, 2)]], # job 2 +1
            [[(0, 0), (15, 3)], [(10, 1), (19, 2)]], # job 2 -1
        ]

        actual_schedules = []
        for s in queue:
            representation = []
            for machine in s.allocation:
                representation.append([(j.start_time, j.id) for j in machine])
            actual_schedules.append(representation)

        self.assertEqual(sorted(actual_schedules), sorted(expected_schedules))


    def test_release_time_neighbors_single_machine(self):
        #TODO
        # jobs = [[(2, 10, 0), (0, 6, 10), (1, 5, 16), (3, 30, 20)]]
        # schedule = [[(0, 2), (10, 0), (1, 16), (3, 21)]
        schedule = create_schedule_single_machine3()
        searcher = PupParallelMake(copy.deepcopy(schedule), 0.1, 0.1, utility_function=parallel_machine_utilities.calculate_cmax)
        queue = searcher._add_release_time_neighbors(schedule)
        self.assertEqual(schedule, searcher.original_schedule)
        self.assertEqual(len(queue), 7)

        expected_schedules = [
            [[(1, 2), (11, 0), (17, 1), (22, 3)]],  # job 2 +1
            [[(0, 2), (10, 0), (16, 1), (21, 3)]],  # job 0 -1
            [[(0, 2), (11, 0), (17, 1), (22, 3)]],  # job 0 +1
            [[(0, 2), (10, 0), (16, 1), (21, 3)]],  # job 1 -1
            [[(0, 2), (10, 0), (17, 1), (22, 3)]],  # job 1 +1
            [[(0, 2), (10, 0), (16, 1), (21, 3)]],  # job 1 +1
            [[(0, 2), (10, 0), (16, 1), (21, 3)]],  # job 1 -1
        ]

        actual_schedules = []
        for s in queue:
            representation = []
            for machine in s.allocation:
                representation.append([(j.start_time, j.id) for j in machine])
            actual_schedules.append(representation)

        self.assertEqual(sorted(actual_schedules), sorted(expected_schedules))
