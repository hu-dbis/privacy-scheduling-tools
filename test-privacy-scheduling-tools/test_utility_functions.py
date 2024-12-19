from unittest import TestCase

from privacyschedulingtools.total_weighted_completion_time.entity.domain import IntegerDomain
from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import ParallelScheduleFactory
from privacyschedulingtools.total_weighted_completion_time.entity.scheduling_parameters import \
    ParallelSchedulingParameters
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

def create_empty_schedule():
    params = ParallelSchedulingParameters(
        job_count=3,
        machine_count=2,
        processing_time_domain=IntegerDomain(5, 30),
        weight_domain=IntegerDomain(1, 5)
    )
    factory = ParallelScheduleFactory(params)
    jobs = [[], []]
    schedule = factory.from_job_tuple_list(jobs)
    return schedule

def create_empty_machine():
    params = ParallelSchedulingParameters(
        job_count=3,
        machine_count=2,
        processing_time_domain=IntegerDomain(5, 30),
        weight_domain=IntegerDomain(1, 5)
    )
    factory = ParallelScheduleFactory(params)
    jobs = [[(1, 5, 1), (2, 30, 1)], []]
    schedule = factory.from_job_tuple_list(jobs)
    return schedule


schedule = create_schedule()
empty_schedule = create_empty_schedule()
empty_machine = create_empty_machine()

class UtilityTest(TestCase):
    def testCmax(self):
        actual_u = parallel_machine_utilities.calculate_cmax(schedule)
        expected_u = 35

        self.assertEqual(expected_u, actual_u)

    def testAvgWait(self):
        actual_u = parallel_machine_utilities.calculate_avg_wait_time(schedule)
        expected_u = 5 / 3

        self.assertEqual(expected_u, actual_u)

    def testTwct(self):
        actual_u = parallel_machine_utilities.calculate_twct(schedule)
        expected_u = 190

        self.assertEqual(expected_u, actual_u)

    def testTct(self):
        actual_u = parallel_machine_utilities.calculate_tct(schedule)
        expected_u = 70

        self.assertEqual(expected_u, actual_u)

    def testCmaxEmpty(self):
        actual_u = parallel_machine_utilities.calculate_cmax(empty_schedule)
        expected_u = 0

        self.assertEqual(expected_u, actual_u)

    def testAvgWaitEmpty(self):
        actual_u = parallel_machine_utilities.calculate_avg_wait_time(empty_schedule)
        expected_u = 0

        self.assertEqual(expected_u, actual_u)

    def testTwctEmpty(self):
        actual_u = parallel_machine_utilities.calculate_twct(empty_schedule)
        expected_u = 0

        self.assertEqual(expected_u, actual_u)

    def testTctEmpty(self):
        actual_u = parallel_machine_utilities.calculate_tct(empty_schedule)
        expected_u = 0

        self.assertEqual(expected_u, actual_u)

    def testCmaxMachineEmpty(self):
        actual_u = parallel_machine_utilities.calculate_cmax(empty_machine)
        expected_u = 35

        self.assertEqual(expected_u, actual_u)

    def testAvgWaitMachineEmpty(self):
        actual_u = parallel_machine_utilities.calculate_avg_wait_time(empty_machine)
        expected_u = 5 / 2

        self.assertEqual(expected_u, actual_u)

    def testTwctMachineEmpty(self):
        actual_u = parallel_machine_utilities.calculate_twct(empty_machine)
        expected_u = 40

        self.assertEqual(expected_u, actual_u)

    def testTctMachineEmpty(self):
        actual_u = parallel_machine_utilities.calculate_tct(empty_machine)
        expected_u = 40

        self.assertEqual(expected_u, actual_u)
