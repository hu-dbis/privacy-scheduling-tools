from unittest import TestCase

from privacyschedulingtools.total_weighted_completion_time.entity.domain import Domain, IntegerDomain
from privacyschedulingtools.total_weighted_completion_time.entity.job import Job
from privacyschedulingtools.total_weighted_completion_time.entity.schedule import Schedule, ScheduleFactory
from privacyschedulingtools.total_weighted_completion_time.entity.scheduling_parameters import SchedulingParameters


class TestSchedule(TestCase):

    test_params = SchedulingParameters(3, IntegerDomain(1, 4), IntegerDomain(1,4))

    def test_schedule_str(self):
        schedule = Schedule(None, None, self.test_params)
        schedule.jobs = {0: Job(0, 1, 1),
                         1: Job(1, 2, 2),
                         2: Job(2, 4, 4)}
        schedule.schedule_order = [0, 2, 1]

        expected_str = "(0, 1, 1), (2, 4, 4), (1, 2, 2)"
        self.assertEqual(expected_str, schedule.__str__())

    def test_weight_vector_equals(self):
        schedule = Schedule(None, None, self.test_params)
        schedule.jobs = {0: Job(0, 1, 1),
                         1: Job(1, 2, 2),
                         2: Job(2, 4, 4)}
        schedule.schedule_order = [0, 1, 2]
        weight_vector_1 = {0: 1, 1: 2, 2:4}
        weight_vector_2 = {0: 1, 1: 2, 2:1}
        self.assertTrue(schedule.weight_vector_equals(weight_vector_1))
        self.assertFalse(schedule.weight_vector_equals(weight_vector_2))

    def test_hash_and_equals(self):
        schedule1 = Schedule(None, None, self.test_params)
        schedule1.jobs = {0: Job(0, 1, 1),
                         1: Job(1, 2, 2),
                         2: Job(2, 4, 4)}
        schedule1.schedule_order = [0, 1, 2]

        schedule2 = Schedule(None, None, self.test_params)
        schedule2.jobs = {0: Job(0, 1, 1),
                         1: Job(1, 2, 2),
                         2: Job(2, 4, 4)}
        schedule2.schedule_order = [0, 1, 2]

        test_dict = {schedule1 : 123 }
        self.assertEqual(123, test_dict.get(schedule1))
        self.assertEqual(123, test_dict.get(schedule2))

        test_dict[schedule2] = 456
        self.assertEqual(456, test_dict.get(schedule1))
        self.assertEqual(456, test_dict.get(schedule2))


class TestScheduleFactory(TestCase):

    def setUp(self):
        self.factory = ScheduleFactory(3, processing_time_domain=MockDomain([3, 2, 1]),
                                       weight_domain=MockDomain([1, 2, 3]))
        print(str(self.factory))

    def test_generate_random_schedule(self):
        schedule = self.factory.generate_random_schedule()
        self.assertEqual([0, 1, 2], schedule.schedule_order)
        self.assertEqual(Job(0, 3, 1), schedule.jobs[0])
        self.assertEqual(Job(1, 2, 2), schedule.jobs[1])
        self.assertEqual(Job(2, 1, 3), schedule.jobs[2])

    def test_generate_random_optimized_schedule(self):
        schedule = self.factory.generate_random_optimized_schedule()

        # this generation method actually changes the job  ids
        # so the ids in schedule order should be sorted
        self.assertEqual([0, 1, 2], schedule.schedule_order)

        # but the ids of the individual schedules should have changed
        self.assertEqual(Job(2, 3, 1), schedule.jobs[2])
        self.assertEqual(Job(1, 2, 2), schedule.jobs[1])
        self.assertEqual(Job(0, 1, 3), schedule.jobs[0])


class MockDomain(Domain):
    def __init__(self, values):
        self.values = values
        self.counter = -1

    def get_random(self):
        self.counter += 1
        return self.values[self.counter]

