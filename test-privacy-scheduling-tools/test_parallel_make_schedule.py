from copy import deepcopy
from unittest import TestCase

from privacyschedulingtools.total_weighted_completion_time.entity.domain import IntegerDomain
from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import ParallelScheduleFactory, \
    ParallelSchedule
from privacyschedulingtools.total_weighted_completion_time.entity.scheduling_parameters import \
    ParallelSchedulingParameters


class ParallelScheduleTest(TestCase):
    def setUp(self) -> None:
        params = ParallelSchedulingParameters(
            job_count=3,
            machine_count=2,
            processing_time_domain=IntegerDomain(5, 30),
            weight_domain=IntegerDomain(1, 5)
        )
        self.factory = ParallelScheduleFactory(params)

    def create_schedule(self) -> ParallelSchedule:
        jobs = [[(1, 5, 1), (2, 30, 1)], [(0, 30, 5)]]
        schedule = self.factory.from_job_tuple_list(jobs)
        return schedule

    def test_compare_schedule_with_diff_weights(self):
        schedule = self.create_schedule()
        copied_schedule = deepcopy(schedule)
        copied_schedule.jobs[0].weight = 3

        self.assertFalse(schedule.__eq__(copied_schedule))
        self.assertTrue(schedule.compare_published_version(copied_schedule))

    def test_compare_schedule_with_diff_processing_times(self):
        schedule = self.create_schedule()
        copied_schedule = deepcopy(schedule)
        copied_schedule.jobs[0].processing_time = 15

        self.assertFalse(schedule.__eq__(copied_schedule))
        self.assertFalse(schedule.compare_published_version(copied_schedule))

    def test_invalid_schedule(self):
        jobs = [[(1, 6, 1), (2, 30, 1)], [(0, 5, 5)]]
        schedule = self.factory.from_job_tuple_list(jobs)

        self.assertFalse(ParallelSchedule.is_valid_twct_schedule(schedule))

    def test_invalid_schedule_empty_machine(self):
        jobs = [[(1, 6, 1), (2, 30, 1)], []]
        schedule = self.factory.from_job_tuple_list(jobs)

        self.assertFalse(ParallelSchedule.is_valid_make_schedule(schedule))

    def test_valid_schedule_equal_c(self):
        jobs = [[(1, 5, 1), (2, 30, 1)], [(0, 5, 5)]]
        schedule = self.factory.from_job_tuple_list(jobs)

        self.assertTrue(ParallelSchedule.is_valid_twct_schedule(schedule))

    def test_valid_schedule_unequal_c(self):
        jobs = [[(1, 4, 1), (2, 30, 1)], [(0, 5, 5)]]
        schedule = self.factory.from_job_tuple_list(jobs)

        self.assertTrue(ParallelSchedule.is_valid_twct_schedule(schedule))

    def test_schedule_generation_by_weights(self):
        jobs = [[(1, 4, 1), (2, 30, 1)], [(0, 5, 5)]]
        schedule = self.factory.from_job_tuple_list(jobs)
        weights = {0: 1, 1: 2, 2: 2}
        schedule_with_diff_w = self.factory.generate_schedule_with_weights(schedule, weights, "wspt")

        self.assertTrue(schedule.compare_published_version(schedule_with_diff_w))
        self.assertFalse(schedule is schedule_with_diff_w)

    def test_schedule_generation_by_weights2(self):
        jobs = [[(1, 5, 5), (2, 5, 5)], [(0, 5, 5)]]
        schedule = self.factory.from_job_tuple_list(jobs)
        weights = {0: 3, 1: 2, 2: 5}
        schedule_with_diff_w = self.factory.generate_schedule_with_weights(schedule, weights, "wspt")

        self.assertFalse(schedule.compare_published_version(schedule_with_diff_w))
        self.assertFalse(schedule is schedule_with_diff_w)

    def test_schedule_single_machine(self):
        params = ParallelSchedulingParameters(
            job_count=3,
            machine_count=1,
            processing_time_domain=IntegerDomain(5, 30),
            weight_domain=IntegerDomain(1, 10)
        )
        factory = ParallelScheduleFactory(params)
        schedule = factory.generate_random_schedule_with_dispatching_rule(schedule_type="make")

        self.assertEquals(3, len(schedule.jobs))
        self.assertEquals(1, len(schedule.allocation))

    def test_schedule_multiple_machine(self):
        params = ParallelSchedulingParameters(
            job_count=10,
            machine_count=4,
            processing_time_domain=IntegerDomain(5, 30),
            weight_domain=IntegerDomain(1, 40)
        )
        factory = ParallelScheduleFactory(params)
        schedule = factory.generate_random_schedule_with_dispatching_rule(schedule_type="make")

        self.assertEquals(4, len(schedule.jobs))
        self.assertEquals(10, len(schedule.allocation))


