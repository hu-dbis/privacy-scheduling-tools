from unittest import TestCase

from privacyschedulingtools.total_weighted_completion_time.entity.domain import IntegerDomain
from privacyschedulingtools.total_weighted_completion_time.entity.job import Job
from privacyschedulingtools.total_weighted_completion_time.entity.schedule import Schedule
from privacyschedulingtools.total_weighted_completion_time.pup import privacy_loss


class TestPrivacyLoss(TestCase):

    def test_distance(self):
        values = [1, 2, 3, 4, 5]
        self.assertEqual(2, privacy_loss.distance(1, values))

    def test_distance_all_values_equal(self):
        values = [3, 3, 3]
        self.assertEqual(0, privacy_loss.distance(3, values))
        self.assertEqual(5, privacy_loss.distance(-2, values))

    def test_distance_float(self):
        values = [3, 3, 6, 6]
        self.assertEqual(1.5, privacy_loss.distance(4, values))


    def test_pl_distance_based_per_job(self):
        schedule = Schedule(None, None, IntegerDomain(1, 5), IntegerDomain(1,5))
        schedule.jobs = {0: Job(0, 1, 4)}
        schedule.schedule_order = [0]

        candidates = [{0: w} for w in [5, 5, 4, 1, 5, 2, 1, 3, 2, 5]]

        pl_result = privacy_loss.distance_based_per_job(schedule, candidates)

        self.assertAlmostEqual(1-(15/14), pl_result[0], places=2)



