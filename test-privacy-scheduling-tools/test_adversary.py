from unittest import TestCase

from privacyschedulingtools.total_weighted_completion_time.entity.adversary.adversary import RandomGuessingAdversary
from privacyschedulingtools.total_weighted_completion_time.entity.domain import IntegerDomain
from privacyschedulingtools.total_weighted_completion_time.entity.job import Job
from privacyschedulingtools.total_weighted_completion_time.entity.schedule import Schedule


class TestRandomGuessingAdversary(TestCase):

    def test_execute_attack(self):
        pt_domain = IntegerDomain(1,4)
        w_domain = IntegerDomain(1,4)

        adversary = RandomGuessingAdversary(pt_domain, w_domain, min_solution_count=3, max_solution_count=5)

        schedule = Schedule(None, None, pt_domain, w_domain)

        schedule.jobs = {0: Job(0, 1, 1),
                         1: Job(1, 2, 2),
                         2: Job(2, 4, 4)}
        schedule.schedule_order = [0, 1, 2]

        solution = adversary.execute_attack(schedule)
        candidates = solution["solutions"]


        self.assertIn(solution["solution_count"], range(3,6))
        self.assertEqual(solution["solution_count"], len(candidates))

        for c in candidates:
            self.assertEqual(3, len(c))
            for weight in c.pl_avg_dict():
                self.assertIn(weight, range(1, 5))
