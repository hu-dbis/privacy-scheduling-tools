from unittest import TestCase

from privacyschedulingtools.total_weighted_completion_time.entity.domain import IntegerDomain, FloatDomain


class TestIntegerDomain(TestCase):

    def setUp(self):
        self.integer_domain = IntegerDomain(1, 10)

    def test_get_min(self):
        self.assertEqual(1, self.integer_domain.get_min())

    def test_get_max(self):
        self.assertEqual(10, self.integer_domain.get_max())

    def test_get_random(self):
        for _ in range(10):
            self.assertIn(self.integer_domain.get_random(), range(1, 10+1))

    def test_expected_distance(self):
        self.assertAlmostEqual(3.1, self.integer_domain.expected_distance(3))

class TestFloatDomain(TestCase):

    def setUp(self):
        self.float_domain = FloatDomain(0, 6)

    def test_get_min(self):
        self.assertEqual(0, self.float_domain.get_min())

    def test_get_max(self):
        self.assertEqual(6, self.float_domain.get_max())

    def test_get_random(self):
        for _ in range(10):
            random_from_domain = self.float_domain.get_random()
            self.assertTrue(random_from_domain < 6)
            self.assertTrue(random_from_domain > 0)

    def test_expected_distance(self):
        self.assertAlmostEqual(1.5, self.float_domain.expected_distance(3))
        self.assertAlmostEqual(5/3, self.float_domain.expected_distance(2))