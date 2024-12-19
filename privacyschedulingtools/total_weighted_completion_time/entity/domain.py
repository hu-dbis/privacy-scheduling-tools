import random

import numpy as np
from numpy import mean
import scipy.integrate as integrate

from privacyschedulingtools.total_weighted_completion_time.pup.util.auto_string import auto_str

@auto_str
class Domain:

    def get_min(self):
        pass

    def get_max(self):
        pass

    def get_random(self):
        pass

    #the expected error between x and a randomly drawn value from the domain
    def expected_distance(self, x):
        pass


# domain including integers between min_value and max_value (inclusively)
# get random draws from the domain with uniform probability
class IntegerDomain(Domain):

    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def get_min(self):
        return self.min_value

    def get_max(self):
        return self.max_value

    def get_random(self):
        return np.random.randint(self.min_value, self.max_value + 1)

    def expected_distance(self, x: int):
        distances = [abs(x - v) for v in range(self.min_value, self.max_value+1)]
        return mean(distances)

    def mean_squared_distance(self, x: int):
        distances = [(x-v)*(x-v) for v in range(self.min_value, self.max_value+1)]
        return mean(distances)

    def to_tuple(self):
        return self.min_value, self.max_value

    def __str__(self):
        return f"({self.min_value}, {self.max_value})"

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.min_value == other.min_value and self.max_value == other.max_value

# domain as an extensional set of valid values (at least interval scale?)
class ExtensionalDomain(Domain):
    def __init__(self, values):
        self.values = values
        self.min_value = min(values)
        self.max_value = max(values)

    def get_min(self):
        return self.min_value

    def get_max(self):
        return self.max_value

    def get_random(self):
        return random.choice(self.values)

    def get_next_greater(self, value):
        return min([v for v in self.values if v > value]) if value != self.max_value else None

    def get_next_lower(self, value):
        return max([v for v in self.values if v < value]) if value != self.min_value else None

    def expected_distance(self, x: int):
        distances = [abs(x - v) for v in self.values]
        return mean(distances)

    def mean_squared_distance(self, x: int):
        distances = [(x-v)*(x-v) for v in self.values]
        return mean(distances)

    def to_tuple(self):
        return tuple(self.values)

    def __str__(self):
        return str(self.values)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.values == other.values

# domain including floats between min_value and max_value, using random.uniform
class FloatDomain(Domain):

    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def get_min(self):
        return self.min_value

    def get_max(self):
        return self.max_value

    def get_random(self):
        return random.uniform(self.min_value, self.max_value)

    def expected_distance(self, x: float):
        # the integration returns a tuple: (estimated value, upper bound of error)
        integral = integrate.quad(lambda v: abs(x-v), self.min_value, self.max_value)

        # the division is by the pdf, pulled out of the integral since it is a constant
        return integral[0] / abs(self.max_value - self.min_value)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.min_value == other.min_value and self.max_value == other.max_value

