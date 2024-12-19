import typing
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum


# the different outcomes for a search with the privacy threshold being based on the attack success probability
from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import ParallelSchedule
from privacyschedulingtools.total_weighted_completion_time.entity.schedule import Schedule

class Outcome(Enum):
    NOT_FOUND = 0
    SIZE_ZERO = 1
    TRUE_W_NOT_IN_SET = 2
    SOLUTION_COUNT_LARGE_ENOUGH = 3
    FOUND = 4
    TIMEOUT = 5


@dataclass
class Result:
    outcome: Outcome = None
    original_schedule: Schedule = None
    privatized_schedule: Schedule = None
    privacy_loss: float = None
    privacy_loss_per_job: typing.List = None
    utility_loss: float = None

@dataclass
class ParallelResult:
    outcome: Outcome = None
    dismissed: bool = None
    original_schedule: ParallelSchedule = None
    privatized_schedule: ParallelSchedule = None
    privacy_loss: float = None
    privacy_loss_per_job: typing.List = None
    utility_loss: float = None
    depth: int = None
    time_found: float = None