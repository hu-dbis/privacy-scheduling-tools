from dataclasses import dataclass

from privacyschedulingtools.total_weighted_completion_time.entity.domain import Domain


@dataclass(frozen=True)
class SchedulingParameters:
    job_count: int
    processing_time_domain: Domain
    weight_domain: Domain


@dataclass(frozen=True)
class ParallelSchedulingParameters:
    job_count: int
    machine_count: int
    processing_time_domain: Domain
    weight_domain: Domain

