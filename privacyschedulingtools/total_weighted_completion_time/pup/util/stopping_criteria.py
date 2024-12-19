from privacyschedulingtools.total_weighted_completion_time.entity.result import Outcome


def found_empty_and_nonempty(x):
    return (any(sol.outcome == Outcome.SIZE_ZERO for sol in x["solutions"]) and
            any(sol.outcome == Outcome.FOUND for sol in x["solutions"]))
