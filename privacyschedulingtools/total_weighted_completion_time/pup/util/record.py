from numpy import mean

class Record:
    def __init__(self, true_schedule, published_schedule, outcome, candidates, pl_exact, pl_distance_per_job):
        self.true_schedule = true_schedule
        self.published_schedule = published_schedule
        self.outcome = outcome
        self.candidates = candidates
        self.pl_exact = pl_exact
        self.pl_distance = pl_distance_per_job

    def csv_list(self):
        return [self.true_schedule,
                self.published_schedule,
                self.outcome,
                len(self.candidates),
                self.pl_exact,
                self.pl_distance,
                min(self.pl_distance),
                max(self.pl_distance),
                mean(self.pl_distance)]

    @staticmethod
    def csv_desc():
        return ["true_schedule", "published_schedule", "outcome", "candidates", \
                "pl_exact", "pl_distance_per_job", "min(pl_distance)", "max(pl_distance)", "mean(pl_distance)"]