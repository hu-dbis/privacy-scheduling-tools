# generate random schedule parameters
import pickle
import typing
from datetime import datetime
import random

from privacyschedulingtools.total_weighted_completion_time.entity.domain import IntegerDomain
from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import ParallelScheduleFactory
from privacyschedulingtools.total_weighted_completion_time.entity.scheduling_parameters import \
    ParallelSchedulingParameters
from privacyschedulingtools.total_weighted_completion_time.pup.util.file_utilities import write_csv_data


class RandomParameterGenerator:
    def __init__(self, trials=1, max_machines=4, job_count_range=(5, 20), processing_time_range=(5, 50), weight_range=(1,10),
                 schedule_type="wspt"):
        self.trials = trials
        self.max_machines = max_machines
        self.job_count_range = job_count_range
        self.processing_time_range = processing_time_range
        #self.w_max = w_max
        self.weight_range = weight_range
        self.generated_parameters = []
        self.generated_schedules = []
        self.schedule_type = schedule_type

    def generate_random_parameter_list(self):
        self.generated_parameters = []
        for i in range(self.trials):
            self.generated_parameters.append(self.generate_random_parameters())

        return self.generated_parameters

    def generate_random_parameters(self):
        machine_count = random.randint(1, self.max_machines)
        job_count = random.randint(self.job_count_range[0], self.job_count_range[1])
        processing_time_domain = self.create_domain(self.processing_time_range)
        weight_domain = IntegerDomain(self.weight_range[0], random.randint(self.weight_range[0]+1, self.weight_range[1]))

        generated_parameter = ParallelSchedulingParameters(
            machine_count=machine_count,
            job_count=job_count,
            processing_time_domain=processing_time_domain,
            weight_domain=weight_domain
        )

        return generated_parameter

    def generate_schedules_from_parameters(self):
        self.generated_schedules = []
        for i, params in enumerate(self.generated_parameters):
            factory = ParallelScheduleFactory(params)
            schedule = factory.generate_random_schedule_with_dispatching_rule(self.schedule_type)
            self.generated_schedules.append((i, schedule))

        return self.generated_schedules

    def create_domain(self, domain):
        if not isinstance(domain, IntegerDomain):
            assert isinstance(domain, typing.Tuple)

            min_x, max_x = domain
            lim1, lim2 = random.randint(min_x, max_x), random.randint(min_x, max_x)
            domain = IntegerDomain(lim1, lim2) if lim1 < lim2 else IntegerDomain(lim2, lim1)

        return domain

    def create_file_name(self):
        w_min, w_max = self.weight_range

        if isinstance(self.processing_time_range, IntegerDomain):
            p_min, p_max = self.processing_time_range.get_min(), self.processing_time_range.get_max()
        else:
            p_min, p_max = self.processing_time_range

        j_min, j_max = self.job_count_range

        file_name = f"{datetime.now().strftime('%m%d%Y')}" \
                    f"_{datetime.now().strftime('%H%M%S')}_"\
                    f"m{self.max_machines}_n{j_min}-{j_max}_" \
                    f"p{p_min}-{p_max}_w{w_min}-{w_max}_" \
                    f"r{self.trials}"

        return file_name

    def save_schedules(self, path):
        if self.generated_schedules:
            with open(path, "wb") as file:
                pickle.dump(self.generated_schedules, file)

    def save_schedule_stats(self, stats_path):
        field_names = ["id", "machine_counts", "job_counts", "w_min", "w_max", "p_min", "p_max",
                       "processing_time_range", "weighted_processing_time_range"]
        stats = []
        for i, schedule_params in enumerate(self.generated_parameters):
            max_p = schedule_params.processing_time_domain.get_max()
            min_p = schedule_params.processing_time_domain.get_min()

            min_w = schedule_params.weight_domain.get_min()
            max_w = schedule_params.weight_domain.get_max()

            # save schedule stats
            stats.append({
                "id": i,
                "machine_counts": schedule_params.machine_count,
                "job_counts": schedule_params.job_count,
                "w_min": min_w,
                "w_max": max_w,
                "p_min": min_p,
                "p_max": max_p,
                "processing_time_range": max_p - min_p,
                "weighted_processing_time_range": max_w * max_p - min_w * min_p
            })

        write_csv_data(stats_path, stats, field_names)
