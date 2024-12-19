path = "Results/"
# TWCT results with new swap neighborhood
old_twct_outcomes, old_twct_measures, twct_params_df = get_data(path,
                         "calculate_twct_07312023_104612_m4_n5-20_p5-50_w1-10_r1000_schedule_params.csv",
                         "calculate_twct")

alt_twct_outcomes, alt_twct_measures, _ = get_data(path,
                         "calculate_twct_07312023_104612_m4_n5-20_p5-50_w1-10_r1000_schedule_params.csv",
                         "calculate_twct_alt")
concat_outcomes = pd.concat([old_twct_outcomes, alt_twct_outcomes], ignore_index=True)
#display(concat_outcomes)
concat_measures = pd.concat([old_twct_measures, alt_twct_measures], ignore_index=True)
#display(concat_measures)
sorted_twct_outcomes = concat_outcomes.sort_values(by='transformation', ascending=True, inplace=False)