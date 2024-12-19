import itertools
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import AutoMinorLocator

str2chars = {
    "PROCESSING_TIMES": "P",
    "MOVE_PROC": "MP",
    "MOVE": "M",
    "SWAP_ALL_PROC": "SP",
    "SWAP_ALL": "S"
}

aggr2words = {
    "avg": "Mean",
    "max": "Max",
    "emd": "EMD",
    "emd_far": "EMD-F&R",
    "Euclidian": "L2",
    "Chebyshev": "L\u221E",
    "Manhattan": "L1",
    "MAE": "MAE"
}

prop2abbr = {
    "waiting_time": "WT",
    "processing_time": "PT",
    "job_completion_time": "JCT",
    "tasks": "T",
    "machine_completion_times": "MCT"
}

# Set font size
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def create_boxplots_all(params, x_attr, col_attr, source_path, target_path):
    df_list = []
    for property_name, transformation in itertools.product(params['properties'], params['transformations']):
        # fig, axes = plt.subplots(nrows=1, ncols=3)
        for i, aggregate_name in enumerate(params['aggregate_functions']):
            df = load_dataset(source_path, aggregate_name, property_name, transformation)
            df_list.append(df)

    df_all = pd.concat(df_list)
    df_all.reset_index(inplace=True)
    grid = sns.FacetGrid(df_all, col=col_attr, row="transformation", margin_titles=True,
                         sharey='col', sharex=True, legend_out=False, height=5, aspect=0.75)

    grid.map_dataframe(sns.boxplot, hue='(\u03B4, \u03B5)', y='dt_difference', x=x_attr,
                       flierprops={"marker": "o"}, fliersize=0.9, palette="Greys")
    for ax in grid.axes.flat:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_facecolor('#EBEBEB')
        ax.grid(which='major', color='white', linewidth=0.6, axis='y')

    grid.set_titles(col_template='{col_name}')
    #grid.set_titles(col_template='{col_name}', row_template='')
    grid.add_legend(title='(\u03B4, \u03B5)', loc="lower left", bbox_to_anchor=(-0.25, 1.1), ncol=4)
    grid.set_axis_labels(y_var="Abs. Diff. in Performance", x_var=None)
    grid.set(xlabel=None)

    grid.savefig(target_path, dpi=300, format="pdf")
    plt.close()

def load_dataset(source_path, aggregate_name, property_name, transformation):
    df = pd.read_csv(os.path.join(source_path, f"{aggregate_name}_{property_name}.csv"))
    df_alt = pd.read_csv(os.path.join(source_path + "_alt", f"{aggregate_name}_{property_name}.csv"))
    df = pd.concat([df, df_alt])
    df.reset_index(inplace=True)

    # Prepare dataset
    df = df[(df['transformation'] == transformation)]
    df.transformation = df.transformation.astype("str").apply(
        lambda t: t.replace(transformation, str2chars[transformation]))

    df = df[((df['utility_threshold'] == 0.005) | (df['utility_threshold'] == 0.02)) & (
            (df['privacy_threshold'] == 0.01) | (df['privacy_threshold'] == 0.5))]
    df['(\u03B4, \u03B5)'] = "(" + df['utility_threshold'].astype("str") + ", " + df['privacy_threshold'].astype(
        "str") + ")"

    df['property'] = prop2abbr[property_name]
    df['aggregate'] = aggr2words[aggregate_name]

    return df

params_all = {
    'aggregate_functions': ['Euclidian', 'Chebyshev', 'Manhattan', "MAE", 'avg', 'max', 'emd'],
    'properties': ["waiting_time", "job_completion_time", "machine_completion_times", "tasks", "processing_time"],
    'transformations': ["PROCESSING_TIMES", "MOVE_PROC", "MOVE", "SWAP_ALL_PROC", "SWAP_ALL"],
    'utility_thresholds': [0.005, 0.02],
    'privacy_thresholds': [0.01, 0.5],
}
directory_path = "pup_parallel_scripts/Results/calculate_twct"
plot_path = "pup_parallel_scripts/plots/downstream_analysis/calculate_twct"
create_boxplots_all(params_all, 'property', 'aggregate', directory_path, os.path.join(plot_path, f"downstream_analysis-calculate_twct-all.pdf"))

directory_path = "pup_parallel_scripts/Results/calculate_avg_wait_time"
plot_path = "pup_parallel_scripts/plots/downstream_analysis/calculate_avg_wait_time"
create_boxplots_all(params_all, 'property', 'aggregate', directory_path, os.path.join(plot_path, f"downstream_analysis_calculate_avg_wait_time-all.pdf"))

params_l1 = {
    'aggregate_functions': ["Manhattan", 'max', 'emd'],
    'properties': ["waiting_time", "job_completion_time"],
    'transformations': ["MOVE"],
    'utility_thresholds': [0.005, 0.02],
    'privacy_thresholds': [0.01, 0.5]
}

directory_path = "pup_parallel_scripts/Results/calculate_twct"
plot_path = "pup_parallel_scripts/plots/downstream_analysis/calculate_twct"
create_boxplots_all(params_l1, 'property', 'aggregate', directory_path, os.path.join(plot_path, f"downstream_analysis-calculate_twct-l1.pdf"))

directory_path = "pup_parallel_scripts/Results/calculate_avg_wait_time"
plot_path = "pup_parallel_scripts/plots/downstream_analysis/calculate_avg_wait_time"
create_boxplots_all(params_l1, 'property', 'aggregate', directory_path, os.path.join(plot_path, f"downstream_analysis_calculate_avg_wait_time-l1.pdf"))

params_mae = {
    'aggregate_functions': ["MAE",'max', 'emd'],
    'properties': ["waiting_time", "job_completion_time"],
    'transformations': ["MOVE"],
    'utility_thresholds': [0.005, 0.02],
    'privacy_thresholds': [0.01, 0.5]
}

directory_path = "pup_parallel_scripts/Results/calculate_twct"
plot_path = "pup_parallel_scripts/plots/downstream_analysis/calculate_twct"
create_boxplots_all(params_mae, 'property', 'aggregate', directory_path, os.path.join(plot_path, f"downstream_analysis-calculate_twct-mae.pdf"))

directory_path = "pup_parallel_scripts/Results/calculate_avg_wait_time"
plot_path = "pup_parallel_scripts/plots/downstream_analysis/calculate_avg_wait_time"
create_boxplots_all(params_mae, 'property', 'aggregate', directory_path, os.path.join(plot_path, f"downstream_analysis_calculate_avg_wait_time-mae.pdf"))