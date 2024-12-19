import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator, PercentFormatter
import matplotlib.patches as mpatches
import matplotlib.colors as colors
from venn import venn, pseudovenn
import os

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

inverse_colors = {v:k for k,v in colors.get_named_colors_mapping().items()}

def load_data(path, utility_name, result_type):
    uf_path = os.path.join(path, utility_name)
    file_ending = result_type + ".csv"
    files = [filename for filename in os.listdir(uf_path) if filename.endswith(file_ending)]
    assert len(files) == 1
    df = pd.read_csv(os.path.join(uf_path, files[0]))

    # swap : \u21C4
    # move : \u21B7
    # proc : \u21C5

    # df.transformation = df.transformation.astype("str").apply(lambda t: t.replace("PROCESSING_TIMES", "P"))
    # df.transformation = df.transformation.astype("str").apply(lambda t: t.replace("ALT_MOVE_PROC", "M*P"))
    # df.transformation = df.transformation.astype("str").apply(lambda t: t.replace("MOVE_PROC", "MP"))
    # df.transformation = df.transformation.astype("str").apply(lambda t: t.replace("ALT_MOVE", "M*"))
    # df.transformation = df.transformation.astype("str").apply(lambda t: t.replace("MOVE", "M"))
    # df.transformation = df.transformation.astype("str").apply(lambda t: t.replace("SWAPPING_JOBS", "S"))
    # df.transformation = df.transformation.astype("str").apply(lambda t: t.replace("SWAP_PROC", "SP"))
    # df.transformation = df.transformation.astype("str").apply(lambda t: t.replace("SWAP_ALL_PROC", "S*P"))
    # df.transformation = df.transformation.astype("str").apply(lambda t: t.replace("SWAP_ALL", "S*"))

    df.transformation = df.transformation.astype("str").apply(lambda t: t.replace("PROCESSING_TIMES", "P"))
    df.transformation = df.transformation.astype("str").apply(lambda t: t.replace("MOVE_PROC", "MP"))
    df.transformation = df.transformation.astype("str").apply(lambda t: t.replace("MOVE", "M"))
    df.transformation = df.transformation.astype("str").apply(lambda t: t.replace("SWAP_ALL_PROC", "SP"))
    df.transformation = df.transformation.astype("str").apply(lambda t: t.replace("SWAP_ALL", "S"))

    print(df)

    return df


def load_measures(path, utility_name):
    return load_data(path, utility_name, 'measures')


def load_outcomes(path, utility_name):
    return load_data(path, utility_name, 'outcomes')


def create_multiple_charts(df, uf, filename, reverse=False, color=True, legend=False, show_labels=False, count_schedules=10):
    df = df[df.utility_function == uf]
    df = df.rename(columns={"utility_threshold": "\u03B4", "privacy_threshold": "\u03B5"})

    col_name = "\u03B5" if reverse else '\u03B4'
    row_name = "\u03B4" if reverse else '\u03B5'

    grid = sns.FacetGrid(df, col=row_name, row=col_name, ylim=(0, count_schedules), margin_titles=True,
                         sharex=True, sharey=True, legend_out=True, height=5, aspect=0.75)
    grid.set_ylabels("Rel. Frequency")
    palette = {"non-empty": "gold", "empty": "orange", "both": "yellow", "exhausted": "teal",
               "timeout": "darkslategray"}
    create_stacked(grid, df, palette)

    # l = grid.add_legend(legend_data={"Found solution, with g(s) \u2260 {} in ISA (Found)": non_zero_patch,
    #                                 "Found solution, with g(s) = {} in ISA (Size_zero)": zero_patch,
    #                                 "Found solutions": both_patch,
    #                                 "No solution found, exhausted search space (Exhausted)": exhausted_patch,
    #                                 "No solution found, timeout (Timeout)": timeout_patch})

    for axes in grid.axes.flat:
        ticks_loc = axes.get_xticks()
        axes.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
        _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=45)

        axes.yaxis.set_major_formatter(ticker.PercentFormatter(count_schedules))
        axes.set_xlabel("Perturbation")
        axes.xaxis.set_tick_params(labelbottom=True)

        if not color:
            for c in axes.containers:
                for i, v in enumerate(c):
                    # palette = {"non-empty": "gold", "empty": "orange", "both": "yellow", "exhausted": "teal", "timeout": "darkslategray"}
                    color_name = inverse_colors[colors.to_hex(v.get_facecolor()).upper()]
                    if color_name == 'gold':
                        v.set_hatch('//')
                    elif color_name == 'orange':
                        v.set_hatch('\\\\')
                    elif color_name == 'yellow':
                        v.set_hatch('xx')
                    elif color_name == 'teal':
                        v.set_hatch('..')
                    elif color_name == 'darkslategrey':
                        v.set_hatch('OO')
                    else:
                        print(f'Error - unknown color: {color_name}')

                    v.set_facecolor('white')

        if show_labels:
            for c in axes.containers:
                labels = []
                for i, v in enumerate(c):
                    labels.append(int(v.get_height()))
                axes.bar_label(c, labels=labels, label_type='center')

    if legend:
        if color:
            non_zero_patch = mpatches.Patch(color='gold', label='Found solution, with g(s) \u2260 {} in ISA')
            zero_patch = mpatches.Patch(color='orange', label='Found solution, with g(s) = {} in ISA')
            both_patch = mpatches.Patch(color='yellow', label='Found solutions')
            exhausted_patch = mpatches.Patch(color='teal', label='No solution found, exhausted search space')
            timeout_patch = mpatches.Patch(color='darkslategray', label='No solution found, timeout')
        else:
            non_zero_patch = mpatches.Patch(facecolor='white', hatch='//',
                                            label='Found solution, with g(s) \u2260 {} in ISA')
            zero_patch = mpatches.Patch(facecolor='white', hatch='\\\\', label='Found solution, with g(s) = {} in ISA')
            both_patch = mpatches.Patch(facecolor='white', hatch='xx', label='Found solutions')
            exhausted_patch = mpatches.Patch(facecolor='white', hatch='..',
                                             label='No solution found, exhausted search space')
            timeout_patch = mpatches.Patch(facecolor='white', hatch='OO', label='No solution found, timeout')
        handles = [zero_patch, non_zero_patch, both_patch, exhausted_patch, timeout_patch]

        l = grid.add_legend(legend_data={"Non-Empty": non_zero_patch,
                                          "Empty": zero_patch,
                                          "Both": both_patch,
                                          "Exhausted": exhausted_patch,
                                          "Timeout": timeout_patch})
        l = sns.move_legend(l, "lower right", bbox_to_anchor=(1.2, 1), frameon=True, ncol=5)
    grid.fig.tight_layout()
    grid.savefig(f'plots/transformations-{uf}-{filename}.pdf', dpi=300, format="pdf")
    # plt.show()
    plt.clf()


def create_stacked(grid, df, palette):
    grid.map_dataframe(sns.histplot, x="transformation",
                       hue="outcome", hue_order=["timeout", "exhausted", "both", "empty", "non-empty"],
                       multiple="stack", shrink=0.8, palette=palette)


def create_dodge(grid, df, palette):
    grid.map_dataframe(sns.histplot, x="transformation",
                       hue="outcome", hue_order=["non-empty", "empty", "both", "exhausted", "timeout"],
                       multiple="dodge", shrink=0.8, palette=palette)

    for ax in grid.axes.flat:
        for container in ax.containers:
            ax.bar_label(container, fontsize=6)


def create_venn5_diagram(uf, ut, pt, df_p, df_m, df_s, df_sp, df_mp, fname, fsize=12):
    df_p = df_p[(df_p.utility_threshold == ut) & (df_p.privacy_threshold == pt)]
    df_m = df_m[(df_m.utility_threshold == ut) & (df_m.privacy_threshold == pt)]
    df_s = df_s[(df_s.utility_threshold == ut) & (df_s.privacy_threshold == pt)]
    df_sp = df_sp[(df_sp.utility_threshold == ut) & (df_sp.privacy_threshold == pt)]
    df_mp = df_mp[(df_mp.utility_threshold == ut) & (df_mp.privacy_threshold == pt)]

    schedules_with_solutions = {
        "P": set(df_p["id"].unique()),
        "S": set(df_s["id"].unique()),
        "M": set(df_m["id"].unique()),
        "MP": set(df_mp["id"].unique()),
        "SP": set(df_sp["id"].unique())
    }

    ax = venn(schedules_with_solutions, fmt="{size}", cmap='gist_gray', fontsize=fsize)  # cmap
    for patch in ax.patches:
        patch.set_edgecolor('black')
    ax.figure.savefig(f'plots/transformations-venn-{fname}-{uf}-ut{ut}-pt{pt}.pdf', format="pdf", dpi=300,
                      bbox_inches='tight')
    plt.clf()


def get_min_time_until_found(joint_df, ut, pt):
    df = joint_df[(joint_df.utility_threshold == ut) & (joint_df.privacy_threshold == pt) & (
                (joint_df.outcome_type == 'empty') | (joint_df.outcome_type == 'non-empty'))]
    df = df.loc[df.groupby(['id_s', 'transformation'])['time_found'].idxmin()]
    pd.set_option('display.max_rows', 100)
    return df


def get_unsuccessful(out_df, param_df, ut, pt):
    df = out_df.join(param_df, on="id", rsuffix="_p", lsuffix="_o")
    df = df[(df.privacy_threshold == pt) & (df.utility_threshold == ut)]
    df = df[(df.outcome == 'exhausted') | (df.outcome == 'timeout')]
    return df


def get_successful(out_df, param_df, ut, pt):
    df = out_df.join(param_df, on="id", rsuffix="_p", lsuffix="_o")
    df = df[(df.privacy_threshold == pt) & (df.utility_threshold == ut)]
    df = df[(df.outcome == 'empty') | (df.outcome == 'non-empty') | (df.outcome == 'both')]
    return df


# %%
def create_time_plot(data, uf, x_axis, f_name):
    plt.clf()
    ax = sns.lineplot(data=data, x=x_axis, y="time_found", hue="transformation", style='transformation',
                      markers=True, dashes=False, err_style='band', palette='Greys')
    # err_kws={'elinewidth': 0.5,'capsize': 2}
    if x_axis == 'w_max':
        ax.set(xlabel='$w_{max}$', ylabel='Time (in s)')
    plt.legend(title='Perturbation')
    # plt.show()

    if x_axis == 'w_max':
        ax.get_figure().savefig(f'plots/time-{uf}-{f_name}-weights.pdf', dpi=300, format="pdf", bbox_inches='tight')
    plt.clf()


def create_time_barplot(data, uf, x_axis, f_name, bar=True):
    plt.clf()
    if bar:
        ax = sns.catplot(data=data, x=x_axis, y="time_found", hue="transformation",
                         kind='bar', palette='Greys', aspect=2, height=6, errwidth=2, capsize=0.1, legend=False)
        name = "time-bar-plot"
    else:
        # ax = sns.catplot(data=data, x=x_axis, y="time_found", hue= "transformation",
        #             kind='point', palette='Greys', aspect=2, height=6, errwidth=2, capsize=0.1, legend=False,
        #                join=False, dodge=True)
        # name = "time-point-plot"
        ax = sns.catplot(data=data, x=x_axis, y="time_found", hue="transformation",
                         kind='box', palette='Greys', aspect=2, height=7, legend=False, showfliers=True)
        name = "time-box-plot"

    if x_axis == 'w_max':
        ax.set(xlabel='$w_{max}$', ylabel='Time (in s)')

    plt.legend(title='Perturbation')
    #plt.show()

    if x_axis == 'w_max':
        ax.savefig(f'plots/{name}-{uf}-{f_name}-weights.pdf', dpi=300, format="pdf", bbox_inches='tight')
    plt.clf()


def visualize_results(path, param_file, utility, fname, schedule_count=1000, legend=True, p=None):
    outcomes_df, measures_df, params_df = get_data(path, param_file, utility)
    visualize_results_from_dfs(outcomes_df, measures_df, params_df, utility.replace("_alt", ""), fname, schedule_count, legend, p)


def get_data(path, param_file, utility):
    if param_file:
        params_path = os.path.join(path, utility, param_file)
        params_df = pd.read_csv(params_path)
    else:
        params_df = None

    outcomes_df = load_outcomes(path, utility)
    measures_df = load_measures(path, utility)

    return outcomes_df, measures_df, params_df


def visualize_results_from_dfs(outcomes_df, measures_df, params_df, utility, fname, schedule_count=1000, legend=True, p=None):
    # 2x2 grid chart with success rates
    if p == 0.01:
        sel_outcomes_df = outcomes_df[(outcomes_df.privacy_threshold == 0.01) & (
                (outcomes_df.utility_threshold == 0.02) ^ (outcomes_df.utility_threshold == 0.005))]
    elif p == 0.5:
        sel_outcomes_df = outcomes_df[(outcomes_df.privacy_threshold == 0.5) & (
                (outcomes_df.utility_threshold == 0.02) ^ (outcomes_df.utility_threshold == 0.005))]
    else:
        sel_outcomes_df = outcomes_df[
            ((outcomes_df.privacy_threshold == 0.01) ^ (outcomes_df.privacy_threshold == 0.5)) & (
                    (outcomes_df.utility_threshold == 0.02) ^ (outcomes_df.utility_threshold == 0.005))]

    create_multiple_charts(sel_outcomes_df, utility, fname, color=False, legend=legend, reverse=True, count_schedules=schedule_count, show_labels=True)

    # Venn Diagram
    p_df = outcomes_df[(outcomes_df.transformation == "P") & (
            (outcomes_df.outcome == "non-empty") | (outcomes_df.outcome == "empty") | (
            outcomes_df.outcome == "both"))]
    s_df = outcomes_df[(outcomes_df.transformation == "S") & (
            (outcomes_df.outcome == "non-empty") | (outcomes_df.outcome == "empty") | (
            outcomes_df.outcome == "both"))]
    m_df = outcomes_df[(outcomes_df.transformation == "M") & (
            (outcomes_df.outcome == "non-empty") | (outcomes_df.outcome == "empty") | (
            outcomes_df.outcome == "both"))]
    sp_df = outcomes_df[(outcomes_df.transformation == "SP") & (
            (outcomes_df.outcome == "non-empty") | (outcomes_df.outcome == "empty") | (
            outcomes_df.outcome == "both"))]
    mp_df = outcomes_df[(outcomes_df.transformation == "MP") & (
            (outcomes_df.outcome == "non-empty") | (outcomes_df.outcome == "empty") | (
            outcomes_df.outcome == "both"))]

    create_venn5_diagram(utility, 0.02, 0.5, p_df, m_df, s_df, sp_df, mp_df, fname, fsize=18)

    # time charts
    #joint_df = measures_df.join(params_df, on="id", rsuffix="_p", lsuffix="_s")
    #times_df = get_min_time_until_found(joint_df, 0.02, 0.5)
    #create_time_plot(times_df, utility, 'w_max', fname)
    #create_time_barplot(times_df, utility, 'w_max', fname, False)


# TWCT results - real world data
#visualize_results("Results/real_world_data",
#                  None,
#                  "calculate_twct",
#                  "real_world_data-no_legend")

# AVGW results - real world data
#visualize_results("Results/real_world_data",
#                  None,
#                  "calculate_avg_wait_time",
#                  "real_world_data-no_legend")


#CMAX results - MAKE synthetic data
visualize_results("Results/make_synthetic/",
                  None,
                  "calculate_cmax",
                  "make-synthetic-1000-p0.01-no_legend",
                  schedule_count=1000,
                  legend=False,
                  p=0.01)

visualize_results("Results/make_synthetic/",
                  None,
                  "calculate_cmax",
                  "make-synthetic-1000-p0.5-no_legend",
                  schedule_count=1000,
                  legend=False,
                  p=0.5)

visualize_results("Results/make_synthetic/",
                  None,
                  "calculate_cmax",
                  "make-synthetic-1000-all-no_legend",
                  schedule_count=1000,
                  legend=False)

#AVGW results - MAKE synthetic data
visualize_results("Results/make_synthetic/",
                  None,
                  "calculate_avg_wait_time_with_release_dates",
                  "make-synthetic-p0.5-no_legend",
                  schedule_count=1000,
                  legend=False,
                  p=0.5)

visualize_results("Results/make_synthetic/",
                  None,
                  "calculate_avg_wait_time_with_release_dates",
                  "make-synthetic-1000-p0.01-no_legend",
                  schedule_count=1000,
                  legend=False,
                  p=0.01)

visualize_results("Results/make_synthetic/",
                  None,
                  "calculate_avg_wait_time_with_release_dates",
                  "make-synthetic-1000-all-no_legend",
                  schedule_count=1000,
                  legend=False)
