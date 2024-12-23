import json
import os
from os.path import exists

import plotly.express as px
import pandas as pd


def main():
    schedules_path = "../pup_parallel_scripts/output_transformations/different_schedules/schedules_data_08172022_calculate_avg_wait_time_m4_n5-20_p5-50_w1-5_r50_lim180_p10.0xE-3_u20.0xE-3.json"
    path = schedules_path.partition("schedules_data_")[2].partition(".json")[0]
    new_directory = "output_transformations/different_schedules/images/" + path

    if not os.path.exists(new_directory):
        os.mkdir(new_directory)

    with open(schedules_path, 'r') as f:
        data = json.load(f)

    for result in data:
        ul = result["utility_loss"]
        pl = result["privacy_loss"]
        pt = result["privacy_threshold"]
        ut = result["utility_threshold"]

        param_id = result["schedule_param_id"]
        utility = result["utility"]
        transformation = result["transformation"][15:]
        time = result["time"]

        title = f"{transformation} for utility {utility} <br> " \
                f"and schedule generated by P{param_id}<br>" \
                f"@ pt:{pt}, ut:{ut} -> pl:{pl}, ul: {ul} in {time}s"
        file_path = f"{new_directory}/P{param_id}_{transformation}_{path}"

        original = result["original_schedule"]
        privatized = result["privatized_schedule"]

        df_o = pd.DataFrame(original)
        df_o['Type'] = "original"

        if privatized is not None:
            df_p = pd.DataFrame(privatized)
            df_p['Type'] = "privatized"
        else:
            df_p = pd.DataFrame(columns=list(df_o.columns))
            empty_job = {
                "Finish": 0,
                "Start": 0,
                "Type": "privatized",
                "Resource": "M0",
                "Task": "Empty"
            }
            df_p = df_p.append(empty_job, ignore_index=True)

        create_image(df_o, df_p, file_path, title)


def create_image(df_o: pd.DataFrame, df_p: pd.DataFrame, path: str, title: str):
    combined = pd.concat([df_o, df_p], ignore_index=True)
    combined['delta'] = combined['Finish'] - combined['Start']

    fig = px.timeline(combined, x_start="Start", x_end="Finish", y="Resource", color="Resource",
                      facet_row="Type", facet_row_spacing=0.1, title=title)
    fig.layout.xaxis.type = 'linear'
    fig.layout.xaxis2.type = 'linear'

    for d in fig.data:
        filt1 = combined['Resource'] == d.name
        type = d.hovertemplate.partition("Type=")[2].partition("<br>")[0]
        filt2 = combined['Type'] == type
        d.x = combined[filt1 & filt2]['delta'].tolist()

    if not exists(path + ".png"):
        fig.write_image(path + ".png")
    if not exists(path + ".html"):
        fig.write_html(path + ".html")


def create_image_original_only(df_o: pd.DataFrame, path: str, title: str):
    pass

if __name__ == "__main__":
    main()
