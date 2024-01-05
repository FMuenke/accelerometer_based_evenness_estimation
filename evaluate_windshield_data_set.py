import numpy as np
import pandas as pd
import os

import seaborn as sns
import matplotlib.pyplot as plt
from data_access_lib.utils import check_n_make_dir

from processing.processing_pipeline import AccelerometerSignalProcessingPipeline

from util.score import score_inter_setup_deviation_raw


def main():
    results_path = "./results"
    check_n_make_dir(results_path)
    results_path = os.path.join(results_path, "analyze-windshield-parameters")
    check_n_make_dir(results_path, clean=True)

    df = pd.read_csv("./data/windshield_data_set.csv")

    aspp_0 = AccelerometerSignalProcessingPipeline([], "RMS")
    aspp_1 = AccelerometerSignalProcessingPipeline(["avg-5"], "STD")
    aspp_2 = AccelerometerSignalProcessingPipeline(["avg-3", "avg-3"], "RMS")

    aspp_list = [aspp_0, aspp_1, aspp_2]
    aspp_ids = [str(aspp) for aspp in aspp_list]

    for aspp in aspp_list:
        df[str(aspp)] = aspp.compute_df(df)

    id_vars = ["note", "car", "phone"]
    
    print("Mounting Strength Test")
    df_one = df[df["account"] == "Mounting Strength Test"]
    for f in aspp_ids:
        sc = score_inter_setup_deviation_raw(df_one, f)
        print("{}: {}".format(f, sc))
    df_one = pd.melt(
        df_one, value_name="Unevenness Prediction", value_vars=aspp_ids, var_name="ASPP", id_vars=id_vars)
    df_one["Mounting"] = df_one["note"]
    df_one = df_one.replace({"Mounting": {"Exp. 4 - Loose Mounting": "Loose", "Exp. 4 - Tight Mounting": "Tight"}})
    sns.boxplot(data=df_one, x="ASPP", y="Unevenness Prediction", hue="Mounting")
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "mounting-strength.png"))
    plt.close()

    df_one = df[df["account"] == "Mounting Type Test"]
    print("Mounting Type Test")
    for f in aspp_ids:
        sc = score_inter_setup_deviation_raw(df_one, f)
        print("{}: {}".format(f, sc))
    df_one = pd.melt(
        df_one, value_name="Unevenness Prediction", value_vars=aspp_ids, var_name="ASPP", id_vars=id_vars)
    df_one["Mounting"] = df_one["note"]
    df_one = df_one.replace({"Mounting": {"Exp. 5 - New Mounting": "2 Joints", "Exp. 5 -Old Mounting": "1 Joint"}})
    sns.boxplot(data=df_one, x="ASPP", y="Unevenness Prediction", hue="Mounting")
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "mounting-type.png"))
    plt.close()

    df_one = df[df["account"] == "Unevenness Test Position"]
    print("Unevenness Test Position")
    for f in aspp_ids:
        sc = score_inter_setup_deviation_raw(df_one, f)
        print("{}: {}".format(f, sc))
    df_one = pd.melt(
        df_one, value_name="Unevenness Prediction", value_vars=aspp_ids, var_name="ASPP", id_vars=id_vars)
    df_one["Position"] = df_one["note"]
    df_one = df_one.replace({"Position": {"Exp. 1 - Position-1": "Middle", "Exp. 1 - Position-2": "Right"}})
    sns.boxplot(data=df_one, x="ASPP", y="Unevenness Prediction", hue="Position")
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "mounting-position.png"))
    plt.close()

    df_one = df[df["account"].isin(["Phone Type Test", "Phone Type Test 2"])]
    print("Phone Type Test")
    for f in aspp_ids:
        sc = score_inter_setup_deviation_raw(df_one, f)
        print("{}: {}".format(f, sc))
    df_one = pd.melt(
        df_one, value_name="Unevenness Prediction", value_vars=aspp_ids, var_name="ASPP", id_vars=id_vars)
    df_one["Smartphone"] = df_one["note"]
    df_one = df_one.replace({"Smartphone": {
        "Exp. 8 - iPhone 13 (no case)": "iPhone 13 (no case)",
        "Exp. 8 - iPhone XR (no case)": "iPhone XR (no case)",
        "Exp. 6 - Smartphone 13": "iPhone 13",
        "Exp. 6 - Smartphone XR": "iPhone XR",
    }})
    sns.boxplot(data=df_one, x="ASPP", y="Unevenness Prediction", hue="Smartphone", hue_order=[
        "iPhone 13", "iPhone XR (no case)", "iPhone XR", "iPhone 13 (no case)"
    ])
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "smartphone-type.png"))
    plt.close()



    print("[INFO]: {} Data Points".format(len(df)))
    print("[INFO]: {} Accounts".format(len(df["account"].unique())))
    print("[INFO]: {} Cars".format(len(df["car"].unique())))
    print("[INFO]: {} phones".format(len(df["phone"].unique())))


if __name__ == "__main__":
    main()
