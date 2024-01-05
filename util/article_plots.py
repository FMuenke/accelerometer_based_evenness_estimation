
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import pearsonr


def plot_feature(df, feature, path, count):
    zeb_df = df[df["source"] == "ZEB"]
    pearson_zwaun, _ = pearsonr(zeb_df[feature], zeb_df["ZWAUN_15"])

    sns.scatterplot(zeb_df, x=feature, y="ZWAUN_15")
    plt.title("ZWAUN Pearson: {}".format(pearson_zwaun))
    plt.savefig(os.path.join(path, "ZWAUN_15-{}-feature-{}.png".format(count, feature)))
    plt.close()

    sns.boxplot(zeb_df, x="vel [km/h] (r)", y=feature, hue="gtr - grade")
    plt.title("ZWAUN Pearson: {}".format(pearson_zwaun))
    plt.savefig(os.path.join(path, "gtr-grade-{}-feature-{}.png".format(count, feature)))
    plt.close()

    plt.figure(figsize=(16, 10))
    ax = sns.boxplot(zeb_df, x="vel [km/h] (r)", y=feature, hue="setup")
    plt.title("ZWAUN Pearson: {}".format(pearson_zwaun))
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(path, "setup-{}-feature-{}.png".format(count, feature)))
    plt.close()


def plot_parameters(feature_df, result_path, ident_key, param_name, title):
    avg_filter_df = []
    print("Moving Avg")
    for f in ["RMS", "STD", "MOM"]:  # ["MAX", "MFFT", "MOM", "STD", "RMS", "GRMS"]:
        for i in [3, 5, 7, 9, 11]:
            k = f"{ident_key}{i}-{f}"

            s_u = round(np.mean(feature_df[feature_df["feature"] == k]["grading"]), 3)
            s_s = round(np.mean(feature_df[feature_df["feature"] == k]["consistency"]), 3)
            s = round(np.mean(feature_df[feature_df["feature"] == k]["overall"]), 3)
            # print("{} & {} & {} & {}\\\\".format(k, s_u, s_s, s))

            avg_filter_df.append({
                "feature": k,
                param_name: i,
                r'$S_U$': s_u,
                r'$S_S$': s_s,
                r'$S$': s,
                "Aggregation": f
            })

    avg_filter_df = pd.DataFrame(avg_filter_df)
    sns.set(font_scale=1.2, style="whitegrid")
    sns.scatterplot(data=avg_filter_df, x=r'$S_U$', y=r'$S_S$', hue=param_name, style="Aggregation", s=200)
    plt.title(title)
    # plt.legend(loc='lower left', ncol=2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, f"moving_{ident_key}.png"))
    plt.savefig(os.path.join(result_path, f"moving_{ident_key}.eps"))
    plt.close()


def plots_for_papers(aspp_df_df, df, result_path):
    df = df[df["source"] == "ZEB"]
    df["Vehicle"] = df["car"]
    df["iPhone"] = df["phone"]
    df["ZEB Grade"] = np.round(df["ZWAUN_15"]).astype(np.int32)
    df = df[df["vel [km/h]"] > 5]
    df = df[df["vel [km/h]"] < 75]

    print("Baseline")
    for i, row in aspp_df_df.iterrows():
        if "raw" in row["feature"]:
            print("{} & {} & {} & {}\\\\".format(
                row["feature"], round(row["grading"], 3), round(row["consistency"], 3), round(row["overall"], 3)))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    df["Unevenness Prediction (raw-RMS)"] = df["raw-RMS"]
    sns.boxplot(data=df, x="vel [km/h] (r)", y="Unevenness Prediction (raw-RMS)", hue="ZEB Grade", ax=axes[0])
    sns.boxplot(data=df, x="Vehicle", y="Unevenness Prediction (raw-RMS)", hue="iPhone", ax=axes[1])
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "baseline.png"))
    plt.close()

    sns.boxplot(data=df, x="iPhone", y="raw-RMS", hue="Vehicle")
    plt.savefig(os.path.join(result_path, "baseline-setup2.png"))
    plt.savefig(os.path.join(result_path, "baseline-setup2.eps"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    df["Unevenness Prediction (avg5-STD)"] = df["avg5-STD"]
    sns.boxplot(data=df, x="vel [km/h] (r)", y="Unevenness Prediction (avg5-STD)", hue="ZEB Grade", ax=axes[0])
    sns.boxplot(data=df, x="Vehicle", y="Unevenness Prediction (avg5-STD)", hue="iPhone", ax=axes[1])
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "aspp_1.png"))
    plt.savefig(os.path.join(result_path, "aspp_1.eps"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    df["Unevenness Prediction (avg3avg3-RMS)"] = df["avg3avg3-RMS"]
    sns.boxplot(data=df, x="vel [km/h] (r)", y="Unevenness Prediction (avg3avg3-RMS)", hue="ZEB Grade", ax=axes[0])
    sns.boxplot(data=df, x="Vehicle", y="Unevenness Prediction (avg3avg3-RMS)", hue="iPhone", ax=axes[1])
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "aspp_2.png"))
    plt.savefig(os.path.join(result_path, "aspp_2.eps"))
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    sns.boxplot(data=df, x="Vehicle", y="Unevenness Prediction (raw-RMS)", hue="iPhone", ax=axes[0])
    axes[0].get_legend().remove()
    sns.boxplot(data=df, x="Vehicle", y="Unevenness Prediction (avg5-STD)", hue="iPhone", ax=axes[1])
    axes[1].get_legend().remove()
    sns.boxplot(data=df, x="Vehicle", y="Unevenness Prediction (avg3avg3-RMS)", hue="iPhone", ax=axes[2])
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "setup_compared.png"))
    plt.savefig(os.path.join(result_path, "setup_compared.eps"))
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    sns.boxplot(data=df, x="vel [km/h] (r)", y="Unevenness Prediction (raw-RMS)", hue="ZEB Grade", ax=axes[0])
    sns.boxplot(data=df, x="vel [km/h] (r)", y="Unevenness Prediction (avg5-STD)", hue="ZEB Grade", ax=axes[1])
    axes[1].get_legend().remove()
    sns.boxplot(data=df, x="vel [km/h] (r)", y="Unevenness Prediction (avg3avg3-RMS)", hue="ZEB Grade", ax=axes[2])
    axes[2].get_legend().remove()
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "grade_compared.png"))
    plt.savefig(os.path.join(result_path, "grade_compared.eps"))
    plt.close()

    plot_parameters(aspp_df_df, result_path, "avg", param_name="Kernel", title="Impact Moving Average Filter")
    plot_parameters(aspp_df_df, result_path, "rmp", param_name="Kernel", title="Impact Ramp Filter")

    print("ASPP 1")
    for i, row in aspp_df_df.sort_values("overall", ascending=False)[:26].iterrows():
        if "raw" in row["feature"]:
            continue
        print("{} & {} & {} & {}\\\\".format(
            row["feature"], round(row["grading"], 3), round(row["consistency"], 3), round(row["overall"], 3)))