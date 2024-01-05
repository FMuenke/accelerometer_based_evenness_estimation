import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from processing.processing_pipeline import GeneralASPP
from util.article_plots import plots_for_papers, plot_feature
from util.path import check_n_make_dir
from util.score import score_inter_setup_deviation, score_grade_capabilities


def digitize_grade(df):
    grade = []
    for zwaun in df["ZWAUN_15"]:
        if zwaun <= 2:
            g = 1
        elif 2 < zwaun <= 4:
            g = 3
        else:
            g = 5
        grade.append(g)

    df["gtr - grade"] = grade
    return df


def score_aspp(df, aspp_id):
    results = {
        "feature": aspp_id,
        "grading": score_grade_capabilities(df, aspp_id),
        "consistency": score_inter_setup_deviation(df, aspp_id)
    }
    results["overall"] = np.mean([results["grading"], results["consistency"]])
    return results


def compute_aspp_scoring_df(df, list_of_aspp_ids):
    print("[INFO] Scoring...")
    aspp_df = []
    for aspp_id in list_of_aspp_ids:
        scores = score_aspp(df, aspp_id)
        aspp_df.append(scores)
    return pd.DataFrame(aspp_df)


def benchmark_features(df, result_path, features_to_check):
    

    sort_by = "overall"

    ax = sns.scatterplot(
        data=features_df.sort_values(sort_by, ascending=False)[:10],
        x="consistency", y="grading", hue="feature")
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "feature_scores.png"))
    plt.savefig(os.path.join(result_path, "feature_scores.eps"))
    plt.close()

    top_k_features = ["vel [km/h]"]
    for feature in features_df.sort_values(sort_by, ascending=False)["feature"].head(4):
        top_k_features.append(str(feature))

    count = 0
    for feature in features_df.sort_values(sort_by, ascending=False)["feature"].head(10):
        plot_feature(df, feature, result_path, count)
        count += 1

    sns.pairplot(data=df, vars=top_k_features, hue="gtr - grade")
    plt.savefig(os.path.join(result_path, "ov_features_grade.png"))
    plt.savefig(os.path.join(result_path, "ov_features_grade.eps"))
    plt.close()

    sns.pairplot(data=df, vars=top_k_features, hue="account")
    plt.savefig(os.path.join(result_path, "ov_features_consistency.png"))
    plt.savefig(os.path.join(result_path, "ov_features_consistency.eps"))
    plt.close()

    print("[INFO] TOP Features on ZEB:")
    print(features_df.sort_values(sort_by, ascending=False))

    return top_k_features


def merge_processed(list_of_processed):
    merged = {}
    for processed in list_of_processed:
        for aspp_id in processed:
            merged[aspp_id] = processed[aspp_id]
    return merged


def main():
    result_path = "./results"
    check_n_make_dir(result_path)
    result_path = os.path.join(result_path, "evaluate-aspp")
    check_n_make_dir(result_path, clean=True)

    full_df = pd.read_csv("./data/zeb_data_set.csv")
    full_df = digitize_grade(full_df)

    # Define the Parameter Spaces
    list_of_operations = [
        "avg-3", "avg-5", "avg-7", "avg-9", "avg-11",
        "rmp-3", "rmp-5", "rmp-7", "rmp-9", "rmp-11",
        "bnd-00/25", "bnd-10/40", "bnd-25/50",
        "bnd-00/10", "bnd-10/20", "bnd-20/30", "bnd-30/40", "bnd-40/50",
    ]

    list_of_aggregations = ["RMS", "STD", "MAX", "MOM", "MFFT"]

    g_aspp_0 = GeneralASPP(list_of_operations, list_of_aggregations, complexity=0)
    processed_0 = g_aspp_0.compute_df(full_df)

    g_aspp_1 = GeneralASPP(list_of_operations, list_of_aggregations, complexity=1)
    processed_1 = g_aspp_1.compute_df(full_df)

    g_aspp_2 = GeneralASPP(list_of_operations, list_of_aggregations, complexity=2)
    processed_2 = g_aspp_2.compute_df(full_df)

    processed = merge_processed([processed_0, processed_1, processed_2])
    aspp_ids = [aspp_id for aspp_id in processed]

    # Add Information to score features and to generate plots
    processed["gtr - grade"] = full_df["gtr - grade"]
    processed["ZWAUN_15"] = full_df["ZWAUN_15"]
    processed["AUN"] = full_df["AUN"]
    processed["setup"] = full_df["setup"]
    processed["car"] = full_df["car"]
    processed["phone"] = full_df["phone"]
    processed["segment_id"] = full_df["segment_id"]
    processed["source"] = full_df["source"]
    processed["vel [km/h]"] = full_df["vel [km/h]"]
    processed["vel [km/h] (r)"] = full_df["vel [km/h] (r)"]

    processed = pd.DataFrame(processed)
    aspp_df = compute_aspp_scoring_df(processed, aspp_ids)
    plots_for_papers(aspp_df, processed, result_path)


if __name__ == "__main__":
    main()
