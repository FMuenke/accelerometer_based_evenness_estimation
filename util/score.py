import pandas as pd
from sklearn.metrics import classification_report, mean_absolute_error, accuracy_score
from scipy.stats import pearsonr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def filter_for_common_segment_ids(df):
    possible_segment_ids = np.arange(df["segment_id"].max())

    segments_collected = possible_segment_ids
    for val, val_grp in df.groupby(["car", "phone"]):
        filtered_segments = []
        val_segments = np.array(val_grp["segment_id"])
        for s in possible_segment_ids:
            if s in segments_collected and s in val_segments:
                filtered_segments.append(s)
        segments_collected = np.array(filtered_segments)

    df = df[df["segment_id"].isin(segments_collected)]
    return df


def score_inter_setup_deviation_raw(df, feature_key):
    features = {}
    diffs = {}
    for ident, grp in df.groupby("note"):
        features[ident] = np.array(grp[feature_key])

    for ident1 in features:
        for ident2 in features:
            if ident1 == ident2:
                continue
            vals_1, vals_2 = features[ident1], features[ident2]
            max_val = np.max([np.max(vals_1), np.max(vals_2)])
            min_val = np.min([np.min(vals_1), np.min(vals_2)])
            vals_1_norm = (vals_1 - min_val) / (max_val - min_val)
            vals_2_norm = (vals_2 - min_val) / (max_val - min_val)
            diffs[f"{ident1}-{ident2}"] = np.abs(np.mean(vals_2_norm) - np.mean(vals_1_norm))
    return 1 - np.mean([diffs[ident] for ident in diffs])


def score_inter_setup_deviation(df, feature_key):
    features = {}
    diffs = {}
    df = df[df["source"] == "ZEB"]
    df = filter_for_common_segment_ids(df)
    df = df.sort_values("segment_id", ascending=True)
    for ident, grp in df.groupby(["car", "phone"]):
        features[ident] = np.array(grp[feature_key])

    for ident1 in features:
        for ident2 in features:
            if ident1 == ident2:
                continue
            vals_1, vals_2 = features[ident1], features[ident2]
            max_val = np.max([np.max(vals_1), np.max(vals_2)])
            min_val = np.min([np.min(vals_1), np.min(vals_2)])
            vals_1_norm = (vals_1 - min_val) / (max_val - min_val)
            vals_2_norm = (vals_2 - min_val) / (max_val - min_val)
            diffs[f"{ident1}-{ident2}"] = np.abs(np.mean(vals_2_norm - vals_1_norm))
    return 1 - np.mean([diffs[ident] for ident in diffs])


def score_grade_capabilities(df, feature_key):
    df = df[df["source"] == "ZEB"]
    target = "ZWAUN_15"
    scores_by_vel = []
    for vel, vel_grp in df.groupby("vel [km/h] (r)"):
        if len(vel_grp[target].unique()) > 1 and len(vel_grp[feature_key].unique()) > 1:
            pearson, _ = pearsonr(vel_grp[feature_key], vel_grp[target])
            scores_by_vel.append(np.abs(pearson))
    return np.mean(scores_by_vel)

