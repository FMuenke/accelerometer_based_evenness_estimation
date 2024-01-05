import pandas as pd

full_df = pd.read_csv("../unevenness-grade/data/zeb_data_set.csv")

full_df = full_df.replace({"car": {
        "Tim Formentor": "SUV",
        "Opel Van": "Van 1",
        "VW Small Van": "Van 2",
        "Mercedes VAN": "Van 3",
        "Tim Private Car": "Car"
    }})

full_df = full_df[["AUN", "ZWAUN_15", "segment_id", "segment_type", "raw_accelerometer_signal", "car", "mounting", "phone", "vel [m/s]", "vel [km/h]", "vel [km/h] (r)"]]
full_df["source"] = "ZEB"
full_df["setup"] = full_df["phone"] + full_df["car"] + full_df["mounting"]
full_df.to_csv("./data/zeb_data_set.csv")


full_df = pd.read_csv("../unevenness-grade/data/outlier_data_set.csv")

full_df = full_df[["raw_accelerometer_signal", "car", "mounting", "phone", "vel [m/s]", "vel [km/h]", "vel [km/h] (r)", "note", "account"]]
full_df["source"] = "ZEB"
full_df["setup"] = full_df["phone"] + full_df["car"] + full_df["mounting"]
full_df.to_csv("./data/windshield_data_set.csv")