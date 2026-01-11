# data_processing.py
# 处理生理数据，将写作数据和生理数据全部处理为训练数据，为模型搭建创造标准化数据

import pandas as pd
import numpy as np

def compute_cluster_score(row, items, tp):
    vals = []
    for i in items:
        col = f"PCL{i}_{tp}"
        if col in row and not pd.isna(row[col]):
            vals.append(row[col])
    return np.sum(vals) if len(vals) > 0 else np.nan

def extract_trajectory_features(values):
    values = np.array(values, dtype=float)
    if np.any(np.isnan(values)):
        return {
            "change": np.nan,
            "mean": np.nan,
            "std": np.nan
        }

    return {
        "change": values[-1] - values[0],
        "mean": values.mean(),
        "std": values.std()
    }


if __name__ == "__main__":
    TASK1_PATH = "./task1_scored.xlsx"     
    DATA_PATH = "./积石山数据_liyu20250221(1).xlsx"      
    OUTPUT_PATH = "./task2_processing_data.xlsx"

    TIMEPOINTS = ["T1", "T2", "T3"]

    # PCL 症状簇（DSM-5）
    PCL_GROUPS = {
        "PCL_intrusion": [1, 2, 3, 4, 5],
        "PCL_avoidance": [6, 7],
        "PCL_mood": list(range(8, 15)),
        "PCL_arousal": list(range(15, 21))
    }


    task1_df = pd.read_excel(TASK1_PATH)
    data_df = pd.read_excel(DATA_PATH)

    task1_df["Name"] = task1_df["Name"].astype(str)
    data_df["Name"] = data_df["Name"].astype(str)


    records = []

    for _, row in data_df.iterrows():
        record = {"Name": row["Name"]}

        for scale in ["PCL", "GAD", "PHQ", "SDQ"]:
            for tp in TIMEPOINTS:
                col = f"{scale}_{tp}"
                if col in row:
                    record[col] = row[col]

        for cluster_name, items in PCL_GROUPS.items():
            cluster_values = []
            for tp in TIMEPOINTS:
                score = compute_cluster_score(row, items, tp)
                record[f"{cluster_name}_{tp}"] = score
                cluster_values.append(score)

            traj = extract_trajectory_features(cluster_values)
            record[f"{cluster_name}_change"] = traj["change"]
            record[f"{cluster_name}_mean"] = traj["mean"]
            record[f"{cluster_name}_std"] = traj["std"]

        records.append(record)

    scale_feature_df = pd.DataFrame(records)

    final_df = task1_df.merge(
        scale_feature_df,
        on="Name",
        how="left"
    )
    final_df.to_excel(OUTPUT_PATH, index=False)
    print("The data has been saved to ", OUTPUT_PATH)

