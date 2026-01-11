# src/modeling/feature_mining.py
"""
我们尝试挖掘高级机制特征：
1. 主观-客观错位 (Dissociation): NLP Arousal vs SUDS
2. 意义加工峰值 (Breakthrough): Max Reflection
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def mine_advanced_features(df_session: pd.DataFrame) -> pd.DataFrame:
    df = df_session.copy()
    
    required_cols = ['Name', 'arousal', 'suds_after', 'reflection']
    for col in required_cols:
        if col not in df.columns:
            print(f"[Warning] 缺少列 {col}，跳过高级特征挖掘。")
            return pd.DataFrame()


    scaler = MinMaxScaler()

    df['arousal_filled'] = df['arousal'].fillna(df['arousal'].median())
    df['suds_filled'] = df['suds_after'].fillna(df['suds_after'].median())

    df['norm_arousal'] = scaler.fit_transform(df[['arousal_filled']])
    df['norm_suds'] = scaler.fit_transform(df[['suds_filled']])

    # 计算主观-客观错位指数
    df['dissociation_index'] = (df['norm_arousal'] - df['norm_suds']).abs()

    # 聚合到被试级
    agg_funcs = {
        'dissociation_index': ['mean', 'max'],
        'reflection': ['max']  
    }
    
    df_mining = df.groupby('Name').agg(agg_funcs)
    df_mining.columns = [f"Advanced_{col[0]}_{col[1]}" for col in df_mining.columns]
    df_mining = df_mining.reset_index()
    
    print(f"[Feature Mining] Constructed {len(df_mining.columns)-1} advanced mechanism features (Dissociation & Breakthrough).")
    return df_mining