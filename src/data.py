import pandas as pd
import numpy as np

def load_data(data_dir="data"):
    train = pd.read_csv(f"{data_dir}/train.csv")
    demo = pd.read_csv(f"{data_dir}/train_demographics.csv")
    return train, demo

def extract_sequence_features(df, is_train=True):
    # Sensör sütunlarını bul (acc_, rot_, thm_, tof_ ile başlayanlar)
    sensor_cols = [c for c in df.columns if c.startswith(('acc_', 'rot_', 'thm_', 'tof_'))]
    
    # Her sequence_id için ortalama ve standart sapma hesapla
    agg_funcs = {col: ['mean', 'std'] for col in sensor_cols}
    features = df.groupby('sequence_id').agg(agg_funcs)
    
    # Sütun isimlerini düzleştir (örn: acc_x_mean)
    features.columns = ['_'.join(col).strip() for col in features.columns.values]
    features = features.reset_index()
    
    # Meta bilgileri (subject ve varsa gesture) ekle
    meta_cols = ['sequence_id', 'subject']
    if is_train and 'gesture' in df.columns:
        meta_cols.append('gesture')
        
    meta = df[meta_cols].drop_duplicates('sequence_id')
    result = pd.merge(meta, features, on='sequence_id', how='left')
    return result

def preprocess(df):
    # Eksik sensör değerlerini (NaN) 0 ile doldur
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    return df