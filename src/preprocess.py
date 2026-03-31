import numpy as np
import pandas as pd


def load_data(path):
    df = pd.read_csv(path, sep=" ", header=None)
    df = df.dropna(axis=1)

    df.columns = (
        ['engine_id', 'cycle'] +
        [f'op_setting_{i}' for i in range(1, 4)] +
        [f'sensor_{i}' for i in range(1, 22)]
    )

    return df


def add_rul_and_label(df, threshold=30):
    max_cycle = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycle.columns = ['engine_id', 'max_cycle']

    df = df.merge(max_cycle, on='engine_id')

    df['RUL'] = df['max_cycle'] - df['cycle']
    df['label'] = (df['RUL'] <= threshold).astype(int)

    return df


def remove_low_variance_features(df):
    variance = df.var(numeric_only=True)
    useful_cols = variance[variance > 1e-5].index.tolist()

    if 'label' not in useful_cols:
        useful_cols.append('label')

    df = df[useful_cols]
    return df


def clean_dataset(df):
    df = remove_low_variance_features(df)

    df = df.drop(columns=['engine_id', 'cycle', 'max_cycle', 'RUL'], errors='ignore')

    df = df.loc[:, ~df.columns.duplicated()]

    return df


def split_by_engine(df):
    engine_ids = df['engine_id'].unique()

    np.random.seed(42)
    np.random.shuffle(engine_ids)

    train_ids = engine_ids[:70]
    val_ids = engine_ids[70:85]
    test_ids = engine_ids[85:]

    train_df = df[df['engine_id'].isin(train_ids)]
    val_df = df[df['engine_id'].isin(val_ids)]
    test_df = df[df['engine_id'].isin(test_ids)]

    return train_df, val_df, test_df


def preprocess_pipeline(path):
    df = load_data(path)
    df = add_rul_and_label(df)

    train_df, val_df, test_df = split_by_engine(df)

    train_df = clean_dataset(train_df)
    val_df = clean_dataset(val_df)
    test_df = clean_dataset(test_df)

    return train_df, val_df, test_df

if __name__ == "__main__":
    train_df, val_df, test_df = preprocess_pipeline("train_FD001.txt")
    print(train_df.shape)
    print(train_df['label'].value_counts())