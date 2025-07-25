# ─── baseline_utils.py ────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import random
import sys
import os 
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib  # for saving models

# Configuration
SEED       = 42
random.seed(SEED)
np.random.seed(SEED)

def load_data(
    train_path: str = r"C:\Users\marco\Desktop\Thesis\data\processed\train_sample_baseline.csv",
    test_path: str = r"C:\Users\marco\Desktop\Thesis\data\processed\test_sample_baseline.csv"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and return train/test DataFrames."""
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test


def process_binary_labels(df: pd.DataFrame):
    """
    Convert labels to binary (0, 1) if they are not already.
    0 = Human, 1 = AI.
    """
    df["binary_label"] = df["model"].apply(lambda m: "human" if m == "human" else "artificial")
    df["binary_label_code"] = (df["binary_label"] == "artificial").astype(int)
    return df

def process_base_ablation_labels(df: pd.DataFrame):
    pruned_features = [
    'gpt2_perplexity',
    'type_token_ratio',
    'repeated_3gram_ratio',
    'unique_2grams',
    'unique_3grams',
    'sentence_length_std',
    'sentence_length_entropy',
    'pos_transition_entropy',
    'punctuation_ratio',
    'avg_word_length',
    'flesch_reading_ease',
    'pos_ratio_X',
    'binary_label',
    'binary_label_code',
    'generation',
    'id',
    'adv_source_id',
    'source_id'
]

    ablation_df = df[pruned_features].copy()
    return ablation_df


def extract_baseline_features(df: pd.DataFrame):
    """
    Extract whatever surface‐level features you used originally,
    e.g. perplexity, token counts, POS ratios, etc.
    Returns X (DataFrame) and y (Series).
    """
    # Example placeholder:
    X = pd.DataFrame({
        "token_count": df.text.str.split().str.len(),
        "avg_word_len": df.text.str.len() / df.text.str.split().str.len(),
        # … add your real features here …
    })
    y = df.label
    return X, y

def train_baseline_model(X_train, y_train, **model_kwargs):
    """Train and return a logistic regression (or whatever) classifier."""
    model = LogisticRegression(**model_kwargs, random_state=SEED)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Compute and return a dict of Accuracy, F1, ROC-AUC.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1"      : f1_score(y_test, y_pred),
        "roc_auc" : roc_auc_score(y_test, y_proba)
    }

def save_model(model, out_path: Path):
    """Persist trained model to disk."""
    joblib.dump(model, str(out_path))

def log_experiment(metrics: dict, params: dict, log_path: Path):
    """
    Append one run to a CSV log.
    metrics: {"accuracy":…, "f1":…, "roc_auc":…}
    params: hyperparameters used
    """
    import csv
    header = ["run_id"] + list(params.keys()) + list(metrics.keys())
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    row   = [run_id] + list(params.values()) + list(metrics.values())
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)
    return run_id
