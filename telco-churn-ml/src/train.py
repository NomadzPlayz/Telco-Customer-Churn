# src/train.py
"""
Training script untuk Telco Customer Churn.
Simpan artifact joblib di models/lgbm_churn.joblib sebagai dict:
{'model','scaler','columns','numeric_cols'}
dan metrics di models/metrics.json
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_OUT = "models/lgbm_churn.joblib"
METRICS_OUT = "models/metrics.json"
RANDOM_STATE = 42

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Place the CSV in the data/ folder.")
    df = pd.read_csv(path)
    return df

def basic_clean(df):
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan))
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    return df

def preprocess(df):
    df = df.copy()
    # target
    if 'Churn' not in df.columns:
        raise ValueError("Expected 'Churn' column in dataset.")
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    # identify numeric vs categorical
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    if 'Churn' in num_cols:
        num_cols.remove('Churn')
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ['Churn']]
    # one-hot encode categorical columns
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    X = df_enc.drop(columns=['Churn'])
    y = df_enc['Churn'].values
    # train-test split (stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    # scaling numeric columns
    scaler = StandardScaler()
    numeric_in_X = [c for c in num_cols if c in X_train.columns]
    if numeric_in_X:
        X_train[numeric_in_X] = scaler.fit_transform(X_train[numeric_in_X])
        X_test[numeric_in_X] = scaler.transform(X_test[numeric_in_X])
    else:
        scaler = None
    return X_train, X_test, y_train, y_test, scaler, numeric_in_X

def train_lgb(X_train, y_train, X_val, y_val):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbosity': -1,
        'seed': RANDOM_STATE
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)

    # Use callbacks for early stopping & logging (compatible across LightGBM versions)
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )
    return model

def evaluate(model, X, y):
    proba = model.predict(X)
    preds = (proba >= 0.5).astype(int)
    m = {
        'accuracy': float(accuracy_score(y, preds)),
        'precision': float(precision_score(y, preds, zero_division=0)),
        'recall': float(recall_score(y, preds, zero_division=0)),
        'f1': float(f1_score(y, preds, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, proba)),
        'confusion_matrix': confusion_matrix(y, preds).tolist()
    }
    return m

def main():
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    print("Loading data...")
    df = load_data()
    df = basic_clean(df)
    print("Preprocessing...")
    X_train, X_test, y_train, y_test, scaler, numeric_cols = preprocess(df)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print("Training LightGBM...")
    model = train_lgb(X_train, y_train, X_test, y_test)
    print("Evaluating...")
    metrics_train = evaluate(model, X_train, y_train)
    metrics_test = evaluate(model, X_test, y_test)
    print("Train metrics:", metrics_train)
    print("Test metrics:", metrics_test)
    print("Saving artifact...")
    artifact = {
        'model': model,
        'scaler': scaler,
        'columns': X_train.columns.tolist(),
        'numeric_cols': numeric_cols
    }
    joblib.dump(artifact, MODEL_OUT)
    with open(METRICS_OUT, 'w') as f:
        json.dump({'train': metrics_train, 'test': metrics_test}, f, indent=2)
    print("Saved model to", MODEL_OUT)
    print("Saved metrics to", METRICS_OUT)

if __name__ == "__main__":
    main()
