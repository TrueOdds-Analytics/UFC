"""
UFC Fight Prediction - Simple XGBoost Training
Just the essentials: load, optimize, evaluate, save selected trials
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
import optuna
import json
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

ACC_THRESHOLD = 0.60  # save trials at/above this accuracy
SAVE_DIR = '../../../saved_models/xgboost/trials/'
os.makedirs(SAVE_DIR, exist_ok=True)


def load_data(train_path='../../../data/train_test/train_data.csv',
              val_path='../../../data/train_test/val_data.csv'):
    """Load train/val and cast string columns to pandas 'category' (aligned)."""
    from pandas.api.types import CategoricalDtype

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)

    drop_cols = ['winner', 'fighter_a', 'fighter_b', 'date', 'current_fight_date']
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols].copy()
    y_train = train_df['winner'].copy()
    X_val   = val_df[feature_cols].copy()
    y_val   = val_df['winner'].copy()

    obj_cols = list(X_train.select_dtypes(include='object').columns)
    for c in obj_cols:
        cats = sorted(set(X_train[c].astype('string').fillna('<NA>')) |
                      set(X_val[c].astype('string').fillna('<NA>')))
        cat_dtype = CategoricalDtype(categories=cats, ordered=False)
        X_train[c] = X_train[c].astype('string').fillna('<NA>').astype(cat_dtype)
        X_val[c]   = X_val[c].astype('string').fillna('<NA>').astype(cat_dtype)

    print(f"Data loaded: Train {X_train.shape}, Val {X_val.shape}")
    if obj_cols:
        print(f"Categorical cols: {obj_cols}")
    return X_train, X_val, y_train, y_val


def optimize_and_save(X_train, y_train, X_val, y_val, n_trials=50):
    """Optuna search + save all good trials"""

    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'device': 'cpu',
            'enable_categorical': True,
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            'early_stopping_rounds': 50
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        val_pred = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_pred)
        val_acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))

        train_pred = model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_pred)
        auc_diff = train_auc - val_auc

        metrics = {
            'train_acc': accuracy_score(y_train, (train_pred > 0.5).astype(int)),
            'train_auc': train_auc,
            'val_acc': val_acc,
            'val_auc': val_auc,
            'auc_diff': auc_diff,
        }

        # Save if it passes threshold
        if val_acc >= ACC_THRESHOLD:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = f"{SAVE_DIR}/trial{trial.number}_acc{val_acc:.3f}_{timestamp}.json"
            model.save_model(model_path)
            with open(f"{SAVE_DIR}/trial{trial.number}_metadata_{timestamp}.json", 'w') as f:
                json.dump({'params': params, 'metrics': metrics}, f, indent=2)
            print(f"✓ Saved trial {trial.number} | acc={val_acc:.3f}, auc={val_auc:.3f}")

        return val_auc  # still optimize AUC

    print(f"\nOptimizing hyperparameters ({n_trials} trials)...")
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")


# ============================================================================

def train_xgboost(n_trials=50):
    """Main pipeline: load, optimize, save good trials"""
    print("=" * 50)
    print("XGBoost Training Pipeline")
    print("=" * 50)

    X_train, X_val, y_train, y_val = load_data()
    optimize_and_save(X_train, y_train, X_val, y_val, n_trials=n_trials)


if __name__ == "__main__":
    train_xgboost(n_trials=50)
