"""
UFC Fight Prediction - Simple XGBoost Training
Just the essentials: load, optimize, train, evaluate, save
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
import optuna
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def load_data(train_path='../../../data/train_test/train_data.csv',
              val_path='../../../data/train_test/val_data.csv'):
    """Load train/val and cast string columns to pandas 'category' (aligned)."""
    import pandas as pd
    from pandas.api.types import CategoricalDtype

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)

    # Drop non-features
    drop_cols = ['winner', 'fighter_a', 'fighter_b', 'date', 'current_fight_date']
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols].copy()
    y_train = train_df['winner'].copy()
    X_val   = val_df[feature_cols].copy()
    y_val   = val_df['winner'].copy()

    # Cast all object columns to categorical with shared category sets
    obj_cols = list(X_train.select_dtypes(include='object').columns)
    for c in obj_cols:
        cats = sorted(set(X_train[c].astype('string').fillna('<NA>')) |
                      set(X_val[c].astype('string').fillna('<NA>')))
        cat_dtype = CategoricalDtype(categories=cats, ordered=False)
        X_train[c] = X_train[c].astype('string').fillna('<NA>').astype(cat_dtype)
        X_val[c]   = X_val[c].astype('string').fillna('<NA>').astype(cat_dtype)

    # Sanity: no leftover object dtype
    assert not list(X_train.select_dtypes(include='object').columns), "object dtype remains in X_train"
    assert not list(X_val.select_dtypes(include='object').columns),   "object dtype remains in X_val"

    print(f"Data loaded: Train {X_train.shape}, Val {X_val.shape}")
    if obj_cols:
        print(f"Categorical cols: {obj_cols}")
    return X_train, X_val, y_train, y_val


def optimize_hyperparams(X_train, y_train, X_val, y_val, n_trials=50):
    """Find best hyperparameters with Optuna"""

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
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        val_pred = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_pred)

        # Check overfitting
        train_pred = model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_pred)
        auc_diff = train_auc - val_auc

        # Penalize overfitting
        if auc_diff > 0.15:
            return val_auc - (auc_diff - 0.15) * 2

        return val_auc

    print(f"\nOptimizing hyperparameters ({n_trials} trials)...")
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Best AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    return study.best_params


def train_final_model(X_train, y_train, X_val, y_val, params):
    """Train XGBoost model with best parameters"""

    # Add fixed params
    params.update({
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'device': 'cpu',
        'n_estimators': 1000,
        'early_stopping_rounds': 50,  # Moved here for newer XGBoost
        'random_state': 42,
    })

    print("\nTraining final model...")
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    return model


def evaluate_model(model, X_train, y_train, X_val, y_val):
    """Calculate and print metrics"""
    train_pred = model.predict_proba(X_train)[:, 1]
    val_pred = model.predict_proba(X_val)[:, 1]

    metrics = {
        'train_acc': accuracy_score(y_train, (train_pred > 0.5).astype(int)),
        'train_auc': roc_auc_score(y_train, train_pred),
        'val_acc': accuracy_score(y_val, (val_pred > 0.5).astype(int)),
        'val_auc': roc_auc_score(y_val, val_pred),
    }
    metrics['auc_diff'] = metrics['train_auc'] - metrics['val_auc']

    print(f"\n{'=' * 40}")
    print(f"FINAL PERFORMANCE:")
    print(f"{'=' * 40}")
    print(f"Train: {metrics['train_acc']:.4f} acc, {metrics['train_auc']:.4f} auc")
    print(f"Val:   {metrics['val_acc']:.4f} acc, {metrics['val_auc']:.4f} auc")
    print(f"Overfit: {metrics['auc_diff']:.4f} auc difference")

    # Check requirements
    if metrics['val_acc'] >= 0.60:
        print(f"✓ Accuracy requirement met (>= 0.60)")
    else:
        print(f"✗ Accuracy below requirement (< 0.60)")

    if metrics['auc_diff'] <= 0.15:
        print(f"✓ Overfitting controlled (<= 0.15)")
    else:
        print(f"✗ Overfitting too high (> 0.15)")

    return metrics


def save_model(model, metrics, params, save_dir='../../../saved_models/xgboost/'):
    """Save model and metadata"""
    import os
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save model
    model_path = f"{save_dir}/model_{timestamp}.json"
    model.save_model(model_path)

    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'metrics': metrics,
        'params': params,
        'best_n_trees': model.best_iteration,
    }

    with open(f"{save_dir}/metadata_{timestamp}.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved: {model_path}")
    return model_path


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_xgboost(n_trials=50):
    """
    Complete training pipeline:
    1. Load data
    2. Optimize hyperparameters with Optuna
    3. Train final model
    4. Evaluate
    5. Save if good enough
    """

    print("=" * 50)
    print("XGBoost Training Pipeline")
    print("=" * 50)

    # Load data
    X_train, X_val, y_train, y_val = load_data()

    # Optimize hyperparameters
    best_params = optimize_hyperparams(X_train, y_train, X_val, y_val, n_trials=n_trials)

    # Train final model with best params
    model = train_final_model(X_train, y_train, X_val, y_val, best_params)

    # Evaluate
    metrics = evaluate_model(model, X_train, y_train, X_val, y_val)

    # Save if meets requirements
    if metrics['val_acc'] >= 0.60 and metrics['auc_diff'] <= 0.15:
        save_model(model, metrics, best_params)
        print("\n✓ Model meets all requirements and was saved!")
    else:
        print("\n✗ Model doesn't meet requirements. Consider:")
        print("  - More Optuna trials")
        print("  - Different feature engineering")
        print("  - More training data")

    return model, metrics


# ============================================================================
# RUN IT
# ============================================================================

if __name__ == "__main__":
    # Train with 50 Optuna trials (increase to 100+ for better results)
    model, metrics = train_xgboost(n_trials=50)
