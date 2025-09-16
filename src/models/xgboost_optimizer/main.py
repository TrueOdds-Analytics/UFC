"""
UFC Fight Prediction - Simple XGBoost Training (with Nested CV option)
Just the essentials: load, optimize/evaluate via nested CV, optionally save selected folds
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

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

warnings.filterwarnings('ignore')

ACC_THRESHOLD = 0.60  # save models at/above this outer-fold accuracy
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


def load_data_for_cv(train_path='../../../data/train_test/train_data.csv',
                     val_path='../../../data/train_test/val_data.csv'):
    """Combine train+val for nested CV and cast strings to pandas 'category'."""
    from pandas.api.types import CategoricalDtype

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    df = pd.concat([train_df, val_df], ignore_index=True)

    drop_cols = ['winner', 'fighter_a', 'fighter_b', 'date', 'current_fight_date']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = df['winner'].copy()

    obj_cols = list(X.select_dtypes(include='object').columns)
    for c in obj_cols:
        cats = sorted(X[c].astype('string').fillna('<NA>').unique())
        cat_dtype = CategoricalDtype(categories=cats, ordered=False)
        X[c] = X[c].astype('string').fillna('<NA>').astype(cat_dtype)

    print(f"Combined data for nested CV: {X.shape}")
    if obj_cols:
        print(f"Categorical cols: {obj_cols}")
    return X, y


def nested_cross_validation(X, y, outer_cv=5, inner_cv=3, optuna_trials=20, save_models=True):
    """
    Nested CV: inner loop tunes hyperparams (Optuna), outer loop evaluates on unseen fold.
    Uses a small validation split from the outer-train for early stopping to avoid leakage.
    """
    print("\n" + "=" * 50)
    print(f"Nested CV: outer={outer_cv} folds, inner={inner_cv} folds, trials={optuna_trials}")
    print("=" * 50)

    outer_skf = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42)

    outer_accs, outer_aucs = [], []
    train_accs, train_aucs = [], []

    best_params_per_fold = []
    best_n_per_fold = []

    fold_idx = 0
    for tr_idx, te_idx in outer_skf.split(X, y):
        fold_idx += 1
        print(f"\n--- Outer Fold {fold_idx}/{outer_cv} ---")

        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        inner_skf = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=42)

        def inner_objective(trial):
            params = {
                'objective': 'binary:logistic',
                'tree_method': 'hist',
                'device': 'cpu',
                'enable_categorical': True,
                'n_estimators': 400,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0.01, 1.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
                'early_stopping_rounds': 30
            }

            fold_aucs = []
            for in_tr_idx, in_va_idx in inner_skf.split(X_tr, y_tr):
                X_in_tr, X_in_va = X_tr.iloc[in_tr_idx], X_tr.iloc[in_va_idx]
                y_in_tr, y_in_va = y_tr.iloc[in_tr_idx], y_tr.iloc[in_va_idx]

                model = xgb.XGBClassifier(**params)
                model.fit(X_in_tr, y_in_tr, eval_set=[(X_in_va, y_in_va)], verbose=False)
                proba = model.predict_proba(X_in_va)[:, 1]
                fold_aucs.append(roc_auc_score(y_in_va, proba))

            return float(np.mean(fold_aucs))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(inner_objective, n_trials=optuna_trials)

        best_params = study.best_params
        best_params_per_fold.append(best_params)
        print(f"  Best inner AUC: {study.best_value:.4f}")
        # ---- Final fit on outer-train (no test in eval_set) ----
        final_params = {
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'device': 'cpu',
            'enable_categorical': True,
            'n_estimators': 800,
            'early_stopping_rounds': 40,
            **best_params
        }

        # small validation split from outer-train for early stopping
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        tr2_idx, va2_idx = next(sss.split(X_tr, y_tr))
        X_tr2, X_va2 = X_tr.iloc[tr2_idx], X_tr.iloc[va2_idx]
        y_tr2, y_va2 = y_tr.iloc[tr2_idx], y_tr.iloc[va2_idx]

        final_model = xgb.XGBClassifier(**final_params)
        final_model.fit(X_tr2, y_tr2, eval_set=[(X_va2, y_va2)], verbose=False)

        # refit on full outer-train with best n_estimators (no early stopping)
        best_n = getattr(final_model, "best_iteration", None)
        if best_n is None:
            best_n = final_model.get_params().get('n_estimators', 800)
        best_n_per_fold.append(int(best_n))

        final_model = xgb.XGBClassifier(**{**final_params, 'n_estimators': int(best_n), 'early_stopping_rounds': None})
        final_model.fit(X_tr, y_tr, verbose=False)

        # evaluate on outer test
        te_proba = final_model.predict_proba(X_te)[:, 1]
        te_acc = accuracy_score(y_te, (te_proba > 0.5).astype(int))
        te_auc = roc_auc_score(y_te, te_proba)

        tr_proba = final_model.predict_proba(X_tr)[:, 1]
        tr_acc = accuracy_score(y_tr, (tr_proba > 0.5).astype(int))
        tr_auc = roc_auc_score(y_tr, tr_proba)

        outer_accs.append(te_acc); outer_aucs.append(te_auc)
        train_accs.append(tr_acc);  train_aucs.append(tr_auc)

        print(f"  Outer test ACC: {te_acc:.4f} | AUC: {te_auc:.4f}")
        print(f"  Outer train ACC: {tr_acc:.4f} | AUC: {tr_auc:.4f}")

        # optionally save good outer-fold models
        if save_models and te_acc >= ACC_THRESHOLD:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = f"{SAVE_DIR}/nested_fold{fold_idx}_acc{te_acc:.3f}_{timestamp}.json"
            final_model.save_model(model_path)
            with open(f"{SAVE_DIR}/nested_fold{fold_idx}_metadata_{timestamp}.json", 'w') as f:
                json.dump({
                    'params': {**final_params, 'n_estimators': int(best_n), 'early_stopping_rounds': None},
                    'metrics': {
                        'train_acc': float(tr_acc),
                        'train_auc': float(tr_auc),
                        'test_acc': float(te_acc),
                        'test_auc': float(te_auc)
                    }
                }, f, indent=2)
            print(f"✓ Saved outer fold {fold_idx} | acc={te_acc:.3f}, auc={te_auc:.3f}")

    print("\n" + "=" * 50)
    print("Nested CV Results (Unbiased Estimate)")
    print("=" * 50)
    print(f"Test Accuracy:  {np.mean(outer_accs):.4f} ± {np.std(outer_accs):.4f}")
    print(f"Test AUC:       {np.mean(outer_aucs):.4f} ± {np.std(outer_aucs):.4f}")
    print(f"Train Accuracy: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
    print(f"Train AUC:      {np.mean(train_aucs):.4f} ± {np.std(train_aucs):.4f}")

    return {
        'outer_accs': outer_accs,
        'outer_aucs': outer_aucs,
        'train_accs': train_accs,
        'train_aucs': train_aucs,
        'best_params_per_fold': best_params_per_fold,
        'best_n_per_fold': best_n_per_fold
    }


def train_xgboost_nested(optuna_trials=20, outer_cv=5, inner_cv=3, save_models=True):
    """Main pipeline: nested CV (tunes inside, evaluates outside)."""
    print("=" * 50)
    print("XGBoost Training - Nested Cross-Validation")
    print("=" * 50)
    X, y = load_data_for_cv()
    results = nested_cross_validation(X, y, outer_cv=outer_cv, inner_cv=inner_cv,
                                      optuna_trials=optuna_trials, save_models=save_models)
    # Optionally write a summary file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f"{SAVE_DIR}/nested_cv_summary_{timestamp}.json", 'w') as f:
        json.dump({
            'mean_test_acc': float(np.mean(results['outer_accs'])),
            'std_test_acc': float(np.std(results['outer_accs'])),
            'mean_test_auc': float(np.mean(results['outer_aucs'])),
            'std_test_auc': float(np.std(results['outer_aucs'])),
            'best_params_per_fold': results['best_params_per_fold'],
            'best_n_per_fold': results['best_n_per_fold']
        }, f, indent=2)
    print("✓ Nested CV summary saved")


# ============================================================================

if __name__ == "__main__":
    # Run nested CV instead of single train/val Optuna
    train_xgboost_nested(optuna_trials=20, outer_cv=5, inner_cv=3, save_models=True)
