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
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / 'data' / 'train_test'
SAVE_DIR = PROJECT_ROOT / 'saved_models' / 'xgboost' / 'trials'
ACC_THRESHOLD = 0.60     # validation accuracy threshold for saving
LOSS_GAP_THRESHOLD = 0.05  # train_loss - val_loss must be <= this value
VAL_ACC_SAVE_THRESHOLD = 0.60  # save models at/above this validation accuracy
LOSS_GAP_THRESHOLD = 0.05      # only save when |train_loss - val_loss| <= threshold
SAVE_DIR = '../../../saved_models/xgboost/trials/'
os.makedirs(SAVE_DIR, exist_ok=True)
TRIAL_PLOTS_DIR = os.path.join(SAVE_DIR, 'trial_plots')
os.makedirs(TRIAL_PLOTS_DIR, exist_ok=True)

SAVE_DIR.mkdir(parents=True, exist_ok=True)
TRIAL_PLOTS_DIR = SAVE_DIR / 'trial_plots'
TRIAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(train_path: str | os.PathLike | None = None,
              val_path: str | os.PathLike | None = None):
    """Load train/val and cast string columns to pandas 'category' (aligned)."""
    from pandas.api.types import CategoricalDtype

    train_path = Path(train_path) if train_path is not None else DATA_DIR / 'train_data.csv'
    val_path = Path(val_path) if val_path is not None else DATA_DIR / 'val_data.csv'

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


def load_data_for_cv(train_path: str | os.PathLike | None = None,
                     val_path: str | os.PathLike | None = None):
    """Combine train+val for nested CV and cast strings to pandas 'category'."""
    from pandas.api.types import CategoricalDtype

    train_path = Path(train_path) if train_path is not None else DATA_DIR / 'train_data.csv'
    val_path = Path(val_path) if val_path is not None else DATA_DIR / 'val_data.csv'

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


def plot_trial_metrics(train_logloss_curves, val_logloss_curves,
                       train_error_curves, val_error_curves,
                       plot_path, trial_number, outer_fold):


    if not train_logloss_curves or not val_logloss_curves:
        return


    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Optuna Trial {trial_number} - Outer Fold {outer_fold}")

    for idx, curve in enumerate(train_logloss_curves):
        rounds = range(1, len(curve) + 1)
        axes[0].plot(rounds, curve, color='tab:blue', alpha=0.4,
                     label='Train Logloss' if idx == 0 else None)
    for idx, curve in enumerate(val_logloss_curves):
        rounds = range(1, len(curve) + 1)
        axes[0].plot(rounds, curve, color='tab:orange', alpha=0.7,
                     label='Validation Logloss' if idx == 0 else None)
    max_len = max(len(curve) for curve in train_logloss_curves + val_logloss_curves)

    def _mean_curve(curves, transform=None):
        arr = np.full((len(curves), max_len), np.nan, dtype=float)
        for idx, curve in enumerate(curves):
            values = np.asarray(curve, dtype=float)
            if transform is not None:
                values = transform(values)
            arr[idx, :len(values)] = values
        return np.nanmean(arr, axis=0)

    mean_train_loss = _mean_curve(train_logloss_curves)
    mean_val_loss = _mean_curve(val_logloss_curves)
    mean_train_acc = _mean_curve(train_error_curves, transform=lambda x: 1.0 - x)
    mean_val_acc = _mean_curve(val_error_curves, transform=lambda x: 1.0 - x)

    iterations = np.arange(1, max_len + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Optuna Trial {trial_number} - Outer Fold {outer_fold}")

    axes[0].plot(iterations, mean_train_loss, label='Train Logloss')
    axes[0].plot(iterations, mean_val_loss, label='Validation Logloss')
    axes[0].set_xlabel('Boosting Rounds')
    axes[0].set_ylabel('Log Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()

    for idx, curve in enumerate(train_error_curves):
        rounds = range(1, len(curve) + 1)
        train_acc = [1.0 - value for value in curve]
        axes[1].plot(rounds, train_acc, color='tab:green', alpha=0.4,
                     label='Train Accuracy' if idx == 0 else None)
    for idx, curve in enumerate(val_error_curves):
        rounds = range(1, len(curve) + 1)
        val_acc = [1.0 - value for value in curve]
        axes[1].plot(rounds, val_acc, color='tab:red', alpha=0.7,
                     label='Validation Accuracy' if idx == 0 else None)
    axes[1].plot(iterations, mean_train_acc, label='Train Accuracy')
    axes[1].plot(iterations, mean_val_acc, label='Validation Accuracy')
    axes[1].set_xlabel('Boosting Rounds')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


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

        fold_plot_dir = TRIAL_PLOTS_DIR / f'outer_fold_{fold_idx:02d}'
        fold_plot_dir.mkdir(parents=True, exist_ok=True)

        fold_plot_dir = os.path.join(TRIAL_PLOTS_DIR, f'outer_fold_{fold_idx:02d}')
        os.makedirs(fold_plot_dir, exist_ok=True)

        def inner_objective(trial):
            params = {
                'objective': 'binary:logistic',
                'tree_method': 'hist',
                'device': 'cpu',
                'enable_categorical': True,
                'n_estimators': 400,
                'eval_metric': ['logloss', 'error'],
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
            train_logloss_curves, val_logloss_curves = [], []
            train_error_curves, val_error_curves = [], []
            for in_tr_idx, in_va_idx in inner_skf.split(X_tr, y_tr):
                X_in_tr, X_in_va = X_tr.iloc[in_tr_idx], X_tr.iloc[in_va_idx]
                y_in_tr, y_in_va = y_tr.iloc[in_tr_idx], y_tr.iloc[in_va_idx]

                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_in_tr,
                    y_in_tr,
                    eval_set=[(X_in_tr, y_in_tr), (X_in_va, y_in_va)],
                    verbose=False
                )
                proba = model.predict_proba(X_in_va)[:, 1]
                fold_aucs.append(roc_auc_score(y_in_va, proba))

                evals_result = model.evals_result()
                train_logloss_curves.append(evals_result['validation_0']['logloss'])
                val_logloss_curves.append(evals_result['validation_1']['logloss'])
                train_error_curves.append(evals_result['validation_0']['error'])
                val_error_curves.append(evals_result['validation_1']['error'])

            plot_path = fold_plot_dir / f"trial_{trial.number:03d}_metrics.png"
            plot_path = os.path.join(
                fold_plot_dir,
                f"trial_{trial.number:03d}_metrics.png"
            )
            plot_trial_metrics(
                train_logloss_curves,
                val_logloss_curves,
                train_error_curves,
                val_error_curves,
                plot_path,
                trial.number,
                fold_idx
            )

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
            'eval_metric': ['logloss', 'error'],
            'early_stopping_rounds': 40,
            **best_params
        }

        # small validation split from outer-train for early stopping
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        tr2_idx, va2_idx = next(sss.split(X_tr, y_tr))
        X_tr2, X_va2 = X_tr.iloc[tr2_idx], X_tr.iloc[va2_idx]
        y_tr2, y_va2 = y_tr.iloc[tr2_idx], y_tr.iloc[va2_idx]

        final_model = xgb.XGBClassifier(**final_params)
        final_model.fit(
            X_tr2,
            y_tr2,
            eval_set=[(X_tr2, y_tr2), (X_va2, y_va2)],
            verbose=False
        )

        evals_result = final_model.evals_result()
        train_loss_curve = evals_result['validation_0']['logloss']
        val_loss_curve = evals_result['validation_1']['logloss']
        train_error_curve = evals_result['validation_0']['error']
        val_error_curve = evals_result['validation_1']['error']

        best_iteration = getattr(final_model, "best_iteration", None)
        metric_index = int(best_iteration) if best_iteration is not None else len(val_loss_curve) - 1
        train_loss_at_best = train_loss_curve[metric_index]
        val_loss_at_best = val_loss_curve[metric_index]
        train_acc_at_best = 1.0 - train_error_curve[metric_index]
        val_acc_at_best = 1.0 - val_error_curve[metric_index]
        loss_gap = train_loss_at_best - val_loss_at_best

        # refit on full outer-train with best n_estimators (no early stopping)
        best_n = metric_index + 1 if best_iteration is not None else final_model.get_params().get('n_estimators', 800)
        best_n_per_fold.append(int(best_n))

        final_model = xgb.XGBClassifier(**{**final_params, 'n_estimators': int(best_n), 'early_stopping_rounds': None})
        loss_gap = abs(train_loss_at_best - val_loss_at_best)

        # refit on full outer-train with best n_estimators (no early stopping)
        best_n_trees = (
            int(best_iteration) + 1
            if best_iteration is not None
            else final_model.get_params().get('n_estimators', 800)
        )
        best_n_per_fold.append(int(best_n_trees))

        final_model = xgb.XGBClassifier(
            **{**final_params, 'n_estimators': int(best_n_trees), 'early_stopping_rounds': None}
        )
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
        print(f"  Holdout val ACC: {val_acc_at_best:.4f} | Loss gap: {loss_gap:.4f}")

        # optionally save good outer-fold models
        if (
            save_models
            and val_acc_at_best >= ACC_THRESHOLD
            and val_acc_at_best >= VAL_ACC_SAVE_THRESHOLD
            and loss_gap <= LOSS_GAP_THRESHOLD
        ):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = SAVE_DIR / f"nested_fold{fold_idx}_acc{te_acc:.3f}_{timestamp}.json"
            final_model.save_model(str(model_path))
            metadata_path = SAVE_DIR / f"nested_fold{fold_idx}_metadata_{timestamp}.json"
            with metadata_path.open('w') as f:
                json.dump({
                    'params': {**final_params, 'n_estimators': int(best_n_trees), 'early_stopping_rounds': None},
                    'metrics': {
                        'train_holdout_acc': float(train_acc_at_best),
                        'train_holdout_loss': float(train_loss_at_best),
                        'val_holdout_acc': float(val_acc_at_best),
                        'val_holdout_loss': float(val_loss_at_best),
                        'loss_gap': float(loss_gap),
                        'loss_gap_abs': float(loss_gap),
                        'train_acc': float(tr_acc),
                        'train_auc': float(tr_auc),
                        'test_acc': float(te_acc),
                        'test_auc': float(te_auc)
                    }
                }, f, indent=2)
            print(
                f"✓ Saved outer fold {fold_idx} | val_acc={val_acc_at_best:.3f}, "
                f"loss_gap={loss_gap:.3f}"
            )

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
    summary_path = SAVE_DIR / f"nested_cv_summary_{timestamp}.json"
    with summary_path.open('w') as f:
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
