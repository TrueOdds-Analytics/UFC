import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import shap
import threading
import time
import sys

# Global variables
should_pause = False
best_accuracy = 0
best_auc = 0
best_auc_diff = 0


def user_input_thread():
    global should_pause
    while True:
        user_input = input("Press 'p' to pause/resume or 'q' to quit: ")
        if user_input.lower() == 'p':
            should_pause = not should_pause
            print("Paused" if should_pause else "Resumed")
        elif user_input.lower() == 'q':
            print("Quitting...")
            should_pause = True
            break


def get_train_val_data(golden_features=[]):
    train_data = pd.read_csv('../data/train test data/train_data.csv')
    val_data = pd.read_csv('../data/train test data/val_data.csv')

    train_labels = train_data['winner']
    val_labels = val_data['winner']
    train_data = train_data.drop(['winner'], axis=1)
    val_data = val_data.drop(['winner'], axis=1)
    columns_to_drop = ['fighter', 'fighter_b', 'fight_date', 'current_fight_date']
    train_data = train_data.drop(columns=columns_to_drop)
    val_data = val_data.drop(columns=columns_to_drop)

    # Shuffle data
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    train_labels = train_labels.sample(frac=1, random_state=42).reset_index(drop=True)
    val_data = val_data.sample(frac=1, random_state=42).reset_index(drop=True)
    val_labels = val_labels.sample(frac=1, random_state=42).reset_index(drop=True)

    # Convert specified columns to category type
    category_columns = [
        'result_fight_1', 'winner_fight_1', 'weight_class_fight_1', 'scheduled_rounds_fight_1',
        'result_b_fight_1', 'winner_b_fight_1', 'weight_class_b_fight_1', 'scheduled_rounds_b_fight_1',
        'result_fight_2', 'winner_fight_2', 'weight_class_fight_2', 'scheduled_rounds_fight_2',
        'result_b_fight_2', 'winner_b_fight_2', 'weight_class_b_fight_2', 'scheduled_rounds_b_fight_2',
        'result_fight_3', 'winner_fight_3', 'weight_class_fight_3', 'scheduled_rounds_fight_3',
        'result_b_fight_3', 'winner_b_fight_3', 'weight_class_b_fight_3', 'scheduled_rounds_b_fight_3'
    ]

    for df in [train_data, val_data]:
        for col in category_columns:
            df[col] = df[col].astype(str).fillna('Unknown')

    # Create a list for per-feature quantization
    per_float_feature_quantization = []
    for feature in golden_features:
        if feature in train_data.columns:
            feature_index = train_data.columns.get_loc(feature)
            per_float_feature_quantization.append(f"{feature_index}:border_count=1024")

    return train_data, val_data, train_labels, val_labels, per_float_feature_quantization


def plot_losses(train_losses, val_losses, train_auc, val_auc, features_removed, accuracy, auc):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Number of iterations')
    ax1.set_ylabel('Log Loss')
    ax1.set_title(
        f'Learning Curves - Log Loss (Features removed: {features_removed}, Val Acc: {accuracy:.4f}, Val AUC: {auc:.4f})')
    ax1.legend()
    ax1.grid()

    ax2.plot(train_auc, label='Train AUC')
    ax2.plot(val_auc, label='Validation AUC')
    ax2.set_xlabel('Number of iterations')
    ax2.set_ylabel('AUC')
    ax2.set_title(
        f'Learning Curves - AUC (Features removed: {features_removed}, Val Acc: {accuracy:.4f}, Val AUC: {auc:.4f})')
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.show()


def create_shap_graph(model_path, X_train):
    model = CatBoostClassifier()
    model.load_model(model_path)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
    feature_importance = sorted(zip(X_train.columns, mean_abs_shap_values), key=lambda x: x[1], reverse=True)

    top_50_features = feature_importance[:50]
    print("Most influential features:")
    for feature, importance in top_50_features:
        print(f"{feature}: {importance}")

    top_50_df = pd.DataFrame(top_50_features, columns=["Feature", "Importance"])

    plt.figure(figsize=(12, 20))
    plt.barh(top_50_df["Feature"], top_50_df["Importance"], color='blue')
    plt.xlabel('Mean Absolute SHAP Value')
    plt.ylabel('Features')
    plt.title("SHAP Values for CatBoost model")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def create_fit_and_evaluate_model(params, X_train, X_val, y_train, y_val):
    cat_features = X_train.select_dtypes(include=['category']).columns.tolist()
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    params['eval_metric'] = 'Logloss'

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, verbose=False)

    results = model.get_evals_result()
    train_losses = results['learn']['Logloss']
    val_losses = results['validation']['Logloss']

    y_val_pred = model.predict(val_pool)
    y_val_pred_proba = model.predict_proba(val_pool)[:, 1]
    accuracy = accuracy_score(y_val, y_val_pred)

    train_auc = roc_auc_score(y_train, model.predict_proba(train_pool)[:, 1])
    val_auc = roc_auc_score(y_val, y_val_pred_proba)

    return model, accuracy, val_auc, train_losses, val_losses, [train_auc], [val_auc]


def adjust_hyperparameter_ranges(study, num_best_trials=20):
    trials = sorted(study.trials, key=lambda t: t.value, reverse=True)
    best_trials = trials[:num_best_trials]

    new_ranges = {}
    for param_name in study.best_params.keys():
        values = [t.params[param_name] for t in best_trials if param_name in t.params]
        if values:
            if isinstance(study.best_params[param_name], int):
                new_ranges[param_name] = {
                    'type': 'int',
                    'low': max(min(values) - 1, 1),
                    'high': min(max(values) + 1, 10)
                }
            elif isinstance(study.best_params[param_name], float):
                new_ranges[param_name] = {
                    'type': 'float',
                    'low': max(min(values) * 0.8, 0.00001),
                    'high': min(max(values) * 1.2, 150.0)
                }
            else:  # For categorical parameters
                new_ranges[param_name] = {
                    'type': 'categorical',
                    'choices': list(set(values))
                }

    return new_ranges


def objective(trial, X_train, X_val, y_train, y_val, per_float_feature_quantization, params=None):
    global should_pause, best_accuracy, best_auc_diff
    while should_pause:
        time.sleep(1)

    if params is None:
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 1, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 25, 35, log=True),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
            'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
        }
        if per_float_feature_quantization:
            params['per_float_feature_quantization'] = per_float_feature_quantization
        if params['bootstrap_type'] == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 10.0)

    params.update({
        'iterations': 1000,
        'eval_metric': 'Logloss',
        'use_best_model': True,
        'early_stopping_rounds': 250,
        'task_type': 'GPU',
        'devices': '0',
        'verbose': False,
        'per_float_feature_quantization': per_float_feature_quantization
    })

    cat_features = X_train.select_dtypes(include=['category']).columns.tolist()
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    model, accuracy, auc, train_losses, val_losses, train_auc, val_auc = create_fit_and_evaluate_model(params, X_train,
                                                                                                       X_val, y_train,
                                                                                                       y_val)

    auc_diff = abs(train_auc[-1] - val_auc[-1])

    if accuracy > 0.65 and (auc_diff < 0.10):
        best_accuracy = accuracy
        best_auc_diff = auc_diff
        model_filename = f'models/catboost/model_{accuracy:.4f}_{len(X_train.columns)}_features_auc_diff_{auc_diff:.4f}.cbm'
        model.save_model(model_filename)
        plot_losses(train_losses, val_losses, train_auc, val_auc, len(X_train.columns), accuracy,
                    auc if auc is not None else 0)

    return accuracy


def optimize_model(X_train, X_val, y_train, y_val, per_float_feature_quantization, n_rounds=1,
                   n_trials_per_round=10000):
    global best_accuracy, best_auc

    for round in range(n_rounds):
        print(f"Starting optimization round {round + 1}/{n_rounds}")

        if round == 0:
            study = optuna.create_study(direction='maximize', sampler=TPESampler(), pruner=MedianPruner())
            study.optimize(
                lambda trial: objective(trial, X_train, X_val, y_train, y_val, per_float_feature_quantization),
                n_trials=n_trials_per_round)
        else:
            new_ranges = adjust_hyperparameter_ranges(study)

            def objective_with_new_ranges(trial):
                params = {}
                for k, v in new_ranges.items():
                    if v['type'] == 'int':
                        params[k] = trial.suggest_int(k, v['low'], v['high'])
                    elif v['type'] == 'float':
                        params[k] = trial.suggest_float(k, max(v['low'], 1e-5), min(v['high'], 100.0), log=True)
                    elif v['type'] == 'categorical':
                        params[k] = trial.suggest_categorical(k, v['choices'])
                return objective(trial, X_train, X_val, y_train, y_val, per_float_feature_quantization, params)

            study = optuna.create_study(direction='maximize', sampler=TPESampler(), pruner=MedianPruner())
            study.optimize(objective_with_new_ranges, n_trials=n_trials_per_round)

        print(f"Round {round + 1} best accuracy: {study.best_value:.4f}")
        print(f"Round {round + 1} best parameters: {study.best_params}")

    return study.best_trial


def get_top_features_and_retrain(model_path, X_train, X_val, y_train, y_val, n_features,
                                 per_float_feature_quantization):
    model = CatBoostClassifier()
    model.load_model(model_path)

    feature_importance = model.get_feature_importance()
    feature_names = X_train.columns
    feature_importance = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)

    top_n_features = [feature for feature, _ in feature_importance[:n_features]]

    print(f"Top {n_features} features:")
    for feature, importance in feature_importance[:n_features]:
        print(f"{feature}: {importance}")

    X_train_filtered = X_train[top_n_features]
    X_val_filtered = X_val[top_n_features]

    print(f"\nRetraining model with top {n_features} features...")
    best_trial = optimize_model(X_train_filtered, X_val_filtered, y_train, y_val, per_float_feature_quantization)

    return best_trial.params, best_trial.value


if __name__ == "__main__":
    input_thread = threading.Thread(target=user_input_thread, daemon=True)
    input_thread.start()

    # Specify your golden features
    golden_features = ['current_fight_open_odds_diff', 'total_strikes_landed_diff_fighter_avg_last_3']

    X_train, X_val, y_train, y_val, per_float_feature_quantization = get_train_val_data(golden_features)
    print("Starting initial optimization and evaluation...")

    best_trial = optimize_model(X_train, X_val, y_train, y_val, per_float_feature_quantization)
    best_params = best_trial.params
    best_accuracy = best_trial.value

    # Create and evaluate the best model
    best_model, accuracy, auc, _, _, _, _ = create_fit_and_evaluate_model(best_params, X_train, X_val, y_train, y_val)
    best_auc = auc

