import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import xgboost as xgb
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


def get_train_val_data():
    train_data = pd.read_csv('data/train test data/train_data.csv')
    val_data = pd.read_csv('data/train test data/val_data.csv')

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
        df[category_columns] = df[category_columns].astype("category")

    return train_data, val_data, train_labels, val_labels


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
    # Load the model from the file
    model = xgb.XGBClassifier(enable_categorical=True)
    model.load_model(model_path)

    # Create the SHAP explainer
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
    plt.title("SHAP Values for XGBoost model")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def create_fit_and_evaluate_model(params, X_train, X_val, y_train, y_val):
    params['enable_categorical'] = True
    params['eval_metric'] = ['logloss', 'auc']  # Add this line

    model = xgb.XGBClassifier(**params)
    eval_set = [(X_train, y_train), (X_val, y_val)]

    # Remove eval_metric from fit method
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    results = model.evals_result()
    train_losses = results['validation_0']['logloss']
    val_losses = results['validation_1']['logloss']

    # Safely get AUC values
    train_auc = results['validation_0'].get('auc', [])
    val_auc = results['validation_1'].get('auc', [])

    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)

    # Calculate AUC manually
    try:
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    except ValueError as e:
        print(f"Error calculating AUC: {str(e)}")
        print(f"Unique values in y_val: {np.unique(y_val)}")
        print(f"Shape of predict_proba: {model.predict_proba(X_val).shape}")
        auc = None

    return model, accuracy, auc, train_losses, val_losses, train_auc, val_auc


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


def objective(trial, X_train, X_val, y_train, y_val, params=None):
    global should_pause, best_accuracy, best_auc_diff
    while should_pause:
        time.sleep(1)

    if params is None:
        params = {
            'lambda': trial.suggest_float('lambda', 25, 35, log=True),
            'alpha': trial.suggest_float('alpha', 25, 35, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 1, 10.0),
            'max_depth': trial.suggest_int('max_depth', 1, 6),
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-3, 10.0, log=True),
            'eta': trial.suggest_float('eta', 0.01, 0.1, log=True),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        }

    params.update({
        'tree_method': 'hist',
        'device': 'cuda',
        'objective': 'binary:logistic',
        'verbosity': 0,
        'n_jobs': -1,
        'n_estimators': 1000,
        'early_stopping_rounds': 250,
        'eval_metric': ['logloss', 'auc'],
        'enable_categorical': True
    })

    model, accuracy, auc, train_losses, val_losses, train_auc, val_auc = create_fit_and_evaluate_model(params, X_train,
                                                                                                       X_val, y_train,
                                                                                                       y_val)

    # Calculate the difference between train and validation AUC
    auc_diff = abs(train_auc[-1] - val_auc[-1])

    if accuracy > 0.66 and (auc_diff < 0.10):
        best_accuracy = accuracy
        best_auc_diff = auc_diff
        model_filename = f'models/xgboost/model_{accuracy:.4f}_{len(X_train.columns)}_features_auc_diff_{auc_diff:.4f}.json'
        model.save_model(model_filename)
        plot_losses(train_losses, val_losses, train_auc, val_auc, len(X_train.columns), accuracy,
                    auc if auc is not None else 0)

    return accuracy


def optimize_model(X_train, X_val, y_train, y_val, n_rounds=1, n_trials_per_round=10000):
    global best_accuracy, best_auc

    for round in range(n_rounds):
        print(f"Starting optimization round {round + 1}/{n_rounds}")

        if round == 0:
            study = optuna.create_study(direction='maximize', sampler=TPESampler(), pruner=MedianPruner())
            study.optimize(lambda trial: objective(trial, X_train, X_val, y_train, y_val), n_trials=n_trials_per_round)
        else:
            new_ranges = adjust_hyperparameter_ranges(study)

            def objective_with_new_ranges(trial):
                params = {}
                for k, v in new_ranges.items():
                    if v['type'] == 'int':
                        params[k] = trial.suggest_int(k, v['low'], v['high'])
                    elif v['type'] == 'float':
                        if k in ['subsample', 'colsample_bytree']:
                            params[k] = trial.suggest_float(k, max(v['low'], 0.1), min(v['high'], 1.0))
                        elif k == 'eta':
                            params[k] = trial.suggest_float(k, max(v['low'], 1e-5), min(v['high'], 1.0), log=True)
                        else:
                            params[k] = trial.suggest_float(k, max(v['low'], 1e-5), min(v['high'], 100.0), log=True)
                    elif v['type'] == 'categorical':
                        params[k] = trial.suggest_categorical(k, v['choices'])
                return objective(trial, X_train, X_val, y_train, y_val, params)

            study = optuna.create_study(direction='maximize', sampler=TPESampler(), pruner=MedianPruner())
            study.optimize(objective_with_new_ranges, n_trials=n_trials_per_round)

        print(f"Round {round + 1} best accuracy: {study.best_value:.4f}")
        print(f"Round {round + 1} best parameters: {study.best_params}")

    return study.best_trial


def get_top_features_and_retrain(model_path, X_train, X_val, y_train, y_val, n_features):
    # Load the model from the file
    model = xgb.XGBClassifier(enable_categorical=True)
    model.load_model(model_path)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
    feature_importance = sorted(zip(X_train.columns, mean_abs_shap_values), key=lambda x: x[1], reverse=True)

    top_n_features = [feature for feature, _ in feature_importance[:n_features]]

    print(f"Top {n_features} features:")
    for feature, importance in feature_importance[:n_features]:
        print(f"{feature}: {importance}")

    X_train_filtered = X_train[top_n_features]
    X_val_filtered = X_val[top_n_features]

    print(f"\nRetraining model with top {n_features} features...")
    best_trial = optimize_model(X_train_filtered, X_val_filtered, y_train, y_val)

    return best_trial.params, best_trial.value


if __name__ == "__main__":
    input_thread = threading.Thread(target=user_input_thread, daemon=True)
    input_thread.start()

    X_train, X_val, y_train, y_val = get_train_val_data()
    print("Starting initial optimization and evaluation...")
    try:
        best_trial = optimize_model(X_train, X_val, y_train, y_val)
        best_params = best_trial.params
        best_accuracy = best_trial.value

        # Create and evaluate the best model
        best_model, accuracy, auc, _, _, _, _ = create_fit_and_evaluate_model(best_params, X_train, X_val, y_train,
                                                                              y_val)
        best_auc = auc

        print("Initial optimization completed.")
        print(f"Best model validation accuracy: {best_accuracy:.4f}")
        print(f"Best model validation AUC: {best_auc:.4f}")
        print("Best parameters:", best_params)
        print("--------------------")

    except KeyboardInterrupt:
        print("Optimization interrupted by user.")

    # print("Creating SHAP graph for the best model")
    # X_train, X_val, y_train, y_val = get_train_val_data()
    # model_path = f'models/xgboost/model_0.7452_338_features.json'
    # create_shap_graph(model_path, X_train)
    # print("SHAP graph creation completed.")
    # print("--------------------")

    # n_features = 100
    #
    # X_train, X_val, y_train, y_val = get_train_val_data()
    # model_path = 'models/xgboost/model_0.6683_338_features.json'
    # retrained_best_params, retrained_accuracy = get_top_features_and_retrain(model_path, X_train, X_val, y_train, y_val,
    #                                                                          n_features)
    #
    # print("--------------------")
    # print("Retraining with top features completed.")
    # print(f"Retrained model validation accuracy: {retrained_accuracy:.4f}")
    # print("Retrained model best parameters:", retrained_best_params)
