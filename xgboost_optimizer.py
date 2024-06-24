import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import shap
import threading
import time

# Global flag for pausing
should_pause = False


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


def get_train_test_data():
    # Load train data from CSV
    train_data = pd.read_csv('data/train_data.csv')
    train_labels = train_data['winner']
    train_data = train_data.drop(['winner'], axis=1)

    # Shuffle train data
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    train_labels = train_labels.sample(frac=1, random_state=42).reset_index(drop=True)

    # Convert specified columns to category type
    category_columns = [
        'result_fight_1', 'winner_fight_1', 'weight_class_fight_1', 'scheduled_rounds_fight_1',
        'result_b_fight_1', 'winner_b_fight_1', 'weight_class_b_fight_1', 'scheduled_rounds_b_fight_1',
        'result_fight_2', 'winner_fight_2', 'weight_class_fight_2', 'scheduled_rounds_fight_2',
        'result_b_fight_2', 'winner_b_fight_2', 'weight_class_b_fight_2', 'scheduled_rounds_b_fight_2',
        'result_fight_3', 'winner_fight_3', 'weight_class_fight_3', 'scheduled_rounds_fight_3',
        'result_b_fight_3', 'winner_b_fight_3', 'weight_class_b_fight_3', 'scheduled_rounds_b_fight_3'
    ]

    train_data[category_columns] = train_data[category_columns].astype("category")

    # Load validation data from CSV
    val_data = pd.read_csv('data/val_data.csv')
    val_labels = val_data['winner']
    val_data = val_data.drop(['winner'], axis=1)

    # Shuffle validation data
    val_data = val_data.sample(frac=1, random_state=42).reset_index(drop=True)
    val_labels = val_labels.sample(frac=1, random_state=42).reset_index(drop=True)

    # Convert specified columns to category type
    val_data[category_columns] = val_data[category_columns].astype("category")

    return train_data, val_data, train_labels, val_labels


def plot_losses(train_losses, val_losses, train_auc, val_auc, features_removed, accuracy, auc):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Number of iterations')
    ax1.set_ylabel('Log Loss')
    ax1.set_title(
        f'Learning Curves - Log Loss (Features removed: {features_removed}, Acc: {accuracy:.4f}, AUC: {auc:.4f})')
    ax1.legend()
    ax1.grid()

    ax2.plot(train_auc, label='Train AUC')
    ax2.plot(val_auc, label='Validation AUC')
    ax2.set_xlabel('Number of iterations')
    ax2.set_ylabel('AUC')
    ax2.set_title(f'Learning Curves - AUC (Features removed: {features_removed}, Acc: {accuracy:.4f}, AUC: {auc:.4f})')
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.savefig(f'learning_curves_{features_removed}_features_removed.png')
    plt.close()


def calculate_shap_values(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    feature_names = X.columns
    shap.summary_plot(shap_values, X, plot_type="bar", feature_names=feature_names)
    return shap_values, feature_names


def train_and_evaluate_model(X_train, X_val, y_train, y_val, features_removed, all_removed_features, best_accuracy,
                             best_auc):
    global should_pause

    def objective(trial):
        global should_pause
        while should_pause:
            time.sleep(1)

        params = {
            'lambda': trial.suggest_float('lambda', 1e-1, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-1, 10.0, log=True),
            'tree_method': 'hist',
            'device': 'cuda',
            'objective': 'binary:logistic',
            'verbosity': 0,
            'n_jobs': -1,
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 5),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9, log=True),
            'gamma': trial.suggest_float('gamma', 0.1, 1.0, log=True),
            'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
            'seed': trial.suggest_int('seed', 1, 100),
            'n_estimators': 10000,
            'early_stopping_rounds': 100,
            'eval_metric': ['logloss', 'auc'],
            'enable_categorical': True
        }

        model = xgb.XGBClassifier(**params)
        eval_set = [(X_train, y_train), (X_val, y_val)]
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        return accuracy

    study = optuna.create_study(direction='maximize', sampler=TPESampler(), pruner=MedianPruner())
    study.optimize(objective, n_trials=1000)

    best_params = study.best_params
    best_params.update({
        'tree_method': 'hist',
        'device': 'cuda',
        'objective': 'binary:logistic',
        'verbosity': 0,
        'n_jobs': -1,
        'n_estimators': 10000,
        'early_stopping_rounds': 100,
        'eval_metric': ['logloss', 'auc'],
        'enable_categorical': True
    })

    model = xgb.XGBClassifier(**best_params)
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    results = model.evals_result()
    train_losses = results['validation_0']['logloss']
    val_losses = results['validation_1']['logloss']
    train_auc = results['validation_0']['auc']
    val_auc = results['validation_1']['auc']

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    if accuracy > best_accuracy or (accuracy == best_accuracy and auc > best_auc):
        best_accuracy = accuracy
        best_auc = auc
        model_filename = f'models/model_{accuracy:.4f}_{features_removed}_features_removed.json'
        model.save_model(model_filename)
        print(f"New best model saved: {model_filename}")

        plot_losses(train_losses, val_losses, train_auc, val_auc, features_removed, accuracy, auc)
        print(f"Learning curves saved as learning_curves_{features_removed}_features_removed.png")

        features_removed_filename = f'removed_features_{accuracy:.4f}_{features_removed}_features.txt'
        with open(features_removed_filename, 'w') as f:
            f.write(f"Total features removed: {features_removed}\n\n")
            f.write("Removed features:\n")
            for feature in all_removed_features:
                f.write(f"{feature}\n")
        print(f"Updated list of removed features saved to {features_removed_filename}")

    return model, accuracy, auc, best_params, train_losses, val_losses, train_auc, val_auc, best_accuracy, best_auc


if __name__ == "__main__":
    # Start the user input thread
    input_thread = threading.Thread(target=user_input_thread, daemon=True)
    input_thread.start()

    # X_train, X_val, y_train, y_val = get_train_test_data()
    # print("Starting initial optimization and evaluation...")
    # try:
    #     initial_model, initial_accuracy, initial_auc, initial_best_params, _, _, _, _, best_accuracy, best_auc = train_and_evaluate_model(
    #         X_train, X_val, y_train, y_val, 0, [], 0, 0
    #     )
    # except KeyboardInterrupt:
    #     print("Optimization interrupted by user.")
    #
    # print("Initial optimization completed.")
    # print(f"Best model accuracy: {best_accuracy:.4f}")
    # print(f"Best model AUC: {best_auc:.4f}")
    # print("Best parameters:", initial_best_params)
    # print("--------------------")

    model_path = 'models/model_0.7079.json'
    original_model = xgb.XGBClassifier(enable_categorical=True)
    original_model.load_model(model_path)

    X_train, X_val, y_train, y_val = get_train_test_data()

    initial_feature_count = X_train.shape[1]
    best_accuracy = 0
    best_auc = 0

    accuracies = []
    aucs = []
    features_removed_list = []
    all_removed_features = []

    original_shap_values, original_feature_names = calculate_shap_values(original_model, X_train)

    mean_shap_values = np.abs(original_shap_values).mean(axis=0)
    feature_importance = sorted(zip(original_feature_names, mean_shap_values), key=lambda x: x[1])

    print("Original SHAP value importances (sorted from least to most important):")
    for feature, importance in feature_importance:
        print(f"{feature}: {importance}")
    print("--------------------")

    print(f"Original number of features: {initial_feature_count}")

    try:
        while len(feature_importance) >= 5:
            while should_pause:
                time.sleep(1)

            features_to_remove = [feature for feature, _ in feature_importance[:5]]
            existing_features_to_remove = list(set(features_to_remove) & set(X_train.columns))

            X_train = X_train.drop(columns=existing_features_to_remove)
            X_val = X_val.drop(columns=existing_features_to_remove)

            features_removed = initial_feature_count - X_train.shape[1]
            features_removed_list.append(features_removed)
            all_removed_features.extend(existing_features_to_remove)
            print(f"Current number of features: {X_train.shape[1]}")
            print(f"Removed features: {existing_features_to_remove}")

            new_model, new_accuracy, new_auc, new_best_params, _, _, _, _, best_accuracy, best_auc = train_and_evaluate_model(
                X_train, X_val, y_train, y_val, features_removed, all_removed_features, best_accuracy, best_auc)

            accuracies.append(new_accuracy)
            aucs.append(new_auc)

            print(f"Current accuracy: {new_accuracy:.4f}, Current AUC: {new_auc:.4f}")
            print(f"Best accuracy: {best_accuracy:.4f}, Best AUC: {best_auc:.4f}")
            print(f"Best parameters: {new_best_params}")
            print("--------------------")

            feature_importance = [item for item in feature_importance if item[0] not in existing_features_to_remove]

        if feature_importance:
            while should_pause:
                time.sleep(1)

            remaining_features = [feature for feature, _ in feature_importance]
            existing_remaining_features = list(set(remaining_features) & set(X_train.columns))

            X_train = X_train.drop(columns=existing_remaining_features)
            X_val = X_val.drop(columns=existing_remaining_features)

            all_removed_features.extend(existing_remaining_features)
            print(f"Removed remaining features: {existing_remaining_features}")
            print(f"Final number of features: {X_train.shape[1]}")

            final_model, final_accuracy, final_auc, final_best_params, _, _, _, _, best_accuracy, best_auc = train_and_evaluate_model(
                X_train, X_val, y_train, y_val, initial_feature_count - X_train.shape[1], all_removed_features,
                best_accuracy, best_auc)

            print(f"Final model performance:")
            print(f"Accuracy: {final_accuracy:.4f}, AUC: {final_auc:.4f}")
            print(f"Best overall accuracy: {best_accuracy:.4f}, Best overall AUC: {best_auc:.4f}")
            print(f"Best parameters: {final_best_params}")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")

    print("Feature removal process completed.")