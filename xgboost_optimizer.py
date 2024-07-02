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

# Global flag for pausing
should_pause = False
best_accuracy = 0
best_auc = 0


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
    # Load train data
    train_data = pd.read_csv('data/train test data/train_data.csv')
    train_labels = train_data['winner']
    train_data = train_data.drop(['winner'], axis=1)

    # Load validation data
    val_data = pd.read_csv('data/train test data/val_data.csv')
    val_labels = val_data['winner']
    val_data = val_data.drop(['winner'], axis=1)

    # Shuffle data
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    train_labels = train_labels.sample(frac=1, random_state=42).reset_index(drop=True)
    val_data = val_data.sample(frac=1, random_state=42).reset_index(drop=True)
    val_labels = val_labels.sample(frac=1, random_state=42).reset_index(drop=True)

    # Convert specified columns to category type
    category_columns = [
        'result_fight_1', 'winner_fight_1', 'weight_class_fight_1', 'scheduled_rounds_fight_1',
        'result_b_fight_1', 'winner_b_fight_1', 'scheduled_rounds_b_fight_1',
        'result_fight_2', 'winner_fight_2', 'scheduled_rounds_fight_2',
        'result_b_fight_2', 'winner_b_fight_2', 'scheduled_rounds_b_fight_2',
        'result_fight_3', 'winner_fight_3', 'scheduled_rounds_fight_3',
        'result_b_fight_3', 'winner_b_fight_3', 'scheduled_rounds_b_fight_3'
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
        f'Learning Curves - Log Loss (Features removed: {features_removed}, Test Acc: {accuracy:.4f}, Test AUC: {auc:.4f})')
    ax1.legend()
    ax1.grid()

    ax2.plot(train_auc, label='Train AUC')
    ax2.plot(val_auc, label='Validation AUC')
    ax2.set_xlabel('Number of iterations')
    ax2.set_ylabel('AUC')
    ax2.set_title(
        f'Learning Curves - AUC (Features removed: {features_removed}, Test Acc: {accuracy:.4f}, Test AUC: {auc:.4f})')
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.show()  # This will display the plot in the IDE


def create_shap_graph(model_path):
    # Load the model
    model = xgb.XGBClassifier(enable_categorical=True)
    model.load_model(model_path)

    # Get the data
    X_train, _, _, _ = get_train_val_data()

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Calculate mean absolute SHAP values
    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
    feature_importance = sorted(zip(X_train.columns, mean_abs_shap_values), key=lambda x: x[1], reverse=True)

    # Print the most influential 150 features
    top_150_features = feature_importance[:50]
    print("Most influential features:")
    for feature, importance in top_150_features:
        print(f"{feature}: {importance}")

    # Create a DataFrame for the top 150 features
    top_150_df = pd.DataFrame(top_150_features, columns=["Feature", "Importance"])

    # Create a vertical bar plot
    plt.figure(figsize=(12, 20))  # Adjust the figure size as needed
    plt.barh(top_150_df["Feature"], top_150_df["Importance"], color='blue')
    plt.xlabel('Mean Absolute SHAP Value')
    plt.ylabel('Features')
    plt.title(f"SHAP Values for model: {model_path}")
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest importance at the top
    plt.tight_layout()
    plt.show()  # Display the plot

    print(f"SHAP summary plot displayed for model: {model_path}")


def train_and_evaluate_model(X_train, X_val, y_train, y_val, features_removed, all_removed_features, best_accuracy,
                             best_auc):
    global should_pause

    def create_fit_and_evaluate_model(params):
        model = xgb.XGBClassifier(**params)
        eval_set = [(X_train, y_train), (X_val, y_val)]
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        results = model.evals_result()
        train_losses = results['validation_0']['logloss']
        val_losses = results['validation_1']['logloss']
        train_auc = results['validation_0']['auc']
        val_auc = results['validation_1']['auc']

        y_val_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

        return model, accuracy, auc, train_losses, val_losses, train_auc, val_auc

    def objective(trial):
        global should_pause, best_accuracy, best_auc
        while should_pause:
            time.sleep(1)

        params = {
            'lambda': trial.suggest_float('lambda', 0.1, 150.0, log=True),  # Increased lower bound
            'alpha': trial.suggest_float('alpha', 0.1, 150.0, log=True),  # Increased lower bound
            'tree_method': 'hist',
            'device': 'cuda',
            'objective': 'binary:logistic',
            'verbosity': 0,
            'n_jobs': -1,
            'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 25.0, log=True),  # Increased lower bound
            'max_depth': trial.suggest_int('max_depth', 1, 10),  # Reduced upper bound
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 0.8),  # Reduced upper bound
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),  # Reduced upper bound
            'gamma': trial.suggest_float('gamma', 0.1, 10.0),  # Increased lower bound
            'eta': trial.suggest_float('eta', 0.0001, 0.1, log=True),  # Reduced upper bound
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'eval_metric': ['logloss', 'auc'],
            'enable_categorical': True
        }

        model, accuracy, auc, train_losses, val_losses, train_auc, val_auc = create_fit_and_evaluate_model(params)

        if accuracy > best_accuracy or (accuracy == best_accuracy and auc > best_auc):
            best_accuracy = accuracy
            best_auc = auc
            model_filename = f'models/xgboost/model_{accuracy:.4f}_{features_removed}_features_removed.json'
            model.save_model(model_filename)

            plot_losses(train_losses, val_losses, train_auc, val_auc, features_removed, accuracy, auc)

            features_removed_filename = f'removed features/removed_features_{accuracy:.4f}_{features_removed}_features.txt'
            with open(features_removed_filename, 'w') as f:
                f.write(f"Total features removed: {features_removed}\n\n")
                f.write("Removed features:\n")
                for feature in all_removed_features:
                    f.write(f"{feature}\n")

        return accuracy

    study = optuna.create_study(direction='maximize', sampler=TPESampler(), pruner=MedianPruner())

    # Keep track of trial accuracies
    trial_accuracies = []

    def callback(study, trial):
        trial_accuracies.append(trial.value)

    study.optimize(objective, n_trials=10000, callbacks=[callback])

    # Print the average accuracy at the end of the study
    avg_accuracy = sum(trial_accuracies) / len(trial_accuracies)

    best_params = study.best_params
    best_params.update({
        'tree_method': 'hist',
        'device': 'cuda',
        'objective': 'binary:logistic',
        'verbosity': 0,
        'n_jobs': -1,
        'n_estimators': 10000,
        'early_stopping_rounds': 50,
        'eval_metric': ['logloss', 'auc'],
        'enable_categorical': True
    })

    best_model, best_accuracy, best_auc, train_losses, val_losses, train_auc, val_auc = create_fit_and_evaluate_model(
        best_params)
    print(f"Average trial accuracy: {avg_accuracy:.4f}")
    return best_model, best_accuracy, best_auc, best_params, train_losses, val_losses, train_auc, val_auc, best_accuracy, best_auc


def get_top_features_and_retrain(model_path, n_features, X_train, X_val, y_train, y_val):
    # Load the model
    model = xgb.XGBClassifier(enable_categorical=True)
    model.load_model(model_path)

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Calculate mean absolute SHAP values
    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
    feature_importance = sorted(zip(X_train.columns, mean_abs_shap_values), key=lambda x: x[1], reverse=True)

    # Get top N features
    top_n_features = [feature for feature, _ in feature_importance[:n_features]]

    print(f"Top {n_features} features:")
    for feature, importance in feature_importance[:n_features]:
        print(f"{feature}: {importance}")

    # Filter data to include only top N features
    X_train_filtered = X_train[top_n_features]
    X_val_filtered = X_val[top_n_features]

    # Retrain the model with top N features
    print(f"\nRetraining model with top {n_features} features...")
    removed_features = [col for col in X_train.columns if col not in top_n_features]
    retrained_model, retrained_accuracy, retrained_auc, retrained_best_params, train_losses, val_losses, train_auc, val_auc, best_accuracy, best_auc = train_and_evaluate_model(
        X_train_filtered, X_val_filtered, y_train, y_val, len(removed_features), removed_features, 0, 0
    )

    return retrained_model, best_accuracy, best_auc, retrained_best_params


if __name__ == "__main__":
    # Start the user input thread
    input_thread = threading.Thread(target=user_input_thread, daemon=True)
    input_thread.start()

    X_train, X_val, y_train, y_val = get_train_val_data()
    print("Starting initial optimization and evaluation...")
    try:
        initial_model, initial_accuracy, initial_auc, initial_best_params, _, _, _, _, best_accuracy, best_auc = train_and_evaluate_model(
            X_train, X_val, y_train, y_val, 0, [], 0, 0
        )
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")

    print("Initial optimization completed.")
    print(f"Best model validation accuracy: {best_accuracy:.4f}")
    print(f"Best model validation AUC: {best_auc:.4f}")
    print("Best parameters:", initial_best_params)
    print("--------------------")

    # # Create SHAP graph for the best model
    # best_model_path = f'models/xgboost/model_0.7723_0_features_removed.json'
    # print(f"Creating SHAP graph for the best model: {best_model_path}")
    # create_shap_graph(best_model_path)
    # print("SHAP graph creation completed.")
    # print("--------------------")
    #
    # X_train, X_val, y_train, y_val = get_train_val_data()
    # n_features = 50  # You can change this to any number you want
    # retrained_model, retrained_accuracy, retrained_auc, retrained_best_params = get_top_features_and_retrain(
    #     best_model_path, n_features, X_train, X_val, y_train, y_val
    # )
    #
    # print("--------------------")
    # print("Retraining with top features completed.")
    # print(f"Retrained model validation accuracy: {retrained_accuracy:.4f}")
    # print(f"Retrained model validation AUC: {retrained_auc:.4f}")
    # print("Retrained model best parameters:", retrained_best_params)
