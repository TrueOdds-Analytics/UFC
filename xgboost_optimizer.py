import pandas as pd
import numpy as np
from optuna.samplers import TPESampler
import joblib
import optuna.importance
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from optuna.pruners import MedianPruner
import matplotlib.pyplot as plt

best_accuracy = 0
best_model_state = None
best_train_losses = []
best_test_losses = []


def objective(trial):
    global best_accuracy, best_model_state, best_train_losses, best_test_losses

    search_space = {
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'tree_method': trial.suggest_categorical('tree_method', ['hist']),
        'device': trial.suggest_categorical('device', ['cuda']),
        "objective": trial.suggest_categorical("objective", ["binary:logistic"]),
        "verbosity": trial.suggest_categorical("verbosity", [0]),
        "n_jobs": trial.suggest_categorical("n_jobs", [-1]),
        "min_child_weight": trial.suggest_int("min_child_weight", 0, 20),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0, log=True),
        "gamma": trial.suggest_float("gamma", .01, 0.4, log=True),
        "eta": trial.suggest_float("eta", 0.001, 0.1, log=True),
        "seed": trial.suggest_int("seed", 1, 100),
        "n_estimators": 10000,
        "early_stopping_rounds": 100,
        "eval_metric": "logloss",
        "enable_categorical": True,
    }

    X_train, X_test, y_train, y_test = get_train_test_data()

    model = xgb.XGBClassifier(**search_space)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    eval_results = model.evals_result()

    train_losses = eval_results['validation_0']['logloss']
    test_losses = eval_results['validation_1']['logloss']

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_state = model
        best_train_losses = train_losses
        best_test_losses = test_losses

        plot_losses(best_train_losses, best_test_losses, trial.number, best_accuracy)

    return accuracy


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

    # Load test data from CSV
    test_data = pd.read_csv('data/test_data.csv')
    test_labels = test_data['winner']
    test_data = test_data.drop(['winner'], axis=1)

    # Shuffle test data
    test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
    test_labels = test_labels.sample(frac=1, random_state=42).reset_index(drop=True)

    # Convert specified columns to category type
    test_data[category_columns] = test_data[category_columns].astype("category")

    X_train = train_data
    y_train = train_labels

    X_test = test_data
    y_test = test_labels

    return X_train, X_test, y_train, y_test


def threshold_selector(clf, X_train, y_train, X_test, y_test, early_stopping_rounds=100):
    X_train = X_train.values
    X_test = X_test.values
    thresholds = np.arange(0.0, 1.0, 0.00001)
    avgs = []

    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        sorted_importances = importances[sorted_indices]

        print("Feature importances (sorted):")
        for idx in sorted_indices:
            print(f"Feature {idx}: {importances[idx]}")

        print(f"Sum of feature importances: {np.sum(importances)}")
    else:
        print("The classifier does not have feature importances.")

    for thresh in thresholds:
        loopavg = []
        for x in range(1):
            selection = SelectFromModel(clf, threshold=thresh, prefit=True)
            select_X_train = selection.transform(X_train)
            select_X_test = selection.transform(X_test)
            n_selected_features = select_X_train.shape[1]

            if n_selected_features < 10:
                print(f"Stopping as n_selected_features < 10 (n={n_selected_features})")
                break

            model = xgb.XGBClassifier(
                verbosity=0,
                reg_lambda=2.2251244020055068,
                reg_alpha=0.12599370580364663,
                tree_method="gpu_hist",
                objective="binary:logistic",
                n_jobs=-1,
                learning_rate=0.0048697574253707306,
                min_child_weight=17,
                max_depth=3,
                max_delta_step=0,
                subsample=0.6982893260507582,
                colsample_bytree=0.4639609815677423,
                gamma=0.08907431463881274,
                n_estimators=933,
                eta=0.12072853862369459,
                random_state=4
            )
            model.set_params(early_stopping_rounds=early_stopping_rounds)

            model.fit(select_X_train, y_train.values.ravel(), eval_set=[(select_X_test, y_test)], verbose=False)
            ypred = model.predict(select_X_test)
            accuracy = accuracy_score(y_test, ypred)
            print(f"Thresh={thresh:.4f}, n={n_selected_features}, Accuracy: {accuracy * 100.0:.2f}")
            loopavg.append(accuracy)

        if n_selected_features < 10:
            break

        avgs.append((thresh, np.mean(loopavg)))

    best_thresh, best_accuracy = max(avgs, key=lambda x: x[1])
    print(f"Best threshold: {best_thresh:.4f}, Best accuracy: {best_accuracy:.4f}")

    selection = SelectFromModel(clf, threshold=best_thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    select_X_test = selection.transform(X_test)
    selected_model = xgb.XGBClassifier(**clf.get_params())
    selected_model.set_params(early_stopping_rounds=early_stopping_rounds)
    selected_model.fit(select_X_train, y_train.values.ravel(), eval_set=[(select_X_test, y_test)], verbose=False)

    return selected_model, best_accuracy


def plot_losses(train_losses, test_losses, trial_number, accuracy):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Number of iterations')
    plt.ylabel('Log Loss')
    plt.title(f'Learning Curves - Log Loss (Trial {trial_number}, Acc: {accuracy:.4f})')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sampler = TPESampler()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=MedianPruner(),
                                study_name="Post categorical")
    try:
        study.optimize(objective, n_trials=10000)
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")

    print("Number of finished trials: ", len(study.trials))
    print("Best trial: ", study.best_trial.params)
    print("Best value: ", study.best_value)

    best_model = best_model_state
    print(f"Best model accuracy: {best_accuracy}")

    joblib.dump(best_model, f'models/best_model_{best_accuracy}.pkl')
    loaded_model = joblib.load(f'models/best_model_{best_accuracy}.pkl')

    X_train, X_test, y_train, y_test = get_train_test_data()
    selected_model, best_accuracy_final = threshold_selector(loaded_model, X_train, y_train, X_test, y_test)
    joblib.dump(selected_model, f'models/best_model_final_{best_accuracy_final}.pkl')
    importances = selected_model.feature_importances_
    feature_names = X_train.columns
    feature_importance_list = list(zip(feature_names, importances))
    feature_importance_list.sort(key=lambda x: x[1], reverse=True)
    print("Most Important Features:")
    for feature, importance in feature_importance_list:
        print(f'{feature}: {importance}')
