import pandas as pd
import numpy as np
from optuna.samplers import TPESampler
import joblib
import optuna.importance
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from optuna.pruners import MedianPruner

best_accuracy = 0
best_model_state = None

def objective(trial):
    global best_accuracy, best_model_state, best_train_accuracy, best_train_losses, best_test_accuracy, best_test_losses

    search_space = {
        "verbosity": -1,
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
        'device_type': trial.suggest_categorical('device_type', ['gpu']),
        "objective": trial.suggest_categorical("objective", ["binary"]),
        "num_threads": trial.suggest_categorical("num_threads", [-1]),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-8, 10.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "boost_from_average": trial.suggest_categorical("boost_from_average", [False]),
        "random_state": trial.suggest_int("random_state", 1, 1000),
    }

    X_train, X_test, y_train, y_test = get_train_test_data()

    model = lgb.LGBMClassifier(**search_space)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)]
    )
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Update the global best model if the current one is better
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_state = model  # Save the best model

    return accuracy


def get_train_test_data():
    # Load train data from CSV
    train_data = pd.read_csv('data/train test data/train_data.csv')
    train_labels = train_data['winner']
    train_data = train_data.drop(['winner'], axis=1)

    # Shuffle train data
    train_data = train_data.sample(frac=1, random_state=20).reset_index(drop=True)
    train_labels = train_labels.sample(frac=1, random_state=20).reset_index(drop=True)

    # Load test data from CSV
    test_data = pd.read_csv('data/train test data/test_data.csv')
    test_labels = test_data['winner']
    test_data = test_data.drop(['winner'], axis=1)

    # Shuffle test data
    test_data = test_data.sample(frac=1, random_state=20).reset_index(drop=True)
    test_labels = test_labels.sample(frac=1, random_state=20).reset_index(drop=True)

    X_train = train_data
    y_train = train_labels

    X_test = test_data
    y_test = test_labels

    return X_train, X_test, y_train, y_test

def threshold_selector(clf, X_train, y_train, X_test, y_test, early_stopping_rounds=10):
    X_train = X_train.values
    X_test = X_test.values
    thresholds = np.arange(0.0, 1.0, 0.0001)
    avgs = []

    # Ensure clf is prefit and has feature importances
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]  # Sort in descending order
        sorted_importances = importances[sorted_indices]

        print("Feature importances (sorted):")
        for idx in sorted_indices:
            print(f"Feature {idx}: {importances[idx]}")

        print(f"Sum of feature importances: {np.sum(importances)}")
    else:
        print("The classifier does not have feature importances.")

    for thresh in thresholds:
        loopavg = []
        # Run each threshold 1x for the avg
        for x in range(1):
            selection = SelectFromModel(clf, threshold=thresh, prefit=True)
            select_X_train = selection.transform(X_train)
            select_X_test = selection.transform(X_test)
            n_selected_features = select_X_train.shape[1]

            if n_selected_features < 10:
                print(f"Stopping as n_selected_features < 10 (n={n_selected_features})")
                break

            model = lgb.LGBMClassifier(
                verbosity=0,
                reg_lambda=0.06149361099544647,
                reg_alpha=0.045998526993829864,
                tree_method="gpu_hist",
                objective="binary:logistic",
                n_jobs=-1,
                learning_rate=0.02920212897819158,
                min_child_weight=5,
                max_depth=18,
                max_delta_step=3,
                subsample=0.2705154119188745,
                colsample_bytree=0.7402834256314674,
                gamma=0.017109525533870292,
                n_estimators=7295,
                eta=0.12635351933273808,
                random_state=42
            )
            # Set early stopping rounds
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
    selected_model = lgb.LGBMClassifier(**clf.get_params())
    selected_model.set_params(early_stopping_rounds=early_stopping_rounds)
    selected_model.fit(select_X_train, y_train.values.ravel(), eval_set=[(select_X_test, y_test)], verbose=False)

    return selected_model, best_accuracy


if __name__ == "__main__":
    sampler = TPESampler()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=MedianPruner())
    try:
        study.optimize(objective, n_trials=100000)
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")

    print("Number of finished trials: ", len(study.trials))
    print("Best trial: ", study.best_trial.params)
    print("Best value: ", study.best_value)

    best_model = best_model_state
    print(f"Best model accuracy: {best_accuracy}")

    # Save the best model as a file
    joblib.dump(best_model, f'models/best_model_{best_accuracy}.pkl')
    loaded_model = joblib.load(f'models/best_model_0.676923076923077.pkl')
    # Run threshold selector using the best model
    X_train, X_test, y_train, y_test = get_train_test_data()
    selected_model = threshold_selector(loaded_model, X_train, y_train, X_test, y_test)

    importances = selected_model.feature_importances_
    feature_names = X_train.columns
    feature_importance_list = list(zip(feature_names, importances))
    feature_importance_list.sort(key=lambda x: x[1], reverse=True)
    print("Top 50 Most Important Features:")
    for feature, importance in feature_importance_list[:50]:
        print(f'{feature}: {importance}')