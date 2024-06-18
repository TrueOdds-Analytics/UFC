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
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'device': trial.suggest_categorical('device', ['gpu']),
        "objective": trial.suggest_categorical("objective", ["binary"]),
        "num_threads": trial.suggest_categorical("num_threads", [-1]),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
        "max_depth": trial.suggest_int("max_depth", 1, 30),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-8, 10.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "boost_from_average": trial.suggest_categorical("boost_from_average", [False]),
        "random_state": trial.suggest_int("random_state", 1, 100)
    }

    model = lgb.LGBMClassifier(**search_space)

    early_stopping_rounds = 10
    model.set_params(early_stopping_rounds=early_stopping_rounds)

    X_train, X_test, y_train, y_test = get_train_test_data()

    best_score = -np.inf
    best_iteration = 0

    for epoch in range(100):
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if accuracy > best_score:
            best_score = accuracy
            best_iteration = model.best_iteration_

        trial.report(accuracy, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

        if model.best_iteration_ >= early_stopping_rounds:
            break

    # Update the global best model if the current one is better
    if best_score > best_accuracy:
        best_accuracy = best_score
        best_model_state = model  # Save the best model

    return best_score


def get_train_test_data():
    # Load train data from CSV
    train_data = pd.read_csv('data/train_data.csv')
    train_labels = train_data['winner']
    train_data = train_data.drop(['winner'], axis=1)

    # Shuffle train data
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    train_labels = train_labels.sample(frac=1, random_state=42).reset_index(drop=True)

    # Load test data from CSV
    test_data = pd.read_csv('data/test_data.csv')
    test_labels = test_data['winner']
    test_data = test_data.drop(['winner'], axis=1)

    # Shuffle test data
    test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
    test_labels = test_labels.sample(frac=1, random_state=42).reset_index(drop=True)

    X_train = train_data
    y_train = train_labels

    X_test = test_data
    y_test = test_labels

    return X_train, X_test, y_train, y_test

def threshold_selector(clf, X_train, y_train, X_test, y_test):
    X_train = X_train.values
    X_test = X_test.values
    thresholds = np.arange(.1, 5, .05)
    avgs = []
    loopavg = []
    count = 1

    for thresh in thresholds:
        # Run each threshold 10x for the avg
        for x in range(0, 10):
            selection = SelectFromModel(clf, threshold=f'{thresh}*median', prefit=True)
            select_X_train = selection.transform(X_train)
            model = lgb.LGBMClassifier(lambda_l1=0.024029166453171127,
                                       lambda_l2=0.6297819805077854,
                                       device="gpu",
                                       objective="binary",
                                       num_threads=-1,
                                       learning_rate=0.008244181542067349,
                                       min_child_samples=15,
                                       max_depth=13,
                                       num_leaves=31,
                                       subsample=0.5361122651825095,
                                       colsample_bytree=0.7225143130683623,
                                       min_child_weight=0.018882987701220703,
                                       n_estimators=183,
                                       boost_from_average=False,
                                       random_state=46)
            count += 1
            model.fit(select_X_train, y_train.values.ravel())
            select_X_test = selection.transform(X_test)
            ypred = model.predict(select_X_test)
            accuracy = accuracy_score(y_test, ypred)
            msg = f"Thresh={thresh}, n={select_X_train.shape[1]}, Accuracy: {accuracy * 100.0}"
            print(msg)
            loopavg.append(accuracy)

        avgs.append((thresh, np.mean(loopavg)))
        loopavg = []

    best_thresh, best_accuracy = max(avgs, key=lambda x: x[1])
    print(f"Best threshold: {best_thresh}, Best accuracy: {best_accuracy}")

    selection = SelectFromModel(clf, threshold=f'{best_thresh}*median', prefit=True)
    select_X_train = selection.transform(X_train)
    selected_model = lgb.LGBMClassifier(**clf.get_params())
    selected_model.fit(select_X_train, y_train.values.ravel())
    return selected_model


if __name__ == "__main__":
    sampler = TPESampler()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=MedianPruner())
    try:
        study.optimize(objective, n_trials=2500)
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