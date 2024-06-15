import pandas as pd
import numpy as np
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
import joblib
import optuna.importance
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

best_accuracy = 0
best_model_state = None

def objective(trial):
    global best_accuracy, best_model_state, best_train_accuracy, best_train_losses, best_test_accuracy, best_test_losses

    search_space = {
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'tree_method': trial.suggest_categorical('tree_method', ['hist']),
        'device': trial.suggest_categorical('device', ['cuda']),
        "objective": trial.suggest_categorical("objective", ["binary:logistic"]),
        "verbosity": trial.suggest_categorical("verbosity", [0]),
        "n_jobs": trial.suggest_categorical("n_jobs", [-1]),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 0, 20),
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0, log=True),
        "gamma": trial.suggest_float("gamma", .01, 0.4, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "eta": trial.suggest_float("eta", 0.1, 0.2, log=True),
        "seed": trial.suggest_int("seed", 1, 100)
    }
    model = xgb.XGBClassifier(**search_space)
    X_train, X_test, y_train, y_test = get_train_test_data()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Update the global best model if the current one is better
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_state = model  # Save the best model

    return accuracy


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
            model = xgb.XGBClassifier(verbosity=0,
                                      reg_lambda=0.022610586588975145,
                                      reg_alpha=0.002842366557825091,
                                      tree_method="gpu_hist",
                                      objective="binary:logistic",
                                      n_jobs=-1,
                                      learning_rate=0.011209810943566266,
                                      min_child_weight=12,
                                      max_depth=1,
                                      max_delta_step=5,
                                      subsample=0.1887556683686304,
                                      colsample_bytree=0.12237412505778385,
                                      gamma=0.01318313292785324,
                                      n_estimators=163,
                                      eta=0.14114199078271825,
                                      seed=85)
            count += 1
            model.fit(select_X_train, y_train.values.ravel())  # Remove feature_names argument
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
    selected_model = xgb.XGBClassifier(**clf.get_params())
    selected_model.fit(select_X_train, y_train.values.ravel())  # Remove feature_names argument
    return selected_model

def plot_metrics(model):
    results = model.evals_result()
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)

    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Test')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Train')
    ax.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Test')
    ax.plot(x_axis, results['validation_1']['error'], label='Train')
    ax.legend()
    plt.ylabel('Classification Error')
    plt.title('XGBoost Classification Error')
    plt.show()


if __name__ == "__main__":
    sampler = TPESampler()
    study = optuna.create_study(direction="maximize", sampler=sampler)
    try:
        study.optimize(objective, n_trials=10000)
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")

    print("Number of finished trials: ", len(study.trials))
    print("Best trial: ", study.best_trial.params)
    print("Best value: ", study.best_value)

    best_model = best_model_state
    print(f"Best model accuracy: {best_accuracy}")

    # Save the best model as a file
    joblib.dump(best_model, f'models/best_model_{best_accuracy}.pkl')
    loaded_model = joblib.load(f'models/best_model_0.6615384615384615.pkl')
    # Run threshold selector using the best model
    X_train, X_test, y_train, y_test = get_train_test_data()
    selected_model = threshold_selector(loaded_model, X_train, y_train, X_test, y_test)

    # Plot metrics for the selected model
    plot_metrics(selected_model)
