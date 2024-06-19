import pandas as pd
import numpy as np
from optuna.samplers import TPESampler
import joblib
import optuna.importance
import catboost as cb
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
        'depth': trial.suggest_int('depth', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.01, 100.0, log=True),
        'iterations': 10000,
        'random_seed': trial.suggest_int('random_seed', 1, 100),
        'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
        'od_wait': trial.suggest_int('od_wait', 10, 50),
        'eval_metric': 'Logloss',
        'task_type': 'GPU'  # Specify GPU task type
    }

    X_train, X_test, y_train, y_test = get_train_test_data()

    model = cb.CatBoostClassifier(**search_space)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)

    eval_result = model.get_evals_result()
    train_losses = eval_result['learn']['Logloss']
    test_losses = eval_result['validation']['Logloss']

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


def threshold_selector(clf, X_train, y_train, X_test, y_test, early_stopping_rounds=10):
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

            model = cb.CatBoostClassifier(
                depth=6,
                learning_rate=0.1,
                l2_leaf_reg=1.0,
                random_strength=0.1,
                bagging_temperature=0.5,
                iterations=500,
                random_seed=42,
                od_type='Iter',
                od_wait=20,
                eval_metric='Logloss'
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
    selected_model = cb.CatBoostClassifier(**clf.get_params())
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
                                study_name="Post Sequence Combination")
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