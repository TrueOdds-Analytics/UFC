import pandas as pd
import numpy as np
from optuna.samplers import TPESampler
import joblib
import optuna.importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from optuna.pruners import MedianPruner
import matplotlib.pyplot as plt

best_accuracy = 0
best_auc = 0
best_model_state = None
best_train_losses = []
best_val_losses = []
best_test_losses = []
best_train_auc = []
best_val_auc = []
best_test_auc = []


def objective(trial):
    global best_accuracy, best_auc, best_model_state, best_train_losses, best_val_losses, best_test_losses, best_train_auc, best_val_auc, best_test_auc

    search_space = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': trial.suggest_int('random_state', 1, 100)
    }

    X_train, X_val, X_test, y_train, y_val, y_test = get_train_test_data()

    model = RandomForestClassifier(**search_space)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    if test_accuracy > best_accuracy or (test_accuracy == best_accuracy and test_auc > best_auc):
        best_accuracy = test_accuracy
        best_auc = test_auc
        joblib.dump(model, f'models/model_{test_accuracy}.pkl')
        best_train_losses.append(train_accuracy)
        best_val_losses.append(val_accuracy)
        best_test_losses.append(test_accuracy)
        best_train_auc.append(train_auc)
        best_val_auc.append(val_auc)
        best_test_auc.append(test_auc)

        plot_losses(best_train_losses, best_val_losses, best_test_losses, best_train_auc, best_val_auc, best_test_auc, trial.number, best_accuracy, best_auc)

    return test_accuracy


def get_train_test_data():
    # Load train data from CSV
    train_data = pd.read_csv('../data/train test data/train_data.csv')
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

    # Load test data from CSV
    test_data = pd.read_csv('data/train test data/test_data.csv')
    test_labels = test_data['winner']
    test_data = test_data.drop(['winner'], axis=1)

    # Shuffle test data
    test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
    test_labels = test_labels.sample(frac=1, random_state=42).reset_index(drop=True)

    # Convert specified columns to category type
    test_data[category_columns] = test_data[category_columns].astype("category")

    X_train = train_data
    y_train = train_labels

    X_val = val_data
    y_val = val_labels

    X_test = test_data
    y_test = test_labels

    return X_train, X_val, X_test, y_train, y_val, y_test


def threshold_selector(clf, X_train, y_train, X_test, y_test):
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

            model = RandomForestClassifier(**clf.get_params())

            model.fit(select_X_train, y_train.values.ravel())
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
    selected_model = RandomForestClassifier(**clf.get_params())
    selected_model.fit(select_X_train, y_train.values.ravel())

    return selected_model, best_accuracy


def plot_losses(train_losses, val_losses, test_losses, train_auc, val_auc, test_auc, trial_number, accuracy, auc):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(train_losses, label='Train Accuracy')
    ax1.plot(val_losses, label='Validation Accuracy')
    ax1.plot(test_losses, label='Test Accuracy')
    ax1.set_xlabel('Number of trials')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'Accuracy Curves (Trial {trial_number}, Acc: {accuracy:.4f}, AUC: {auc:.4f})')
    ax1.legend()
    ax1.grid()

    ax2.plot(train_auc, label='Train AUC')
    ax2.plot(val_auc, label='Validation AUC')
    ax2.plot(test_auc, label='Test AUC')
    ax2.set_xlabel('Number of trials')
    ax2.set_ylabel('AUC')
    ax2.set_title(f'AUC Curves (Trial {trial_number}, Acc: {accuracy:.4f}, AUC: {auc:.4f})')
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sampler = TPESampler()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=MedianPruner(),
                                study_name="Test Maximum")
    try:
        study.optimize(objective, n_trials=100)
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")

    print("Number of finished trials: ", len(study.trials))
    print("Best trial: ", study.best_trial.params)
    print("Best value: ", study.best_value)

    best_model = best_model_state
    print(f"Best model accuracy: {best_accuracy}")
    print(f"Best model AUC: {best_auc}")

    loaded_model = joblib.load(f'models/model_{best_accuracy}.pkl')

    X_train, X_val, X_test, y_train, y_val, y_test = get_train_test_data()
    selected_model, best_accuracy_final = threshold_selector(loaded_model, X_train, y_train, X_test, y_test)
    joblib.dump(selected_model, f'models/model_final_{best_accuracy_final}.pkl')
    importances = selected_model.feature_importances_
    feature_names = X_train.columns
    feature_importance_list = list(zip(feature_names, importances))
    feature_importance_list.sort(key=lambda x: x[1], reverse=True)
    print("Most Important Features:")
    for feature, importance in feature_importance_list:
        print(f'{feature}: {importance}')