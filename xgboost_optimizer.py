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

# Global variables for user interaction
should_pause = False
best_accuracy = 0
best_auc = 0
best_auc_diff = 0


def user_input_thread():
    """Background thread to handle user input for pausing/quitting the optimization."""
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


def load_and_preprocess_data(odds_type='close_odds'):
    """
    Load and preprocess the training and validation data.

    Args:
        odds_type: Which odds to use in the model: 'open_odds', 'close_odds', 'drop_open' or None

    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    print(f"Loading data with odds type: {odds_type}")

    # Load data from CSV files
    train_data = pd.read_csv('data/train test data/train_data.csv')
    val_data = pd.read_csv('data/train test data/val_data.csv')

    # Extract target variable
    train_labels = train_data['winner']
    val_labels = val_data['winner']

    # Remove target from feature sets
    train_data = train_data.drop(['winner'], axis=1)
    val_data = val_data.drop(['winner'], axis=1)

    # Determine columns to drop based on odds_type
    if odds_type == 'open_odds':
        columns_to_drop = [
            'fighter_a', 'fighter_b', 'current_fight_date',
            'current_fight_closing_odds', 'current_fight_closing_odds_b',
            'current_fight_closing_odds_ratio', 'current_fight_closing_odds_diff'
        ]
    elif odds_type == 'close_odds':
        columns_to_drop = ['fighter_a', 'fighter_b', 'current_fight_date']
    elif odds_type == 'drop_open':
        base_columns_to_drop = ['fighter_a', 'fighter_b', 'current_fight_date']

        # Assuming df is your DataFrame, this will find all columns with 'open' in their name
        open_columns = [col for col in df.columns if 'open' in col.lower()]

        # Combine the lists
        columns_to_drop = base_columns_to_drop + open_columns
    else:
        columns_to_drop = list(train_data.columns[train_data.columns.str.contains('odd', case=False)]) + [
            'fighter_a', 'fighter_b', 'current_fight_date'
        ]

    # Drop specified columns
    train_data = train_data.drop(columns=columns_to_drop)
    val_data = val_data.drop(columns=columns_to_drop)

    # Shuffle data (with consistent random state for reproducibility)
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    train_labels = train_labels.sample(frac=1, random_state=42).reset_index(drop=True)
    val_data = val_data.sample(frac=1, random_state=42).reset_index(drop=True)
    val_labels = val_labels.sample(frac=1, random_state=42).reset_index(drop=True)

    # Convert categorical columns
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

    print(f"Data loaded and preprocessed. Training features: {train_data.shape[1]}")
    return train_data, val_data, train_labels, val_labels


def plot_learning_curves(train_losses, val_losses, train_auc, val_auc, feature_count, accuracy, auc):
    """
    Plot learning curves for model training.

    Args:
        train_losses: List of training loss values
        val_losses: List of validation loss values
        train_auc: List of training AUC values
        val_auc: List of validation AUC values
        feature_count: Number of features used
        accuracy: Final validation accuracy
        auc: Final validation AUC
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot loss curves
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Number of iterations')
    ax1.set_ylabel('Log Loss')
    ax1.set_title(
        f'Learning Curves - Log Loss (Features: {feature_count}, Val Acc: {accuracy:.4f}, Val AUC: {auc:.4f})')
    ax1.legend()
    ax1.grid()

    # Plot AUC curves
    ax2.plot(train_auc, label='Train AUC')
    ax2.plot(val_auc, label='Validation AUC')
    ax2.set_xlabel('Number of iterations')
    ax2.set_ylabel('AUC')
    ax2.set_title(f'Learning Curves - AUC (Features: {feature_count}, Val Acc: {accuracy:.4f}, Val AUC: {auc:.4f})')
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.show()


def create_feature_importance_plot(model_path, X_train, top_n, print_count):
    """
    Create and display SHAP feature importance plot.

    Args:
        model_path: Path to the saved model file
        X_train: Training data used for SHAP analysis
        top_n: Number of top features to display in the plot
        print_count: Number of top features to print to console
    """
    print(f"Creating SHAP feature importance plot for model: {model_path}")

    # Load the model
    model = xgb.XGBClassifier(enable_categorical=True)
    model.load_model(model_path)

    # Create SHAP explainer and calculate values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Calculate mean absolute SHAP values
    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
    feature_importance = sorted(zip(X_train.columns, mean_abs_shap_values), key=lambda x: x[1], reverse=True)

    # Print top features
    print("Most influential features:")
    for feature, importance in feature_importance[:print_count]:
        print(f"{feature}: {importance:.6f}")

    # Create plot
    top_n_features = feature_importance[:top_n]
    top_n_df = pd.DataFrame(top_n_features, columns=["Feature", "Importance"])

    plt.figure(figsize=(12, 20))
    plt.barh(top_n_df["Feature"], top_n_df["Importance"], color='blue')
    plt.xlabel('Mean Absolute SHAP Value')
    plt.ylabel('Features')
    plt.title(f"Top {top_n} Features by SHAP Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    return [feature for feature, _ in feature_importance]


def train_xgboost_model(params, X_train, X_val, y_train, y_val):
    """
    Train an XGBoost model with the given parameters and evaluate it.

    Args:
        params: Dictionary of XGBoost parameters
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels

    Returns:
        Tuple of (model, accuracy, auc, train_losses, val_losses, train_auc, val_auc)
    """
    # Ensure required parameters are set
    params['enable_categorical'] = True
    params['eval_metric'] = ['logloss', 'auc']

    # Default parameters if not specified
    default_params = {
        'tree_method': 'hist',
        'device': 'cuda',
        'objective': 'binary:logistic',
        'verbosity': 0,
        'n_jobs': -1,
        'n_estimators': 1000,
        'early_stopping_rounds': 150
    }

    # Update params with defaults if not already set
    for key, value in default_params.items():
        if key not in params:
            params[key] = value

    # Initialize and train the model
    model = xgb.XGBClassifier(**params)
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    # Extract training history
    results = model.evals_result()
    train_losses = results['validation_0']['logloss']
    val_losses = results['validation_1']['logloss']
    train_auc = results['validation_0'].get('auc', [])
    val_auc = results['validation_1'].get('auc', [])

    # Evaluate model
    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)

    # Calculate AUC
    try:
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    except ValueError as e:
        print(f"Error calculating AUC: {str(e)}")
        print(f"Unique values in y_val: {np.unique(y_val)}")
        auc = 0.5  # Default value when AUC calculation fails

    return model, accuracy, auc, train_losses, val_losses, train_auc, val_auc


def objective(trial, X_train, X_val, y_train, y_val, params=None):
    """
    Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels
        params: Optional parameter dict (if None, will suggest parameters)

    Returns:
        Validation accuracy score
    """
    global should_pause, best_accuracy, best_auc, best_auc_diff

    # Wait if paused
    while should_pause:
        time.sleep(1)

    # Suggest parameters if not provided
    if params is None:
        params = {
            'lambda': trial.suggest_float('lambda', 5, 10, log=True),
            'alpha': trial.suggest_float('alpha', 5, 10, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 1, 10.0),
            'max_depth': trial.suggest_int('max_depth', 1, 6),
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-3, 10.0, log=True),
            'eta': trial.suggest_float('eta', 0.01, 0.1, log=True),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        }

    # Train and evaluate model
    model, accuracy, auc, train_losses, val_losses, train_auc, val_auc = train_xgboost_model(
        params, X_train, X_val, y_train, y_val
    )

    # Calculate train-validation AUC difference (for overfitting detection)
    if train_auc and val_auc:
        auc_diff = abs(train_auc[-1] - val_auc[-1])
    else:
        auc_diff = 1.0  # High difference when AUC isn't available

    # Save all models that meet the criteria (66% accuracy and AUC diff < 0.1)
    if accuracy >= 0.66 and (auc_diff < 0.1):
        # Keep track of best model for reporting purposes
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_auc_diff = auc_diff
            print(f"New best model found! Accuracy: {accuracy:.4f}, AUC diff: {auc_diff:.4f}")

        # Save the model
        model_dir = 'models/xgboost/jan2024-dec2025/dynamicmatchup sorted 300/'
        model_filename = f'{model_dir}model_{accuracy:.4f}_auc_diff_{auc_diff:.4f}.json'
        model.save_model(model_filename)
        print(f"Model saved: {model_filename}")

        # Plot learning curves for models that meet criteria
        plot_learning_curves(train_losses, val_losses, train_auc, val_auc,
                             len(X_train.columns), accuracy, auc)

    # Report trial progress
    print(f"Trial {trial.number}: Accuracy={accuracy:.4f}, AUC diff={auc_diff:.4f}")

    return accuracy


def optimize_model(X_train, X_val, y_train, y_val, n_trials=10000):
    """
    Perform hyperparameter optimization for XGBoost model.

    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels
        n_trials: Number of Optuna trials to run

    Returns:
        Best Optuna trial
    """
    print(f"Starting hyperparameter optimization with {n_trials} trials...")

    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(),
        pruner=MedianPruner()
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, X_train, X_val, y_train, y_val),
        n_trials=n_trials
    )

    # Print results
    print(f"Best validation accuracy: {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")

    return study.best_trial


def train_model_with_top_features(X_train, X_val, y_train, y_val, model_path, num_top_features):
    """
    Train a model using only the top features from an existing model.

    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels
        model_path: Path to the existing model to extract features from
        num_top_features: Number of top features to use

    Returns:
        Path to the saved model with top features
    """
    print(f"Training model with top {num_top_features} features from {model_path}")

    # Get feature importance from existing model
    feature_importance_list = create_feature_importance_plot(model_path, X_train, 50, num_top_features)
    top_features = feature_importance_list[:num_top_features]

    print(f"Selected top {num_top_features} features")

    # Filter data to include only top features
    X_train_top = X_train[top_features]
    X_val_top = X_val[top_features]

    # Run hyperparameter optimization for top features
    print(f"Optimizing model with top {num_top_features} features...")
    best_trial = optimize_model(X_train_top, X_val_top, y_train, y_val, n_trials=10000)

    # Train final model with best parameters
    best_params = best_trial.params
    print(f"Training final model with best parameters: {best_params}")

    best_model_top, accuracy_top, auc_top, train_losses, val_losses, train_auc, val_auc = train_xgboost_model(
        best_params, X_train_top, X_val_top, y_train, y_val
    )

    # Calculate AUC difference
    auc_diff = abs(train_auc[-1] - val_auc[-1]) if train_auc and val_auc else 1.0

    # Save model
    model_dir = 'models/xgboost/jan2024-july2024/split 125/'
    model_filename = f'{model_dir}model_top{num_top_features}_{accuracy_top:.4f}_auc_diff_{auc_diff:.4f}.json'
    best_model_top.save_model(model_filename)

    print(f"Final model results:")
    print(f"  Validation accuracy: {accuracy_top:.4f}")
    print(f"  Validation AUC: {auc_top:.4f}")
    print(f"  AUC difference: {auc_diff:.4f}")
    print(f"Model saved to: {model_filename}")

    # Plot learning curves
    plot_learning_curves(train_losses, val_losses, train_auc, val_auc,
                         num_top_features, accuracy_top, auc_top)

    return model_filename


def main():
    """Main program execution"""
    # Start user input thread
    input_thread = threading.Thread(target=user_input_thread, daemon=True)
    input_thread.start()

    try:
        # Load and preprocess data
        X_train, X_val, y_train, y_val = load_and_preprocess_data(odds_type='drop_open')
        print("Data loaded. Starting optimization...")

        # Phase 1: Train initial model with all features
        best_trial = optimize_model(X_train, X_val, y_train, y_val, n_trials=10000)

        # Use an existing model for feature selection and visualization
        existing_model_path = 'models/xgboost/jan2024-dec2025/dynamicmatchup sorted/model_0.6940_auc_diff_0.0102.json'
        print("Creating feature importance visualization for existing model")

        # Phase 2: Train model with top features only
        num_top_features = 300
        top_features_model_path = train_model_with_top_features(
            X_train, X_val, y_train, y_val, existing_model_path, num_top_features
        )

        print("Training pipeline completed successfully")

    except KeyboardInterrupt:
        print("Process interrupted by user")
    except Exception as e:
        print(f"Error during execution: {str(e)}")


if __name__ == "__main__":
    main()