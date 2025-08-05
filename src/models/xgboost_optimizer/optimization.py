"""
Hyperparameter optimization with Optuna.
"""
import time
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import config
from model import train_xgboost_model, save_model
from visualization import plot_learning_curves, create_feature_importance_plot


def _suggest_parameters(trial):
    """
    Suggest parameters for the trial using the search space defined in config.

    Args:
        trial: Optuna trial object

    Returns:
        Dictionary of suggested parameters
    """
    params = {}
    for param_name, param_spec in config.PARAM_SEARCH_SPACE.items():
        if isinstance(param_spec[0], list):  # Categorical parameter
            params[param_name] = trial.suggest_categorical(param_name, param_spec[0])
        elif param_spec[2]:  # Log scale
            params[param_name] = trial.suggest_float(param_name, param_spec[0], param_spec[1], log=True)
        elif isinstance(param_spec[0], int) and isinstance(param_spec[1], int):  # Integer parameter
            params[param_name] = trial.suggest_int(param_name, param_spec[0], param_spec[1])
        else:  # Float parameter
            params[param_name] = trial.suggest_float(param_name, param_spec[0], param_spec[1])
    return params


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
    # Wait if paused
    while config.should_pause:
        time.sleep(1)

    # Suggest parameters if not provided
    if params is None:
        params = _suggest_parameters(trial)

    # Train and evaluate model
    model, accuracy, auc, train_losses, val_losses, train_auc, val_auc = train_xgboost_model(
        params, X_train, X_val, y_train, y_val
    )

    # Calculate train-validation AUC difference (for overfitting detection)
    if train_auc and val_auc:
        auc_diff = abs(train_auc[-1] - val_auc[-1])
    else:
        auc_diff = 1.0  # High difference when AUC isn't available

    # Save all models that meet the criteria
    if accuracy >= config.MIN_ACCURACY_THRESHOLD and (auc_diff < config.MAX_AUC_DIFF_THRESHOLD):
        # Keep track of best model for reporting purposes
        if accuracy > config.best_accuracy:
            config.best_accuracy = accuracy
            config.best_auc_diff = auc_diff
            print(f"New best model found! Accuracy: {accuracy:.4f}, AUC diff: {auc_diff:.4f}")

        # Save the model
        save_model(model, config.MODEL_DIR, accuracy, auc_diff)

        # Plot learning curves for models that meet criteria
        plot_learning_curves(train_losses, val_losses, train_auc, val_auc,
                             len(X_train.columns), accuracy, auc)

    # Report trial progress
    print(f"Trial {trial.number}: Accuracy={accuracy:.4f}, AUC diff={auc_diff:.4f}")

    return accuracy


def optimize_model(X_train, X_val, y_train, y_val, n_trials=None):
    """
    Perform hyperparameter optimization for XGBoost model.

    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels
        n_trials: Number of Optuna trials to run (defaults to config.N_TRIALS)

    Returns:
        Best Optuna trial
    """
    if n_trials is None:
        n_trials = config.N_TRIALS

    print(f"Starting hyperparameter optimization with {n_trials} trials...")

    # Create Optuna study
    study = optuna.create_study(
        direction=config.OPTUNA_DIRECTION,
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


def train_model_with_top_features(X_train, X_val, y_train, y_val, model_path, num_top_features=None):
    """
    Train a model using only the top features from an existing model.

    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels
        model_path: Path to the existing model to extract features from
        num_top_features: Number of top features to use (defaults to config.TOP_FEATURES_COUNT)

    Returns:
        Path to the saved model with top features
    """
    if num_top_features is None:
        num_top_features = config.TOP_FEATURES_COUNT

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
    best_trial = optimize_model(X_train_top, X_val_top, y_train, y_val)

    # Train final model with best parameters
    best_params = best_trial.params
    print(f"Training final model with best parameters: {best_params}")

    best_model_top, accuracy_top, auc_top, train_losses, val_losses, train_auc, val_auc = train_xgboost_model(
        best_params, X_train_top, X_val_top, y_train, y_val
    )

    # Calculate AUC difference
    auc_diff = abs(train_auc[-1] - val_auc[-1]) if train_auc and val_auc else 1.0

    # Save model
    model_filename = save_model(best_model_top, config.MODEL_DIR, accuracy_top, auc_diff)

    print(f"Final model results:")
    print(f"  Validation accuracy: {accuracy_top:.4f}")
    print(f"  Validation AUC: {auc_top:.4f}")
    print(f"  AUC difference: {auc_diff:.4f}")
    print(f"Model saved to: {model_filename}")

    # Plot learning curves
    plot_learning_curves(train_losses, val_losses, train_auc, val_auc,
                         num_top_features, accuracy_top, auc_top)

    return model_filename
