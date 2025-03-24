"""
XGBoost model training and evaluation functions.
"""
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from config import DEFAULT_XGB_PARAMS


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

    # Update params with defaults if not already set
    for key, value in DEFAULT_XGB_PARAMS.items():
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


def save_model(model, model_dir, accuracy, auc_diff):
    """
    Save a trained model to disk.

    Args:
        model: Trained XGBoost model
        model_dir: Directory to save the model
        accuracy: Model accuracy to include in filename
        auc_diff: AUC difference to include in filename

    Returns:
        Path to the saved model file
    """
    model_filename = f'{model_dir}model_{accuracy:.4f}_auc_diff_{auc_diff:.4f}.json'
    model.save_model(model_filename)
    print(f"Model saved: {model_filename}")
    return model_filename


def load_model(model_path):
    """
    Load a trained XGBoost model from disk.

    Args:
        model_path: Path to the saved model file

    Returns:
        Loaded XGBoost model
    """
    model = xgb.XGBClassifier(enable_categorical=True)
    model.load_model(model_path)
    return model