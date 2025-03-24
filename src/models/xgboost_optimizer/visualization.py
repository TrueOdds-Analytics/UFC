"""
Visualization functions for model analysis.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from model import load_model


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


def create_feature_importance_plot(model_path, X_train, top_n=50, print_count=20):
    """
    Create and display SHAP feature importance plot.

    Args:
        model_path: Path to the saved model file
        X_train: Training data used for SHAP analysis
        top_n: Number of top features to display in the plot
        print_count: Number of top features to print to console

    Returns:
        List of feature names sorted by importance
    """
    print(f"Creating SHAP feature importance plot for model: {model_path}")

    # Load the model
    model = load_model(model_path)

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