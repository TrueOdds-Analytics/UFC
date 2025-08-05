"""
Main script for running the XGBoost optimizer.
"""
from data_loader import load_and_preprocess_data
from optimization import optimize_model, train_model_with_top_features
from utils import start_user_input_thread
from visualization import create_feature_importance_plot
import config


def main():
    """
    Main execution function for the XGBoost optimizer.

    Pipeline:
    1. Load and preprocess data
    2. (Optional) Train initial model with all features
    3. Analyze feature importance from existing model
    4. Train optimized model with top features only
    """
    # Start user input thread for interactive control
    input_thread = start_user_input_thread()

    try:
        # Step 1: Load and preprocess data
        X_train, X_val, y_train, y_val = load_and_preprocess_data(odds_type='None')
        print("Data loaded. Starting optimization...")

        # Step 2: Train initial model with all features (optional)
        # Uncomment to run initial optimization
        # best_trial = optimize_model(X_train, X_val, y_train, y_val)

        # Step 3: Use an existing model for feature selection and visualization and Train model with top features only
        print("Creating feature importance visualization for existing model")
        top_features_model_path = train_model_with_top_features(
            X_train, X_val, y_train, y_val, config.EXISTING_MODEL_PATH
        )

        # print("Training pipeline completed successfully")
        # print(f"Final model saved at: {top_features_model_path}")

    except KeyboardInterrupt:
        print("Process interrupted by user")
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()