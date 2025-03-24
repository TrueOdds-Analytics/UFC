"""
Data preprocessing utilities for MMA betting analysis
"""
import os
import pandas as pd
import xgboost as xgb
from encoders import CategoryEncoder


def preprocess_data(data, encoder=None, fit=False):
    """
    Preprocess data with consistent categorical encodings

    Args:
        data: DataFrame to preprocess
        encoder: Optional CategoryEncoder instance
        fit: Whether to fit the encoder on this data

    Returns:
        Preprocessed DataFrame and encoder
    """
    # Create a new encoder if none provided
    if encoder is None:
        encoder = CategoryEncoder()

    # Apply categorical encoding
    if fit:
        # Fit and transform (for training/validation data)
        processed_data = encoder.fit_transform(data)
    else:
        # Only transform (for test/prediction data)
        processed_data = encoder.transform(data)

    return processed_data, encoder


def load_model(model_path, model_type='xgboost'):
    """
    Load a trained model from disk

    Args:
        model_path: Path to the model file
        model_type: Type of model to load

    Returns:
        Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(enable_categorical=True)
            model.load_model(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


def prepare_datasets(val_data_path, test_data_path, encoder_path, display_columns):
    """
    Load and prepare validation and test datasets

    Args:
        val_data_path: Path to validation data CSV
        test_data_path: Path to test data CSV
        encoder_path: Path to save/load the category encoder
        display_columns: Columns to keep for display purposes

    Returns:
        Tuple of (X_val, y_val, X_test, y_test, test_data_with_display, encoder)
    """
    # Load data
    val_data = pd.read_csv(val_data_path)
    test_data = pd.read_csv(test_data_path)

    # Separate target variables
    y_val, y_test = val_data['winner'], test_data['winner']

    # Check if encoder already exists
    if os.path.exists(encoder_path):
        # Load existing encoder
        print(f"Loading existing encoder from {encoder_path}")
        encoder = CategoryEncoder.load(encoder_path)

        # Process validation data with the loaded encoder (without refitting)
        X_val, _ = preprocess_data(
            val_data.drop(['winner'] + display_columns, axis=1),
            encoder=encoder,
            fit=False  # Don't fit again, just transform
        )
    else:
        # Initialize our category encoder and fit it
        print("Creating and fitting new encoder")
        encoder = CategoryEncoder()

        # First fit the encoder on validation data to learn consistent categorical mappings
        X_val, encoder = preprocess_data(
            val_data.drop(['winner'] + display_columns, axis=1),
            encoder=encoder,
            fit=True  # Important: fit the encoder on validation data
        )

        # Ensure the encoder directory exists
        os.makedirs(os.path.dirname(encoder_path), exist_ok=True)

        # Save the encoder for future use
        encoder.save(encoder_path)
        print(f"Encoder saved to {encoder_path}")

    # Now use the encoder to transform test data (without fitting again)
    X_test, _ = preprocess_data(
        test_data.drop(['winner'] + display_columns, axis=1),
        encoder=encoder,
        fit=False  # Don't fit on test data, just apply the mappings
    )

    # Concatenate features with display columns
    test_data_with_display = pd.concat([X_test, test_data[display_columns], y_test], axis=1)

    # Separate data into features and target variable
    X_test = test_data_with_display.drop(display_columns + ['winner'], axis=1)
    y_test = test_data_with_display['winner']

    return X_val, y_val, X_test, y_test, test_data_with_display, encoder