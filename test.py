import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import xgboost as xgb
import os

def preprocess_data(data):
    category_columns = [
        'result_fight_1', 'winner_fight_1', 'weight_class_fight_1', 'scheduled_rounds_fight_1',
        'result_b_fight_1', 'winner_b_fight_1', 'weight_class_b_fight_1', 'scheduled_rounds_b_fight_1',
        'result_fight_2', 'winner_fight_2', 'weight_class_fight_2', 'scheduled_rounds_fight_2',
        'result_b_fight_2', 'winner_b_fight_2', 'weight_class_b_fight_2', 'scheduled_rounds_b_fight_2',
        'result_fight_3', 'winner_fight_3', 'weight_class_fight_3', 'scheduled_rounds_fight_3',
        'result_b_fight_3', 'winner_b_fight_3', 'weight_class_b_fight_3', 'scheduled_rounds_b_fight_3'
    ]
    data[category_columns] = data[category_columns].astype("category")
    return data

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        model = xgb.XGBClassifier(enable_categorical=True)
        model.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def main():
    # Load test data
    test_data = pd.read_csv('data/train test data/test_data.csv')

    # Preprocess data
    display_columns = ['current_fight_date', 'fighter', 'fighter_b']
    y_test = test_data['winner']
    X_test = test_data.drop(['winner'] + display_columns, axis=1)
    X_test = preprocess_data(X_test)

    # Load model
    model_path = os.path.abspath('models/xgboost/jun2022-jun2024/model_0.7632_auc_diff_0.0994.json')
    model = load_model(model_path)

    # Ensure X_test has the same features as the model expects
    expected_features = model.get_booster().feature_names
    X_test = X_test.reindex(columns=expected_features)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.4f}")

    # Additional metrics
    true_positives = np.sum((y_test == 1) & (y_pred == 1))
    true_negatives = np.sum((y_test == 0) & (y_pred == 0))
    false_positives = np.sum((y_test == 0) & (y_pred == 1))
    false_negatives = np.sum((y_test == 1) & (y_pred == 0))

    print(f"True Positives: {true_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")

    # Calculate class-wise accuracy
    class_0_accuracy = true_negatives / (true_negatives + false_positives)
    class_1_accuracy = true_positives / (true_positives + false_negatives)

    print(f"Class 0 Accuracy: {class_0_accuracy:.4f}")
    print(f"Class 1 Accuracy: {class_1_accuracy:.4f}")


if __name__ == "__main__":
    main()