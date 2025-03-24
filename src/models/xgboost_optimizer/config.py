"""
Configuration settings for the XGBoost optimizer.
"""
import os

# File paths
DATA_DIR = '../../../data/train_test/'
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_data.csv')
VAL_DATA_PATH = os.path.join(DATA_DIR, 'val_data.csv')
MODEL_DIR = '../../../saved_models/xgboost/jan2024-dec2025/dynamicmatchup sorted/'
MODEL_DIR_425 = '../../saved_models/xgboost/jan2024-dec2025/dynamicmatchup sorted 425/'

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MODEL_DIR_425, exist_ok=True)

# Default XGBoost parameters
DEFAULT_XGB_PARAMS = {
    'tree_method': 'hist',
    'device': 'cuda',
    'objective': 'binary:logistic',
    'verbosity': 0,
    'n_jobs': -1,
    'n_estimators': 1000,
    'early_stopping_rounds': 150,
    'enable_categorical': True,
    'eval_metric': ['logloss', 'auc']
}

# Optuna optimization settings
N_TRIALS = 10000
OPTUNA_DIRECTION = 'maximize'

# Feature selection
TOP_FEATURES_COUNT = 425
EXISTING_MODEL_PATH = os.path.join(MODEL_DIR, 'model_0.7017_auc_diff_0.0111.json')

# Model saving criteria
MIN_ACCURACY_THRESHOLD = 0.70
MAX_AUC_DIFF_THRESHOLD = 0.1

# Global state variables - initialized here but can be modified at runtime
should_pause = False
best_accuracy = 0
best_auc = 0
best_auc_diff = 0

# Categorical columns in the dataset
CATEGORY_COLUMNS = [
    'result_fight_1', 'winner_fight_1', 'weight_class_fight_1', 'scheduled_rounds_fight_1',
    'result_b_fight_1', 'winner_b_fight_1', 'weight_class_b_fight_1', 'scheduled_rounds_b_fight_1',
    'result_fight_2', 'winner_fight_2', 'weight_class_fight_2', 'scheduled_rounds_fight_2',
    'result_b_fight_2', 'winner_b_fight_2', 'weight_class_b_fight_2', 'scheduled_rounds_b_fight_2',
    'result_fight_3', 'winner_fight_3', 'weight_class_fight_3', 'scheduled_rounds_fight_3',
    'result_b_fight_3', 'winner_b_fight_3', 'weight_class_b_fight_3', 'scheduled_rounds_b_fight_3'
]

# Optuna parameter search space
PARAM_SEARCH_SPACE = {
    'lambda': (25, 30, True),  # (min, max, log)
    'alpha': (25, 30, True),
    'min_child_weight': (1, 10.0, False),
    'max_depth': (1, 6, False),
    'max_delta_step': (0, 10, False),
    'subsample': (0.5, 1.0, False),
    'colsample_bytree': (0.5, 1.0, False),
    'gamma': (1e-3, 10.0, True),
    'eta': (0.01, 0.1, True),
    'grow_policy': (['depthwise', 'lossguide'], False)  # (categories, is_range)
}