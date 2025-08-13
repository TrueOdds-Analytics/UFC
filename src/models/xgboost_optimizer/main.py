"""
Simplified UFC Fight Prediction Model - XGBoost Best Practices
Focus: Clean, effective training with proper validation and visualization
"""

import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import TimeSeriesSplit, cross_validate
import optuna
from optuna.samplers import TPESampler
import json

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')


# ============================================================================
# Simple Configuration
# ============================================================================

class Config:
    """Simplified configuration for XGBoost training"""

    # Paths
    train_path = '../../../data/train_test/train_data.csv'
    val_path = '../../../data/train_test/val_data.csv'
    model_dir = '../../../saved_models/xgboost/simplified/'

    # Training requirements
    min_accuracy = 0.60
    max_auc_diff = 0.15

    # XGBoost best practices
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 42,
        'enable_categorical': True,
    }

    # Hyperparameter tuning
    n_trials = 2  # Optuna trials
    n_cv_folds = 5  # Cross-validation folds

    def __init__(self):
        os.makedirs(self.model_dir, exist_ok=True)


# ============================================================================
# Data Handler
# ============================================================================

class DataHandler:
    """Simple data loading and preprocessing"""

    @staticmethod
    def load_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load and prepare data"""
        print("Loading data...")

        # Load CSVs
        train_df = pd.read_csv(config.train_path)
        val_df = pd.read_csv(config.val_path)

        # Separate features and target
        X_train = train_df.drop(['winner', 'fighter_a', 'fighter_b'], axis=1, errors='ignore')
        y_train = train_df['winner']
        X_val = val_df.drop(['winner', 'fighter_a', 'fighter_b'], axis=1, errors='ignore')
        y_val = val_df['winner']

        # Handle categorical columns
        for df in [X_train, X_val]:
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype('category')

        print(f"Data loaded: Train {X_train.shape}, Val {X_val.shape}")
        return X_train, X_val, y_train, y_val


# ============================================================================
# Model Trainer with Best Practices
# ============================================================================

class XGBoostTrainer:
    """Simplified XGBoost trainer with best practices"""

    def __init__(self, config: Config):
        self.config = config
        self.best_model = None
        self.best_params = None
        self.training_history = []

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val):
        """Optimize hyperparameters using Optuna with validation monitoring"""
        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION")
        print("="*60)

        def objective(trial):
            # XGBoost best practice parameters
            params = {
                **self.config.xgb_params,
                'n_estimators': 1000,  # Use early stopping instead
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0.01, 1.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            }

            # Train with early stopping
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )

            # Calculate metrics
            train_pred = model.predict_proba(X_train)[:, 1]
            val_pred = model.predict_proba(X_val)[:, 1]

            train_auc = roc_auc_score(y_train, train_pred)
            val_auc = roc_auc_score(y_val, val_pred)
            val_acc = accuracy_score(y_val, model.predict(X_val))
            auc_diff = abs(train_auc - val_auc)

            # Store for visualization
            self.training_history.append({
                'trial': trial.number,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'auc_diff': auc_diff,
                'params': params
            })

            # Print progress for qualifying trials
            if val_acc >= self.config.min_accuracy and auc_diff <= self.config.max_auc_diff:
                print(f"✓ Trial {trial.number}: Val Acc={val_acc:.3f}, Val AUC={val_auc:.3f}, AUC Diff={auc_diff:.3f}")
            else:
                print(f"✗ Trial {trial.number}: Val Acc={val_acc:.3f}, Val AUC={val_auc:.3f}, AUC Diff={auc_diff:.3f}")

            # Objective: Maximize val_auc while penalizing high AUC difference
            if auc_diff > self.config.max_auc_diff:
                return val_auc - (auc_diff - self.config.max_auc_diff) * 2
            else:
                return val_auc

        # Run optimization
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=True)

        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best value: {study.best_value:.4f}")

        return study.best_params

    def train_final_model(self, X_train, y_train, X_val, y_val, params):
        """Train final model with best parameters"""
        print("\n" + "="*60)
        print("TRAINING FINAL MODEL")
        print("="*60)

        # Combine best params with defaults
        final_params = {
            **self.config.xgb_params,
            **params,
            'n_estimators': 1000,
        }

        # Train with early stopping
        self.best_model = xgb.XGBClassifier(**final_params)

        eval_result = {}
        self.best_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            early_stopping_rounds=50,
            verbose=True,
            callbacks=[xgb.callback.record_evaluation(eval_result)]
        )

        # Calculate final metrics
        train_pred = self.best_model.predict_proba(X_train)[:, 1]
        val_pred = self.best_model.predict_proba(X_val)[:, 1]

        metrics = {
            'train_acc': accuracy_score(y_train, self.best_model.predict(X_train)),
            'val_acc': accuracy_score(y_val, self.best_model.predict(X_val)),
            'train_auc': roc_auc_score(y_train, train_pred),
            'val_auc': roc_auc_score(y_val, val_pred),
            'auc_diff': abs(roc_auc_score(y_train, train_pred) - roc_auc_score(y_val, val_pred)),
            'eval_history': eval_result
        }

        # Print results
        print(f"\nFinal Model Performance:")
        print(f"  Train Accuracy: {metrics['train_acc']:.4f}")
        print(f"  Val Accuracy: {metrics['val_acc']:.4f}")
        print(f"  Train AUC: {metrics['train_auc']:.4f}")
        print(f"  Val AUC: {metrics['val_auc']:.4f}")
        print(f"  AUC Difference: {metrics['auc_diff']:.4f}")

        # Check requirements
        if metrics['val_acc'] >= self.config.min_accuracy:
            print(f"  ✓ Accuracy requirement met (>= {self.config.min_accuracy})")
        else:
            print(f"  ✗ Accuracy below requirement (< {self.config.min_accuracy})")

        if metrics['auc_diff'] <= self.config.max_auc_diff:
            print(f"  ✓ AUC difference requirement met (<= {self.config.max_auc_diff})")
        else:
            print(f"  ✗ AUC difference too high (> {self.config.max_auc_diff})")

        return metrics

    def cross_validate_model(self, X, y):
        """Perform time series cross-validation"""
        print("\n" + "="*60)
        print("CROSS-VALIDATION")
        print("="*60)

        tscv = TimeSeriesSplit(n_splits=self.config.n_cv_folds)

        # Use the best parameters
        model_params = {
            **self.config.xgb_params,
            **self.best_params,
            'n_estimators': 100,  # Faster for CV
        }

        model = xgb.XGBClassifier(**model_params)

        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y,
            cv=tscv,
            scoring=['accuracy', 'roc_auc'],
            return_train_score=True,
            n_jobs=-1
        )

        # Calculate statistics
        results = {
            'train_acc': cv_results['train_accuracy'].mean(),
            'val_acc': cv_results['test_accuracy'].mean(),
            'train_auc': cv_results['train_roc_auc'].mean(),
            'val_auc': cv_results['test_roc_auc'].mean(),
            'val_acc_std': cv_results['test_accuracy'].std(),
            'val_auc_std': cv_results['test_roc_auc'].std(),
        }

        print(f"Cross-Validation Results ({self.config.n_cv_folds} folds):")
        print(f"  Val Accuracy: {results['val_acc']:.4f} ± {results['val_acc_std']:.4f}")
        print(f"  Val AUC: {results['val_auc']:.4f} ± {results['val_auc_std']:.4f}")
        print(f"  Train-Val AUC Diff: {abs(results['train_auc'] - results['val_auc']):.4f}")

        return results


# ============================================================================
# Visualization
# ============================================================================

class Visualizer:
    """Simple, clear visualizations"""

    @staticmethod
    def plot_training_results(trainer: XGBoostTrainer, metrics: Dict):
        """Plot training curves and performance metrics"""

        fig = plt.figure(figsize=(15, 10))

        # 1. Training curves
        ax1 = plt.subplot(2, 3, 1)
        eval_history = metrics['eval_history']
        epochs = range(1, len(eval_history['validation_0']['logloss']) + 1)

        ax1.plot(epochs, eval_history['validation_0']['auc'], 'b-', label='Train AUC', linewidth=2)
        ax1.plot(epochs, eval_history['validation_1']['auc'], 'r-', label='Val AUC', linewidth=2)
        ax1.fill_between(epochs,
                         eval_history['validation_0']['auc'],
                         eval_history['validation_1']['auc'],
                         alpha=0.2, color='gray')
        ax1.set_xlabel('Boosting Round')
        ax1.set_ylabel('AUC')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Optimization history
        ax2 = plt.subplot(2, 3, 2)
        trial_history = trainer.training_history
        trials = [h['trial'] for h in trial_history]
        val_accs = [h['val_acc'] for h in trial_history]

        colors = ['green' if acc >= trainer.config.min_accuracy else 'red' for acc in val_accs]
        ax2.scatter(trials, val_accs, c=colors, alpha=0.6)
        ax2.axhline(y=trainer.config.min_accuracy, color='black', linestyle='--',
                   label=f'Min Accuracy ({trainer.config.min_accuracy})')
        ax2.set_xlabel('Trial')
        ax2.set_ylabel('Validation Accuracy')
        ax2.set_title('Hyperparameter Optimization')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. AUC Difference monitoring
        ax3 = plt.subplot(2, 3, 3)
        auc_diffs = [h['auc_diff'] for h in trial_history]
        colors = ['green' if diff <= trainer.config.max_auc_diff else 'red' for diff in auc_diffs]
        ax3.scatter(trials, auc_diffs, c=colors, alpha=0.6)
        ax3.axhline(y=trainer.config.max_auc_diff, color='black', linestyle='--',
                   label=f'Max AUC Diff ({trainer.config.max_auc_diff})')
        ax3.set_xlabel('Trial')
        ax3.set_ylabel('AUC Difference')
        ax3.set_title('Overfitting Monitor')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Feature importance (top 20)
        ax4 = plt.subplot(2, 3, 4)
        importance = trainer.best_model.feature_importances_
        feature_names = trainer.best_model.get_booster().feature_names

        # Get top 20 features
        indices = np.argsort(importance)[-20:]
        top_features = [feature_names[i] for i in indices]
        top_importance = importance[indices]

        ax4.barh(range(len(top_features)), top_importance)
        ax4.set_yticks(range(len(top_features)))
        ax4.set_yticklabels([f[:30] for f in top_features], fontsize=8)
        ax4.set_xlabel('Importance')
        ax4.set_title('Top 20 Features')
        ax4.grid(True, alpha=0.3)

        # 5. Performance summary
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')

        summary_text = f"""
        FINAL MODEL PERFORMANCE
        {'='*30}
        
        Accuracy:
        • Train: {metrics['train_acc']:.4f}
        • Validation: {metrics['val_acc']:.4f}
        
        AUC:
        • Train: {metrics['train_auc']:.4f}
        • Validation: {metrics['val_auc']:.4f}
        
        Overfitting Check:
        • AUC Difference: {metrics['auc_diff']:.4f}
        • Status: {'✓ PASS' if metrics['auc_diff'] <= trainer.config.max_auc_diff else '✗ FAIL'}
        
        Requirements:
        • Min Accuracy: {'✓' if metrics['val_acc'] >= trainer.config.min_accuracy else '✗'}
        • Max AUC Diff: {'✓' if metrics['auc_diff'] <= trainer.config.max_auc_diff else '✗'}
        """

        ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # 6. Loss curves
        ax6 = plt.subplot(2, 3, 6)
        ax6.plot(epochs, eval_history['validation_0']['logloss'], 'b-', label='Train Loss', linewidth=2)
        ax6.plot(epochs, eval_history['validation_1']['logloss'], 'r-', label='Val Loss', linewidth=2)
        ax6.set_xlabel('Boosting Round')
        ax6.set_ylabel('Log Loss')
        ax6.set_title('Loss Convergence')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.suptitle('XGBoost Model Training Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        return fig


# ============================================================================
# Model Saver
# ============================================================================

class ModelSaver:
    """Save model and metadata"""

    @staticmethod
    def save_model(trainer: XGBoostTrainer, config: Config, metrics: Dict, cv_results: Dict):
        """Save model with all relevant information"""

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save model
        model_filename = f"xgb_model_acc{metrics['val_acc']:.3f}_auc{metrics['val_auc']:.3f}_{timestamp}.json"
        model_path = os.path.join(config.model_dir, model_filename)
        trainer.best_model.save_model(model_path)
        print(f"\nModel saved to: {model_path}")

        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'performance': metrics,
            'cross_validation': cv_results,
            'best_params': trainer.best_params,
            'requirements_met': {
                'min_accuracy': metrics['val_acc'] >= config.min_accuracy,
                'max_auc_diff': metrics['auc_diff'] <= config.max_auc_diff
            },
            'feature_importance': {
                name: float(imp)
                for name, imp in zip(
                    trainer.best_model.get_booster().feature_names,
                    trainer.best_model.feature_importances_
                )
            }
        }

        metadata_path = os.path.join(config.model_dir, f"metadata_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")

        return model_path, metadata_path


# ============================================================================
# Main Pipeline
# ============================================================================

def train_xgboost_model():
    """Main training pipeline - simple and effective"""

    print("\n" + "="*60)
    print("UFC FIGHT PREDICTION - XGBOOST TRAINING")
    print("Simple, Effective, Best Practices")
    print("="*60)

    # 1. Setup
    config = Config()
    data_handler = DataHandler()

    # 2. Load data
    X_train, X_val, y_train, y_val = data_handler.load_data(config)

    # 3. Initialize trainer
    trainer = XGBoostTrainer(config)

    # 4. Optimize hyperparameters
    best_params = trainer.optimize_hyperparameters(X_train, y_train, X_val, y_val)
    trainer.best_params = best_params

    # 5. Train final model
    metrics = trainer.train_final_model(X_train, y_train, X_val, y_val, best_params)

    # 6. Cross-validation for robustness check
    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])
    cv_results = trainer.cross_validate_model(X_combined, y_combined)

    # 7. Visualize results
    visualizer = Visualizer()
    fig = visualizer.plot_training_results(trainer, metrics)

    # 8. Save model if requirements are met
    if metrics['val_acc'] >= config.min_accuracy and metrics['auc_diff'] <= config.max_auc_diff:
        print("\n✓ All requirements met! Saving model...")
        model_saver = ModelSaver()
        model_path, metadata_path = model_saver.save_model(trainer, config, metrics, cv_results)
        print("\n" + "="*60)
        print("TRAINING COMPLETE - MODEL SAVED")
        print("="*60)
    else:
        print("\n✗ Requirements not met. Adjust hyperparameters and try again.")
        print("  Consider:")
        print("  - Increasing regularization (reg_alpha, reg_lambda)")
        print("  - Reducing model complexity (max_depth, min_child_weight)")
        print("  - Using more training data")

    return trainer, metrics


# ============================================================================
# Run Training
# ============================================================================

if __name__ == "__main__":
    trainer, metrics = train_xgboost_model()