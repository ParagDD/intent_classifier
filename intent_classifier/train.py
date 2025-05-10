"""
Main training script for the intent classifier.
"""

import logging
import pandas as pd
import optuna
from optuna.trial import Trial

from .models.classifier import IntentClassifier
from .config import LEARNING_RATE, BATCH_SIZE, MAX_LENGTH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def objective(trial: Trial, df: pd.DataFrame) -> float:
    """
    Objective function for Optuna hyperparameter optimization.
    
    Args:
        trial (Trial): Optuna trial object
        df (pd.DataFrame): Training data
        
    Returns:
        float: F1 score for the trial
    """
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    max_length = trial.suggest_categorical('max_length', [128, 256, 512])

    logger.info(f"Learning rate: {learning_rate}, Batch size: {batch_size}, Max length: {max_length}")
    
    classifier = IntentClassifier()
    metrics = classifier.train_with_cross_validation(
        df,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_length=max_length
    )
    
    return metrics['f1']

def main():
    """Main training function."""
    # Load the data
    df = pd.read_excel('improved_test.xlsx')
    
    # Create Optuna study
    study = optuna.create_study(direction='maximize')
    
    # Run hyperparameter optimization
    logger.info("Starting hyperparameter optimization...")
    study.optimize(lambda trial: objective(trial, df), n_trials=20)
    
    # Get best hyperparameters
    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")
    
    # Train final model with best hyperparameters
    logger.info("Training final model with best hyperparameters...")
    final_classifier = IntentClassifier()
    final_metrics = final_classifier.train_with_cross_validation(
        df,
        learning_rate=best_params['learning_rate'],
        batch_size=best_params['batch_size'],
        max_length=best_params['max_length']
    )
    
    logger.info(f"Final model metrics: {final_metrics}")
    
    # Save the model
    final_classifier.save_model()
    logger.info("Training completed and model saved!")

if __name__ == "__main__":
    main() 