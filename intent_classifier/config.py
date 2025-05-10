"""
Configuration settings for the intent classifier.
"""

# Model configuration
MODEL_NAME = 'roberta-base'
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
NUM_FOLDS = 5

# Intent categories
INTENTS = [
    'Intent_Lease_Abstraction',
    'Intent_Comparison_LOI_Lease',
    'Intent_Clause_Protect',
    'Intent_Company_research',
    'Intent_Transaction_Date_navigator',
    'Intent_Amendment_Abstraction',
    'Intent_Sales_Listings_Comparison',
    'Intent_Lease_Listings_Comparison'
]

# Create label mapping
INTENT_TO_ID = {intent: idx for idx, intent in enumerate(INTENTS)}
ID_TO_INTENT = {idx: intent for intent, idx in INTENT_TO_ID.items()}

# Training configuration
RANDOM_SEED = 42
DEVICE = 'cuda'  # Will be updated to 'cpu' if CUDA is not available

# File paths
MODEL_SAVE_DIR = "intent_classifier_model"
HYPERPARAMS_FILE = "hyperparameters.json"
INTENTS_FILE = "intents.txt" 