"""
Intent classification model implementation.
"""

import logging
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import os
import json

from ..config import (
    MODEL_NAME, NUM_FOLDS, NUM_EPOCHS, BATCH_SIZE,
    LEARNING_RATE, MAX_LENGTH, MODEL_SAVE_DIR,
    HYPERPARAMS_FILE, INTENTS_FILE, INTENTS,
    ID_TO_INTENT
)
from ..data.dataset import EmailDataset

logger = logging.getLogger(__name__)

class IntentClassifier:
    """
    Intent classification model using RoBERTa.
    
    This class implements the intent classification model using the RoBERTa
    architecture, with support for training, evaluation, and inference.
    """
    
    def __init__(self, model_name: str = MODEL_NAME, num_labels: int = len(INTENTS)):
        """
        Initialize the intent classifier.
        
        Args:
            model_name (str): Name of the pre-trained model to use
            num_labels (int): Number of intent classes
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)
        
        logger.info(f"Model initialized with {num_labels} labels")

    def predict(self, text: str) -> dict:
        """
        Predict the intent of a given text.
        
        Args:
            text (str): Input text to classify
            
        Returns:
            dict: Dictionary containing predicted intent and confidence scores
        """
        self.model.eval()
        
        # Tokenize input
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            
        # Get predicted class and confidence
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # Get all class probabilities
        class_probabilities = {
            ID_TO_INTENT[i]: prob.item()
            for i, prob in enumerate(probabilities[0])
        }
        
        return {
            'predicted_intent': ID_TO_INTENT[predicted_class],
            'confidence': confidence,
            'class_probabilities': class_probabilities
        }

    def load_model(self, model_dir: str = MODEL_SAVE_DIR):
        """
        Load a trained model from disk.
        
        Args:
            model_dir (str): Directory containing the saved model
        """
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} not found")
            
        self.model = RobertaForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_dir)
        self.model.to(self.device)
        logger.info(f"Model loaded from {model_dir}")

    def evaluate(self, df) -> dict:
        """
        Evaluate the model on a dataset.
        
        Args:
            df: DataFrame containing evaluation data
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        # Create dataset and dataloader
        eval_dataset = EmailDataset(
            df['email_text'].values,
            df['intent'].values,
            self.tokenizer,
            MAX_LENGTH
        )
        eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)
        
        # Get predictions
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for i, intent in enumerate(INTENTS):
            class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, labels=[i], average='binary'
            )
            per_class_metrics[intent] = {
                'precision': class_precision[0],
                'recall': class_recall[0],
                'f1': class_f1[0]
            }
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        return {
            'overall': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'per_class': per_class_metrics,
            'confusion_matrix': conf_matrix
        }

    def train_with_cross_validation(self, df, learning_rate: float = LEARNING_RATE,
                                  batch_size: int = BATCH_SIZE, max_length: int = MAX_LENGTH,
                                  n_splits: int = NUM_FOLDS) -> dict:
        """
        Train the model using k-fold cross-validation.
        
        Args:
            df: DataFrame containing the training data
            learning_rate (float): Learning rate for optimization
            batch_size (int): Batch size for training
            max_length (int): Maximum sequence length
            n_splits (int): Number of folds for cross-validation
            
        Returns:
            dict: Dictionary containing average metrics across folds
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        all_metrics = []

        logger.info("Starting cross-validation training...")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(df), 1):
            logger.info(f"Training fold {fold}/{n_splits}")
            
            train_dataset = EmailDataset(
                df.iloc[train_idx]['email_text'].values,
                df.iloc[train_idx]['intent'].values,
                self.tokenizer,
                max_length
            )
            
            val_dataset = EmailDataset(
                df.iloc[val_idx]['email_text'].values,
                df.iloc[val_idx]['intent'].values,
                self.tokenizer,
                max_length
            )

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
            
            # Training loop
            self.model.train()
            for epoch in range(NUM_EPOCHS):
                for batch in train_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            # Validation
            self.model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels']

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

                    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.numpy())

            # Calculate metrics
            accuracy = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='weighted'
            )
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            logger.info(f"Fold {fold} metrics: {metrics}")
            all_metrics.append(metrics)

        # Calculate average metrics
        avg_metrics = {
            metric: np.mean([m[metric] for m in all_metrics])
            for metric in all_metrics[0].keys()
        }
        
        logger.info(f"Average metrics across folds: {avg_metrics}")
        return avg_metrics

    def save_model(self, output_dir: str = MODEL_SAVE_DIR):
        """
        Save the model and related files.
        
        Args:
            output_dir (str): Directory to save the model
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save hyperparameters
        hyperparams = {
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'max_length': MAX_LENGTH,
            'model_name': MODEL_NAME
        }
        with open(os.path.join(output_dir, HYPERPARAMS_FILE), 'w') as f:
            json.dump(hyperparams, f)
        
        with open(os.path.join(output_dir, INTENTS_FILE), 'w') as f:
            f.write('\n'.join(INTENTS))
        
        logger.info(f"Model and tokenizer saved to {output_dir}") 