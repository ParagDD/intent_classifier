"""
Utility functions for model evaluation and visualization.
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def plot_confusion_matrix(y_true: List[str], y_pred: List[str], 
                         labels: List[str], title: str = "Confusion Matrix") -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true (List[str]): True labels
        y_pred (List[str]): Predicted labels
        labels (List[str]): List of unique labels
        title (str): Plot title
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def plot_class_distribution(distribution: Dict[str, int], 
                          title: str = "Class Distribution") -> None:
    """
    Plot class distribution.
    
    Args:
        distribution (Dict[str, int]): Dictionary mapping classes to counts
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.bar(distribution.keys(), distribution.values())
    plt.title(title)
    plt.xlabel('Intent Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def get_detailed_metrics(y_true: List[str], y_pred: List[str], 
                        labels: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Get detailed classification metrics.
    
    Args:
        y_true (List[str]): True labels
        y_pred (List[str]): Predicted labels
        labels (List[str]): List of unique labels
        
    Returns:
        Dict[str, Dict[str, float]]: Dictionary containing per-class metrics
    """
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    return report

def save_evaluation_results(metrics: Dict[str, Dict[str, float]], 
                          output_file: str) -> None:
    """
    Save evaluation metrics to a CSV file.
    
    Args:
        metrics (Dict[str, Dict[str, float]]): Evaluation metrics
        output_file (str): Path to output file
    """
    # Convert metrics to DataFrame
    df = pd.DataFrame(metrics).T
    
    # Save to CSV
    df.to_csv(output_file)
    logger.info(f"Saved evaluation metrics to {output_file}")

def plot_training_history(history: Dict[str, List[float]], 
                         title: str = "Training History") -> None:
    """
    Plot training history.
    
    Args:
        history (Dict[str, List[float]]): Dictionary containing training metrics
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    for metric, values in history.items():
        plt.plot(values, label=metric)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show() 