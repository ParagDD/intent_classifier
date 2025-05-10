"""
Intent Classifier Package

A package for classifying email intents using RoBERTa.
"""

from .models.classifier import IntentClassifier
from .data.dataset import EmailDataset
from . import inference
from . import evaluate

__version__ = "0.1.0"
__all__ = ["IntentClassifier", "EmailDataset", "inference", "evaluate"] 