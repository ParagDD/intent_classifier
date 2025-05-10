"""
Utility functions for data preprocessing and validation.
"""

import re
import logging
from typing import List, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

def clean_email_text(text: str) -> str:
    """
    Clean email text by removing common email artifacts.
    
    Args:
        text (str): Raw email text
        
    Returns:
        str: Cleaned email text
    """
    # Remove email headers
    text = re.sub(r'From:.*?\n', '', text)
    text = re.sub(r'To:.*?\n', '', text)
    text = re.sub(r'Subject:.*?\n', '', text)
    text = re.sub(r'Date:.*?\n', '', text)
    
    # Remove email signatures
    text = re.sub(r'--\s*\n.*$', '', text, flags=re.DOTALL)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate the input DataFrame for training/evaluation.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        bool: True if validation passes, False otherwise
        
    Raises:
        ValueError: If validation fails
    """
    required_columns = ['email_text', 'intent']
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for empty values
    if df['email_text'].isnull().any():
        raise ValueError("Found empty email texts")
    if df['intent'].isnull().any():
        raise ValueError("Found empty intent labels")
    
    # Check for empty strings
    if (df['email_text'] == '').any():
        raise ValueError("Found empty email texts")
    if (df['intent'] == '').any():
        raise ValueError("Found empty intent labels")
    
    return True

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Clean email texts
    processed_df['email_text'] = processed_df['email_text'].apply(clean_email_text)
    
    return processed_df

def get_class_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get the distribution of intent classes in the dataset.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        Dict[str, int]: Dictionary mapping intent classes to their counts
    """
    return df['intent'].value_counts().to_dict() 