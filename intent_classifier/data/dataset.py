"""
Dataset handling for email intent classification.
"""

import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
from ..config import INTENT_TO_ID

class EmailDataset(Dataset):
    """
    Dataset class for email intent classification.
    
    This class handles the loading and preprocessing of email data for training
    and inference using the RoBERTa model.
    
    Args:
        texts (list): List of email texts
        labels (list): List of intent labels
        tokenizer (RobertaTokenizer): RoBERTa tokenizer instance
        max_length (int): Maximum sequence length for tokenization
    """
    
    def __init__(self, texts, labels, tokenizer: RobertaTokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = [INTENT_TO_ID[label] for label in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: Dictionary containing input_ids, attention_mask, and labels
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        } 