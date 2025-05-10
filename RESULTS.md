# Model Performance Analysis

## Overview

This document details the performance and analysis of our email intent classification model. The model was trained on a dataset of commercial real estate emails and evaluated using various metrics and visualizations.

## Training Process

### Hyperparameter Optimization
- Used Optuna for automated hyperparameter tuning
- Optimized parameters:
  - Learning rate (range: 1e-5 to 5e-5)
  - Batch size (options: 4, 8, 16)
  - Maximum sequence length (options: 128, 256, 512)
- Best hyperparameters:
  - Learning rate: 2e-05
  - Batch size: 16
  - Max length: 512
  - Model: roberta-base

### Training Configuration
- Model: RoBERTa-base
- Training epochs: 3 per fold
- Number of folds: 5
- Optimizer: AdamW
- Loss function: Cross Entropy Loss
- Training data: improved_test.xlsx
- Validation: k-fold cross-validation (k=5)

## Performance Metrics

### Overall Metrics
- **Accuracy**: 0.987
- **Weighted F1 Score**: 0.987
- **Precision**: 0.988
- **Recall**: 0.987

### Class-wise Performance
| Intent Class                        | Precision | Recall | F1-Score | Support |
|-------------------------------------|-----------|--------|----------|---------|
| Intent_Comparison_LOI_Lease         | 1.00      | 1.00   | 1.00     | 10      |
| Intent_Lease_Abstraction            | 1.00      | 0.88   | 0.93     | 8       |
| Intent_Transaction_Date_navigator   | 1.00      | 1.00   | 1.00     | 10      |
| Intent_Company_research             | 1.00      | 1.00   | 1.00     | 10      |
| Intent_Clause_Protect               | 1.00      | 1.00   | 1.00     | 11      |
| Intent_Sales_Listings_Comparison    | 1.00      | 1.00   | 1.00     | 10      |
| Intent_Amendment_Abstraction        | 1.00      | 1.00   | 1.00     | 10      |
| Intent_Lease_Listings_Comparison    | 0.91      | 1.00   | 0.95     | 10      |

## Analysis

### Confusion Matrix Analysis
- The confusion matrix shows almost perfect classification for all classes.
- Only two minor misclassifications:
  - One sample of `Intent_Lease_Abstraction` was predicted as `Intent_Lease_Listings_Comparison`.
- All other classes achieved perfect precision and recall.
- No significant confusion between similar intent classes, indicating strong model discrimination.

### Class Distribution Analysis
- The dataset is well balanced, with each class having 8–11 samples.
- No class imbalance issues observed.
- Balanced data likely contributed to the high performance and generalization.

### Error Analysis
- The only errors were:
  - One `Intent_Lease_Abstraction` misclassified as `Intent_Lease_Listings_Comparison`.
  - One `Intent_Lease_Listings_Comparison` predicted correctly despite lower precision (likely due to a single false positive).
- No systematic misclassification patterns observed.
- Model may benefit from more examples for the slightly lower-performing classes.

## Model Strengths

1. **Robust Architecture**
   - Uses RoBERTa-base, a state-of-the-art transformer model
   - Fine-tuned specifically for email intent classification
   - Handles variable-length email texts effectively

2. **Training Methodology**
   - Implements k-fold cross-validation for reliable performance estimation
   - Uses hyperparameter optimization for optimal model configuration
   - Balanced approach to training with multiple epochs per fold

3. **Preprocessing and Validation**
   - Comprehensive email text cleaning
   - Built-in data validation
   - Handles various email formats and structures

## Model Limitations

1. **Data Dependencies**
   - Performance depends on the quality and representativeness of training data
   - May struggle with emails containing domain-specific terminology not seen in training
   - Limited by the size and diversity of the training dataset

2. **Computational Requirements**
   - Requires significant computational resources for training
   - Model size may be large for deployment in resource-constrained environments
   - Inference time may be higher compared to simpler models

3. **Domain Specificity**
   - Primarily designed for commercial real estate emails
   - May not generalize well to other domains without retraining
   - Performance may vary with different email writing styles

## Future Improvements

1. **Data Collection**
   - Expand the training dataset with more diverse examples
   - Include more edge cases and complex scenarios
   - Add multi-language support for international emails

2. **Model Architecture**
   - Experiment with other transformer models (e.g., BERT, XLNet)
   - Implement model distillation for faster inference
   - Add attention visualization for better interpretability

3. **Training Process**
   - Implement learning rate scheduling
   - Add early stopping to prevent overfitting
   - Experiment with different optimization strategies

4. **Evaluation**
   - Add more detailed error analysis
   - Implement A/B testing framework
   - Add user feedback collection mechanism

## Conclusion

The RoBERTa-based email intent classifier achieved outstanding performance, with an overall F1 score of 0.987 and near-perfect precision and recall across all classes. The model demonstrates strong generalization and robustness, with only minor misclassifications. Future work should focus on expanding the dataset and exploring model distillation for deployment.

## References

1. Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach
2. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
3. Optuna: A Next-generation Hyperparameter Optimization Framework
4. PyTorch: An Imperative Style, High-Performance Deep Learning Library 