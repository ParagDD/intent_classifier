# Email Intent Classification

A deep learning-based system for classifying email intents using RoBERTa. This project implements a multi-class classifier that can identify various intents in commercial real estate emails.

## Overview

This system uses a fine-tuned RoBERTa model to classify emails into different intents such as:
- Lease Abstraction
- Comparison of LOI/Lease
- Clause Protection
- Company Research
- Transaction Date Navigation
- Amendment Abstraction
- Sales Listings Comparison
- Lease Listings Comparison

## Features

- **Advanced Model**: Uses RoBERTa-base for state-of-the-art text classification
- **Robust Training**: Implements k-fold cross-validation for reliable performance
- **Hyperparameter Optimization**: Uses Optuna for automated hyperparameter tuning
- **Comprehensive Evaluation**: Provides detailed metrics and visualizations
- **Easy Inference**: Simple API for both single and batch predictions
- **Data Preprocessing**: Built-in email text cleaning and validation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ParagDD/intent_classifier.git
cd intent_classifier
```

2. Install the package:
```bash
pip install -e .
```

## Usage

### Training

```python
from intent_classifier import IntentClassifier
import pandas as pd

# Load your data
df = pd.read_excel('your_data.xlsx')

# Initialize and train the classifier
classifier = IntentClassifier()
metrics = classifier.train_with_cross_validation(df)

# Save the model
classifier.save_model('model_directory')
```

### Inference

```python
from intent_classifier import IntentClassifier

# Load the trained model
classifier = IntentClassifier()
classifier.load_model('model_directory')

# Predict a single email
result = classifier.predict("Please provide the lease abstraction for 123 Main St.")
print(f"Predicted Intent: {result['predicted_intent']}")
print(f"Confidence: {result['confidence']}")
```

### Evaluation

```python
from intent_classifier import evaluate

# Run evaluation on test data
evaluate.main()
```

## Running Evaluation and Inference

### Evaluation
To evaluate the trained model on your dataset and generate metrics and visualizations:

```bash
cd intent_classifier
python evaluate.py
```

- This will print a classification report, plot the confusion matrix, and show class distribution.
- Make sure your evaluation data (e.g., `improved_test.xlsx`) is in the project directory.

### Inference
To predict the intent of a new email or a batch of emails:

```bash
cd intent_classifier
python inference.py
```

- By default, the script will run a sample prediction and print the result.
- To run batch inference on a file:

```bash
python inference.py your_emails.xlsx
```

- The results will be saved to `inference_results.csv` in the same directory.

**Note:**
- Ensure the trained model files are present in the `intent_classifier_model` directory before running these scripts.
- You can modify the scripts to point to different model or data file locations as needed.

## Model Performance

### Metrics
- **Accuracy:** 0.987
- **Weighted F1 Score:** 0.987
- **Precision:** 0.988
- **Recall:** 0.987

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

### Confusion Matrix
- The model achieved near-perfect classification for all classes.
- Only one minor misclassification: a single `Intent_Lease_Abstraction` sample was predicted as `Intent_Lease_Listings_Comparison`.
- No significant confusion between similar intent classes.

### Class Distribution
- The dataset is well balanced, with each class having 8–11 samples.
- No class imbalance issues observed.

### Summary
The RoBERTa-based email intent classifier achieved outstanding performance, with an overall F1 score of 0.987 and near-perfect precision and recall across all classes. The model demonstrates strong generalization and robustness, with only minor misclassifications.

## Project Structure

```
intent_classifier/
├── intent_classifier/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifier.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   └── evaluation.py
│   ├── __init__.py
│   ├── inference.py
│   └── evaluate.py
├── tests/
├── README.md
├── requirements.txt
└── setup.py
```

## Requirements

- Python 3.8+
- PyTorch >= 1.9.0
- Transformers >= 4.11.0
- Pandas >= 1.3.0
- NumPy >= 1.19.5
- scikit-learn >= 0.24.2
- Optuna >= 2.10.0
- Matplotlib >= 3.4.3
- Seaborn >= 0.11.2

## Future Improvements

1. Add support for more email formats
2. Implement real-time email processing
3. Add API endpoints for web integration
4. Enhance preprocessing for better handling of special cases
5. Add support for multi-language emails

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 