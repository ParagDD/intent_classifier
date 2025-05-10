import pandas as pd
from intent_classifier import IntentClassifier
from intent_classifier.utils import (
    preprocess_dataframe,
    validate_dataframe,
    plot_confusion_matrix,
    plot_class_distribution,
    get_detailed_metrics
)

# Load the trained model
model_dir = "intent_classifier_model"
classifier = IntentClassifier()
classifier.load_model(model_dir)

# Load and preprocess the evaluation data
input_file = "improved_test.xlsx"  # Change to your test/validation file if needed
df = pd.read_excel(input_file)
validate_dataframe(df)
df = preprocess_dataframe(df)

# Get true and predicted labels
y_true = df['intent'].tolist()
y_pred = [classifier.predict(text)['predicted_intent'] for text in df['email_text']]

# Print detailed metrics
labels = list(classifier.tokenizer.get_vocab().keys())  # Use intent list instead
labels = list(classifier.tokenizer.get_vocab().keys())
labels = list(set(y_true))  # Use only present labels
metrics = get_detailed_metrics(y_true, y_pred, labels)
print("\nClassification Report:")
for label in labels:
    print(f"{label}: {metrics[label]}")
print("\nOverall:")
print(metrics['weighted avg'])

# Plot confusion matrix
plot_confusion_matrix(y_true, y_pred, labels)

# Plot class distribution
from intent_classifier.utils import get_class_distribution
plot_class_distribution(get_class_distribution(df)) 