import sys
import pandas as pd
from intent_classifier import IntentClassifier
from intent_classifier.utils import preprocess_dataframe

# Load the trained model
model_dir = "intent_classifier_model"
classifier = IntentClassifier()
classifier.load_model(model_dir)

# Example usage: predict a single email text
sample_text = """
Dear team,\n\nPlease provide the lease abstraction for the new property at 123 Main St.\n\nThanks,\nJohn Doe\n"""

# Preprocess the text (optional, if you want to use the same cleaning as training)
from intent_classifier.utils import clean_email_text
cleaned_text = clean_email_text(sample_text)

# Predict
result = classifier.predict(cleaned_text)

print("Input Email:")
print(sample_text)
print("\nPredicted Intent:", result['predicted_intent'])
print("Confidence: {:.2f}".format(result['confidence']))
print("Class Probabilities:")
for intent, prob in result['class_probabilities'].items():
    print(f"  {intent}: {prob:.2f}")

# Optionally, predict for a batch of emails from a file
if len(sys.argv) > 1:
    input_file = sys.argv[1]
    df = pd.read_excel(input_file) if input_file.endswith('.xlsx') else pd.read_csv(input_file)
    df = preprocess_dataframe(df)
    predictions = [classifier.predict(text) for text in df['email_text']]
    df['predicted_intent'] = [p['predicted_intent'] for p in predictions]
    df['confidence'] = [p['confidence'] for p in predictions]
    df.to_csv('inference_results.csv', index=False)
    print(f"\nBatch inference complete. Results saved to inference_results.csv") 
