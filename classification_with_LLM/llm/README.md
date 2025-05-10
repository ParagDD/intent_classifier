# Gemini LLM Email Intent Classifier

This module uses Google's Gemini LLM to classify real estate-related emails into one of eight business intents using a structured prompt.

## Setup

1. **Install dependencies:**
   ```bash
   pip install google-genai pydantic
   ```
   Or use the provided requirements file:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your Gemini API key:**
   - Obtain your API key from Google AI Studio or your Gemini account.
   - Set it as an environment variable:
     ```bash
     export GEMINI_API_KEY=your_api_key_here
     ```
     (On Windows, use `set GEMINI_API_KEY=your_api_key_here`)

## Usage

### Single Email Classification
```python
from gemini_classifier import classify_email_intent

email = "Please abstract the lease for the Johnson project (PDF attached). We need to know the base rent, commencement and expiry dates, renewal options, and escalation schedule."
intent = classify_email_intent(email)
print(f"Predicted intent: {intent}")
```

### Batch Classification
```python
from gemini_classifier import batch_classify_email_intents

emails = [
    "Please abstract the lease for the Johnson project...",
    "Compare the signed lease to the LOI we submitted last month..."
]
results = batch_classify_email_intents(emails)
print(results)
```

## Output
- The classifier returns the predicted intent as a single line string, e.g., `Intent_Lease_Abstraction`.

## Notes
- The prompt and output format are designed for robust, business-specific intent classification.
- Ensure your API key is kept secure and not hard-coded in scripts.

## License
MIT License 