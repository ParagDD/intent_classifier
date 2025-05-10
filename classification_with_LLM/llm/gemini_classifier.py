import os
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Structured output format
class EmailIntent(BaseModel):
    email_text: str
    intent: str

# Load Gemini API key from environment variable for security
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Prompt template
PROMPT_TEMPLATE = '''# Email Intent Classification System

## Task
Analyze the email content and classify it into one of the following 8 business intents related to real estate operations:

1. Intent_Lease_Abstraction - Extract lease metadata and clauses (rent, term, landlord, tenant, renewal)
2. Intent_Comparison_LOI_Lease - Compare Letter of Intent with final lease to identify deviations  
3. Intent_Clause_Protect - Detect risky or missing lease clauses (subletting rights, break clauses)
4. Intent_Company_research - Research company details (credibility, litigation history)
5. Intent_Transaction_Date_navigator - Extract or schedule transaction-related dates
6. Intent_Amendment_Abstraction - Extract new terms from lease amendments and identify changes
7. Intent_Sales_Listings_Comparison - Compare pricing, sq ft, etc. across multiple sales listings
8. Intent_Lease_Listings_Comparison - Compare lease listings to identify best terms and conditions

## Instructions
1. Analyze the email content carefully
2. Consider the primary request being made
3. Identify keywords and phrases that signal specific intents
4. Classify the email into exactly ONE of the eight intent categories
5. Return ONLY the intent category as a single line response (e.g., "Intent_Lease_Abstraction")

## Examples

Email: "Hi team, Can you pull together a schedule of important dates for the escrow process on the 125 King St deal? We're especially concerned with closing and due diligence periods. Thanks!"
Classification: Intent_Transaction_Date_navigator

Email: "Hey, I'm reviewing the lease on the 3rd Avenue property. Can you check if there are any red flags—like missing indemnity clauses or unfavorable assignment terms?"
Classification: Intent_Clause_Protect

Email: "Please abstract the lease for the Johnson project (PDF attached). We need to know the base rent, commencement and expiry dates, renewal options, and escalation schedule."
Classification: Intent_Lease_Abstraction

Email: "Compare the signed lease to the LOI we submitted last month. I want to know what terms got changed or added, especially around TI allowances."
Classification: Intent_Comparison_LOI_Lease

Email: "Could you do a background check on Wexford Corp before we proceed? I'm particularly interested in any public disputes or bankruptcies in the past 5 years."
Classification: Intent_Company_research

Email: "Please summarize the changes from the latest amendment to the original lease for the Grandview Tower property."
Classification: Intent_Amendment_Abstraction

Email: "We have three broker sales packages for the 42nd St building. Can you give me a side-by-side summary of pricing, cap rate, and avg. PSF?"
Classification: Intent_Sales_Listings_Comparison

Email: "Compare the lease listings we received for Midtown West. Looking to identify which has the most favorable terms per sq ft."
Classification: Intent_Lease_Listings_Comparison

## Email to Classify:
{email_text}
'''

def classify_email_intent(email_text: str) -> str:
    """
    Classify the intent of a single email using Gemini LLM and the provided prompt.
    Returns the predicted intent as a single line string.
    """
    prompt = PROMPT_TEMPLATE.format(email_text=email_text)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
        config=types.GenerateContentConfig(
            max_output_tokens=100,
            temperature=0.1,
            response_mime_type="text/plain"
        ),
    )
    # Parse the response to extract the intent (should be a single line)
    result = response.text.strip()
    # If the model returns 'Classification: ...', extract only the intent
    if "Classification:" in result:
        result = result.split("Classification:")[-1].strip()
    return result

def batch_classify_email_intents(email_texts: List[str]) -> List[str]:
    """
    Classify a batch of emails. Returns a list of predicted intents.
    """
    return [classify_email_intent(text) for text in email_texts]

if __name__ == "__main__":
    # Example usage
    sample_email = "Please abstract the lease for the Johnson project (PDF attached). We need to know the base rent, commencement and expiry dates, renewal options, and escalation schedule."
    intent = classify_email_intent(sample_email)
    print(f"Predicted intent: {intent}") 