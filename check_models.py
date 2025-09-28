import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load the .env file to get the API key
load_dotenv()

try:
    # Configure the API with your key
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    print("Successfully configured API. Checking for available models...\n")

    # List the models and print their names
    for model in genai.list_models():
        # We only care about models that support the 'generateContent' method
        if 'generateContent' in model.supported_generation_methods:
            print(model.name)

except Exception as e:
    print(f"An error occurred: {e}")