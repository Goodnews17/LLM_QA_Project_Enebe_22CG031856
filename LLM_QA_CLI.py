import os
import re
import string
from google import genai

# --- Configuration ---
# Uses the environment variable GEMINI_API_KEY for initialization
try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini Client. Ensure GEMINI_API_KEY is set.")
    # Exit or raise error if client cannot be initialized
    # For now, we'll continue and let it fail later if not set.

# --- Core Functions ---

def basic_preprocessing(question: str) -> str:
    """
    Applies basic preprocessing to the natural-language question.

    Requirements:
    - Lowercasing
    - Tokenization (handled implicitly by the LLM later, but we normalize the text)
    - Punctuation removal
    
    :param question: The raw natural-language question from the user.
    :return: The processed question string.
    """
    # 1. Lowercasing
    processed_q = question.lower()
    
    # 2. Punctuation removal
    # Create a translator to replace punctuation with a space
    # We use a space instead of removing it completely to avoid merging words 
    # (e.g., "hello,world" -> "helloworld")
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    processed_q = processed_q.translate(translator)
    
    # Optional: Normalize multiple spaces to a single space
    processed_q = re.sub(r'\s+', ' ', processed_q).strip()

    print(f"\n[DEBUG] Processed Question: '{processed_q}'")
    
    # For a Q&A system, the *original* question often yields better results 
    # when sent to the LLM, but we must meet the preprocessing requirement. 
    # We will send the *processed* question.
    return processed_q

def get_llm_answer(question: str) -> str:
    """
    Constructs a prompt and sends it to the Gemini LLM API.
    
    :param question: The processed question to send to the LLM.
    :return: The LLM's generated answer.
    """
    if not os.getenv("GEMINI_API_KEY"):
        return "ERROR: LLM API Key is not configured. Please set GEMINI_API_KEY environment variable."
        
    try:
        # Construct the prompt
        # A simple prompt can just be the question itself.
        # For better answers, we can add a system instruction:
        prompt = f"Answer the following question clearly and concisely:\n\n{question}"
        
        # Send to the LLM API
        # We'll use a fast model like gemini-2.5-flash
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        
        return response.text
        
    except Exception as e:
        return f"An error occurred during API call: {e}"


def run_cli_app():
    """
    The main loop for the CLI application.
    """
    print("🤖 LLM Q&A System CLI Application")
    print("Enter 'exit' or 'quit' to close the application.")
    
    while True:
        # Accept natural-language questions from the user
        raw_question = input("\nEnter your question: ")
        
        if raw_question.lower() in ['exit', 'quit']:
            break
            
        if not raw_question.strip():
            continue

        # 1. Apply basic preprocessing
        processed_question = basic_preprocessing(raw_question)
        
        # 2. Construct a prompt and send it to the chosen LLM API
        print("\n[INFO] Sending question to LLM...")
        answer = get_llm_answer(processed_question)
        
        # 3. Display the final answer to the user
        print("\n--- LLM ANSWER ---")
        print(answer)
        print("------------------")

if __name__ == "__main__":
    run_cli_app()