# app.py
from flask import Flask, render_template, request
from LLM_QA_CLI import basic_preprocessing, get_llm_answer

# Initialize Flask Application
app = Flask(__name__)

# --- Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles the main Q&A page.
    - GET: Displays the initial form.
    - POST: Processes the question and displays the results.
    """
    
    # Initialize variables for the template
    raw_question = None
    processed_question = None
    llm_response = None
    
    if request.method == 'POST':
        # 1. Get the question from the user
        raw_question = request.form.get('question_input')
        
        if raw_question:
            # 2. View the processed question
            processed_question = basic_preprocessing(raw_question)
            
            # 3. See the LLM API response / Display the generated answer
            llm_response = get_llm_answer(processed_question)
            
    # Render the template, passing the data to be displayed
    return render_template(
        'index.html',
        raw_question=raw_question,
        processed_question=processed_question,
        llm_response=llm_response
    )

if __name__ == '__main__':
    # Flask runs on port 5000 by default
    app.run(debug=True)