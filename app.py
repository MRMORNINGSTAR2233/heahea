from io import BytesIO
import PyPDF2
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import pytesseract
import openai
import fitz
import os

app = Flask(__name__)


# Configure OpenAI API key
openai.api_key = os.getenv("")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/mental')
def mental():
    return render_template('mental.html')

@app.route('/ask')
def disease():
    return render_template('RAG.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['user_message']
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are strictly a medical chatbot. Do not provide information outside of the medical domain. If a question isn't medical, inform the user and ask for a medical question."},
            {"role": "user", "content": user_message},
        ],
        max_tokens=100
    )
    ai_response = response['choices'][0]['message']['content']
    return jsonify({'ai_response': ai_response})

@app.route('/mental', methods=['POST'])
def mental_chat():
    user_message = request.form['user_message']
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a strictly a mental health chatbot and do not provide information outside of the mental health domain. If the question isn't mental health related, inform the user to ask for a mental health related question."},
            {"role": "user", "content": user_message},
        ],
        max_tokens=100
    )
    ai_response = response['choices'][0]['message']['content']
    return jsonify({'ai_response': ai_response})


@app.route('/ask', methods=['POST'])
def ask():
    # Get user question from the form
    user_question = request.form['question']

    # Handle PDF file upload
    pdf_file = request.files['pdf']
    pdf_text = extract_text_from_pdf(pdf_file)

    # Combine user question and PDF text for input to GPT-4
    input_text = f"{user_question} {pdf_text}"

    # Call GPT-4 for answering the question
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a medical assistant."},
            {"role": "user", "content": input_text}
        ]
    )
    ai_response = response['choices'][0]['message']['content']
    return jsonify({'answer': ai_response})

def extract_text_from_pdf(pdf_file):
    try:
        pdf_bytes = pdf_file.read()
        # Use PyMuPDF (fitz) to extract text from PDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


if __name__ == '__main__':
    app.run(debug=True)
