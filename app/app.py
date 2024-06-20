from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import fitz 
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


model_name = "distilbert-base-uncased"
model_path = "../models/distilbert_qa_finetuned.pt"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
model.load_state_dict(torch.load(model_path))


def predicted_answer(model, tokenizer, context, question):
    inputs = tokenizer(question, context, return_tensors='pt', truncation=True, padding=True, return_offsets_mapping=True)
    offset_mapping = inputs.pop('offset_mapping').cpu().numpy()[0]
    with torch.no_grad():
        outputs = model(**inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores)
    actual_start_idx = int(offset_mapping[start_idx][0])  
    actual_end_idx = int(offset_mapping[end_idx][1])      
    
    predicted_answer = tokenizer.decode(inputs['input_ids'][0][start_idx:end_idx+1], skip_special_tokens=True)

    return predicted_answer, actual_start_idx, actual_end_idx

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file uploaded'}), 400

    pdf_file = request.files['pdf']
    
    if pdf_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(pdf_file.filename)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf_file.save(pdf_path)

    context = extract_text_from_pdf(pdf_path)
    
    return jsonify({'text': context})

@app.route('/qa', methods=['POST'])
def qa():
    if 'question' not in request.form:
        return jsonify({'error': 'Please provide a question'}), 400

    question = request.form['question']
    context = request.form.get('text', '')

    answer, start, end = predicted_answer(model, tokenizer, context, question)
    return jsonify({'answer': answer, 'start_index': start, 'end_index': end})



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(pdf_file.filename)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf_file.save(pdf_path)

    text = extract_text_with_styling_from_pdf(pdf_path)
    return jsonify({'text': text})

def extract_text_with_styling_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text_with_styling = []

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        blocks = page.get_text('dict')['blocks']

        for b in blocks:
            for l in b['lines']:
                for s in l['spans']:
                    font_size = s['size']
                    font_name = s['font']
                    text = s['text']
                    style = ""

                    if s['flags'] & 1 << 0:  # bold
                        style += "font-weight: bold;"
                    if s['flags'] & 1 << 1:  # italic
                        style += "font-style: italic;"

                    text_with_styling.append(f'<span style="{style}">{text}</span>')

    return ' '.join(text_with_styling)


if __name__ == '__main__':
    app.run(debug=True)
