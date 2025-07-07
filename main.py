import torch
import base64
import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the model (global variables)
model = None
processor = None
device = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_model():
    """Initialize the OCR model"""
    global model, processor, device
    
    if model is None:
        print("Initializing OCR model...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "allenai/olmOCR-7B-0225-preview", 
            torch_dtype=torch.bfloat16
        ).eval()
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Model loaded on device: {device}")

def process_document(file_path, file_type):
    """Process document and extract text using OlmOCR"""
    try:
        initialize_model()
        
        if file_type.lower() == 'pdf':
            # Render PDF to image
            image_base64 = render_pdf_to_base64png(file_path, 1, target_longest_image_dim=1024)
            anchor_text = get_anchor_text(file_path, 1, pdf_engine="pdfreport", target_length=4000)
        else:
            # For image files, convert to base64
            with open(file_path, 'rb') as f:
                image_data = f.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')
            anchor_text = ""  # No anchor text for images
        
        # Build the prompt
        prompt = build_finetuning_prompt(anchor_text)
        
        # Build the full prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ]

        # Apply the chat template and processor
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

        inputs = processor(
            text=[text],
            images=[main_image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for (key, value) in inputs.items()}

        # Generate the output
        output = model.generate(
            **inputs,
            temperature=0.8,
            max_new_tokens=50,
            num_return_sequences=1,
            do_sample=True,
        )

        # Decode the output
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output[:, prompt_length:]
        text_output = processor.tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )
        
        return {
            'success': True,
            'extracted_text': text_output[0],
            'file_type': file_type,
            'file_path': file_path
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'file_type': file_type,
            'file_path': file_path
        }

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'OCR Backend is running'})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and process document"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Determine file type
            file_type = filename.rsplit('.', 1)[1].lower()
            
            # Process the document
            result = process_document(file_path, file_type)
            
            return jsonify(result)
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pipeline-status', methods=['GET'])
def pipeline_status():
    """Get pipeline status and configuration"""
    return jsonify({
        'model_loaded': model is not None,
        'device': str(device) if device else None,
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'upload_folder': app.config['UPLOAD_FOLDER']
    })

if __name__ == '__main__':
    print("Starting OCR Backend Server...")
    print("Initializing model on startup...")
    initialize_model()
    print("Model initialized successfully!")
    print("Server starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
