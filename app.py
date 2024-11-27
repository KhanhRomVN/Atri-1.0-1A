from flask import Flask, request, jsonify
from transformers import EncoderDecoderModel, BertTokenizer
import torch

app = Flask(__name__)

# Load model and tokenizer
model_path = "models/saved_models/best_model"
tokenizer_path = "models/saved_models/best_tokenizer"
model = EncoderDecoderModel.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    input_text = data.get('input_text', '')
    
    # Prepare input text
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True
    ).to(device)
    
    # Generate response
    outputs = model.generate(
        inputs.input_ids,
        decoder_start_token_id=tokenizer.cls_token_id,
        max_length=128,
        num_beams=5,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        early_stopping=True,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 