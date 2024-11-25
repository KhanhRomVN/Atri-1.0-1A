from flask import Flask, request, jsonify
from src.test_model import load_model, get_response
from datetime import datetime

app = Flask(__name__)

# Load model khi khởi động server
model, vectorizer = load_model()

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
            
        # Lấy response từ model
        bot_response = get_response(user_message, model, vectorizer)
        
        # Tạo response object
        response_data = {
            'response': bot_response,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)