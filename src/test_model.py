import pickle
import pandas as pd
from datetime import datetime

def load_model():
    # Load model và vectorizer
    with open('models/chatbot_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def preprocess_text(text):
    return str(text).lower().strip()

def get_response(user_input, model, vectorizer):
    # Tiền xử lý input
    processed_input = preprocess_text(user_input)
    # Vectorize input
    input_vector = vectorizer.transform([processed_input])
    # Dự đoán câu trả lời
    response = model.predict(input_vector)[0]
    return response

def save_conversation_csv(conversation):
    # Create DataFrame from conversation
    df = pd.DataFrame(conversation)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/csv/conversation_{timestamp}.csv"
    
    # Save to CSV
    df.to_csv(filename, index=False)
    return filename

def main():
    model, vectorizer = load_model()
    
    # Initialize conversation list
    conversation = []
    
    print("Chatbot đã sẵn sàng! Gõ 'quit' để thoát.")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'quit':
            break
            
        # Add user message to conversation
        conversation.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Get and print response
        response = get_response(user_input, model, vectorizer)
        print("Bot:", response)
        
        # Add bot response to conversation
        conversation.append({
            "role": "assistant", 
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
    
    # Save conversation if there were any exchanges
    if conversation:
        save_conversation_csv(conversation)
        print("\nConversation saved in data/csv directory!")

if __name__ == "__main__":
    main()