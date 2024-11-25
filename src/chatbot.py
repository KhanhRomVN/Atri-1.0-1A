from transformers import AutoTokenizer, BlenderbotSmallForConditionalGeneration
import torch
import json
from datetime import datetime
import os

class Chatbot:
    def __init__(self):
        # Initialize model and tokenizer
        self.model_name = "facebook/blenderbot_small-90M"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = BlenderbotSmallForConditionalGeneration.from_pretrained(self.model_name)
        
        # Create data directory if it doesn't exist
        os.makedirs("data/raw", exist_ok=True)
        
    def generate_response(self, user_input):
        # Encode the input text
        inputs = self.tokenizer([user_input], return_tensors="pt")
        
        # Generate response
        reply_ids = self.model.generate(**inputs)
        
        # Decode the response
        response = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
        return response
        
    def save_conversation(self, conversation):
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/raw/conversation_{timestamp}.json"
        
        # Save conversation to JSON file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, indent=2, ensure_ascii=False)

def main():
    chatbot = Chatbot()
    conversation = []
    
    print("Chatbot initialized! Type 'quit' to end the conversation.")
    
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
        
        # Generate and print response
        response = chatbot.generate_response(user_input)
        print("Bot:", response)
        
        # Add bot response to conversation
        conversation.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
    
    # Save conversation if there were any exchanges
    if conversation:
        chatbot.save_conversation(conversation)
        print("\nConversation saved in data/raw directory!")

if __name__ == "__main__":
    main()