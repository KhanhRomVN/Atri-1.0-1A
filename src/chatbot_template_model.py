from transformers import AutoTokenizer, BlenderbotSmallForConditionalGeneration
from datetime import datetime
import os
import csv

class Chatbot:
    def __init__(self):
        # Initialize model and tokenizer
        self.model_name = "facebook/blenderbot_small-90M"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = BlenderbotSmallForConditionalGeneration.from_pretrained(self.model_name)
        
        # Create data directory if it doesn't exist
        os.makedirs("data/csv", exist_ok=True)
        
    def generate_response(self, user_input):
        # Encode the input text
        inputs = self.tokenizer([user_input], return_tensors="pt")
        
        # Generate response
        reply_ids = self.model.generate(**inputs)
        
        # Decode the response
        response = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
        return response
            
    def save_conversation_csv(self, conversation):
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/raw/conversations/chat_logs/conversation_chat_logs_{timestamp}.csv"
        
        # Save conversation to CSV file
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['question', 'answer'])  # Write header
            # Write conversation pairs
            for i in range(0, len(conversation), 2):
                if i + 1 < len(conversation):  # Check if there's a pair
                    question = conversation[i]['content']
                    answer = conversation[i + 1]['content']
                    writer.writerow([question, answer])

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
        chatbot.save_conversation_csv(conversation)
        print("\nConversation saved in data/csv directory!")

if __name__ == "__main__":
    main()