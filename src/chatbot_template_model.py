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
        
        # Set chat template
        self.tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}\n"
            "{% elif message['role'] == 'user' %}"
            "User: {{ message['content'] }}\n"
            "{% elif message['role'] == 'assistant' %}"
            "Assistant: {{ message['content'] }}\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}Assistant: {% endif %}"
        )
        
        # System prompt
        self.system_prompt = """Tôi là một trợ lý AI thân thiện và hữu ích. Tôi sẽ cố gắng trả lời một cách 
        tự nhiên và đầy đủ nhất có thể. Tôi sẽ sử dụng ngôn ngữ phù hợp với văn hóa và bối cảnh của người dùng."""
        
        # Create data directory if it doesn't exist 
        os.makedirs("data/raw/conversations/chat_logs", exist_ok=True)
        
    def generate_response(self, user_input, conversation_history=None):
        # Format conversation with system prompt
        if conversation_history is None:
            conversation_history = []
            
        messages = [
            {"role": "system", "content": self.system_prompt},
            *conversation_history,
            {"role": "user", "content": user_input}
        ]
        
        # Apply chat template
        formatted_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate with adjusted parameters
        inputs = self.tokenizer([formatted_input], return_tensors="pt")
        reply_ids = self.model.generate(
            **inputs,
            max_length=150,
            temperature=0.85,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1
        )
        
        response = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
        return response
        
    def save_conversation(self, conversation):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/raw/conversations/chat_logs/conversation_{timestamp}.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['role', 'content', 'timestamp'])
            for message in conversation:
                writer.writerow([
                    message['role'],
                    message['content'],
                    message.get('timestamp', '')
                ])

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
        
        try:
            # Get bot response
            response = chatbot.generate_response(user_input, conversation)
            print(f"Bot: {response}")
            
            # Add bot response to conversation
            conversation.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Bot: I'm sorry, I couldn't understand that. Could you rephrase your question?")
    
    # Save conversation at the end
    if conversation:
        chatbot.save_conversation(conversation)
        print("\nConversation saved!")

if __name__ == "__main__":
    main()