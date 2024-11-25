from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from datetime import datetime
import csv

class SimpleBot:
    def __init__(self, model_name="facebook/blenderbot-400M-distill"):
        # Khởi tạo model và tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.chat_history_file = "chat_history.csv"
        
        # Tạo file CSV nếu chưa tồn tại
        self._initialize_csv()

    def _initialize_csv(self):
        try:
            with open(self.chat_history_file, 'x', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'user_input', 'bot_response'])
        except FileExistsError:
            pass

    def _save_conversation(self, user_input, bot_response):
        # Lưu cuộc hội thoại vào CSV
        with open(self.chat_history_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), user_input, bot_response])

    def generate_response(self, user_input):
        # Tạo response từ model
        inputs = self.tokenizer(user_input, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=1000,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Lưu cuộc hội thoại
        self._save_conversation(user_input, response)
        
        return response

    def chat(self):
        print("Bot: Hi! How can I help you today?")
        
        while True:
            user_input = input("You: ")
            
            if user_input.lower() == 'quit':
                print("Bot: See you next time!")
                break
                
            response = self.generate_response(user_input)
            print(f"Bot: {response}")

# Khởi tạo và chạy bot
if __name__ == "__main__":
    bot = SimpleBot()
    bot.chat()