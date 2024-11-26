import json
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import torch
from datasets import Dataset
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load và xử lý dữ liệu từ file JSON"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Chuyển đổi conversations thành format phù hợp
    formatted_data = []
    for conv in data['conversations']:
        # Kết hợp system prompt với các đoạn hội thoại
        text = conv['system'] + "\n"
        for message in conv['conversations']:
            if message['from'] == 'human':
                text += f"Human: {message['value']}\n"
            else:
                text += f"Assistant: {message['value']}\n"
        formatted_data.append({'text': text})
    
    return Dataset.from_list(formatted_data)

def train_model():
    # Khởi tạo model và tokenizer
    model_name = "facebook/opt-350m"  # hoặc model khác tùy chọn
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Thêm special tokens nếu cần
    special_tokens = {
        'additional_special_tokens': ['Human:', 'Assistant:']
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # Load dữ liệu
    train_data = load_data('data/processed/train.json')

    # Tokenize dữ liệu
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding='max_length'
        )

    tokenized_data = train_data.map(tokenize_function, batched=True)

    # Thiết lập training arguments
    training_args = TrainingArguments(
        output_dir="model/pre-trained",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_dir='logs',
        logging_steps=100,
        learning_rate=2e-5,
    )

    # Khởi tạo data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Khởi tạo trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        data_collator=data_collator,
    )

    # Bắt đầu training
    try:
        logger.info("Bắt đầu training model...")
        trainer.train()
        
        # Lưu model và tokenizer
        output_dir = "model/pre-trained"
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Đã lưu model và tokenizer vào {output_dir}")
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()