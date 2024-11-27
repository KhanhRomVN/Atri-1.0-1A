from transformers import EncoderDecoderModel, BertTokenizer
import torch
import os

def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    model_path = "models/saved_models/best_model"
    tokenizer_path = "models/saved_models/best_tokenizer"
    
    model = EncoderDecoderModel.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    
    return model, tokenizer
    
def generate_response(model, tokenizer, input_text, max_length=128):
    """Generate response for given input text"""
    # Encode input text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
    
    # Generate response
    outputs = model.generate(
        inputs.input_ids,
        decoder_start_token_id=tokenizer.cls_token_id,
        max_length=max_length,
        min_length=10,  # Thêm độ dài tối thiểu
        num_beams=5,
        no_repeat_ngram_size=3,  # Tăng để tránh lặp từ
        length_penalty=1.0,  # Phạt câu dài
        early_stopping=True,
        bad_words_ids=[[tokenizer.unk_token_id]],  # Tránh sinh unknown tokens
        do_sample=True,  # Thêm sampling
        top_k=50,  # Giới hạn top k tokens
        top_p=0.95  # Nucleus sampling
    )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    
    print("\nBắt đầu chat với bot (gõ 'quit' để thoát):")
    print("-" * 50)
    
    while True:
        user_input = input("\nBạn: ")
        if user_input.lower() == 'quit':
            print("Tạm biệt!")
            break
            
        response = generate_response(model, tokenizer, user_input)
        print(f"Bot: {response}")
        print("-" * 50)

if __name__ == "__main__":
    main()