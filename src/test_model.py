from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AtriChatbot:
    def __init__(self, model_path="model/pre-trained"):
        """Khởi tạo chatbot với model đã train"""
        try:
            logger.info("Đang tải model và tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logger.info(f"Đã tải model thành công. Sử dụng device: {self.device}")
        except Exception as e:
            logger.error(f"Lỗi khi tải model: {str(e)}")
            raise

    def generate_response(self, user_input, max_length=100):
        """Tạo phản hồi cho input của người dùng"""
        try:
            # Tạo prompt với format phù hợp
            prompt = f"Human: {user_input}\nAssistant:"
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = inputs.to(self.device)

            # Tạo output
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # Decode và xử lý response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Lấy phần response của Assistant
            response = response.split("Assistant:")[-1].strip()
            
            return response

        except Exception as e:
            logger.error(f"Lỗi khi tạo response: {str(e)}")
            return "Xin lỗi, tôi gặp lỗi khi xử lý câu hỏi của bạn."

def main():
    # Khởi tạo chatbot
    chatbot = AtriChatbot()
    
    print("Atri: Xin chào! Tôi là Atri. Bạn có thể trò chuyện với tôi! (Gõ 'quit' để thoát)")
    
    while True:
        # Nhận input từ người dùng
        user_input = input("\nBạn: ").strip()
        
        # Kiểm tra điều kiện thoát
        if user_input.lower() == 'quit':
            print("\nAtri: Tạm biệt! Hẹn gặp lại bạn!")
            break
            
        # Tạo response
        response = chatbot.generate_response(user_input)
        print(f"\nAtri: {response}")

if __name__ == "__main__":
    main()