import pandas as pd
from datetime import datetime
import json

class ConversationAnalyzer:
    def __init__(self):
        self.log_file = "data/conversation_logs.jsonl"
        
    def log_conversation(self, conversation):
        with open(self.log_file, 'a', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'conversation': conversation
            }, f)
            f.write('\n')
    
    def analyze_conversations(self):
        # Đọc logs
        df = pd.read_json(self.log_file, lines=True)
        
        # Phân tích độ dài câu trả lời
        response_lengths = []
        for conv in df['conversation']:
            responses = [msg['content'] for msg in conv if msg['role'] == 'assistant']
            lengths = [len(resp.split()) for resp in responses]
            response_lengths.extend(lengths)
            
        return {
            'avg_response_length': sum(response_lengths) / len(response_lengths),
            'max_response_length': max(response_lengths),
            'min_response_length': min(response_lengths)
        }