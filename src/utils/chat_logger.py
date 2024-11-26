from datetime import datetime
import os
import json
from enum import Enum
from pathlib import Path

class ChatType(Enum):
    GENERATIVE = "generative"
    NORMAL = "normal"
    ERROR = "error"

class ChatLogger:
    def __init__(self, base_path="data/raw/conversations/chat_logs"):
        self.base_path = Path(base_path)
        
    def _create_log_path(self, chat_type: ChatType) -> Path:
        now = datetime.now()
        year_dir = self.base_path / str(now.year)
        month_dir = year_dir / f"{now.month:02d}"
        
        # Tạo tên file theo format yêu cầu
        filename = f"{now.day:02d}-{now.hour:02d}_{now.minute:02d}_{now.second:02d}-{chat_type.value}.jsonl"
        
        # Tạo các thư mục nếu chưa tồn tại
        month_dir.mkdir(parents=True, exist_ok=True)
        
        return month_dir / filename
        
    def log_conversation(self, messages, chat_type: ChatType):
        log_path = self._create_log_path(chat_type)
        
        # Format messages thành dạng JSONL
        with open(log_path, 'a', encoding='utf-8') as f:
            for msg in messages:
                json_line = json.dumps({
                    'timestamp': datetime.now().isoformat(),
                    'role': msg['role'],
                    'content': msg['content']
                }, ensure_ascii=False)
                f.write(json_line + '\n')