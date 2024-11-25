import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

# Tạo thư mục models nếu chưa tồn tại
os.makedirs("models", exist_ok=True)

# Đọc dữ liệu training từ CSV
df = pd.read_csv('data/csv/conversation_20241125_154319.csv')

# Tiền xử lý dữ liệu
def preprocess_text(text):
    # Lowercase và loại bỏ dấu câu cơ bản
    text = str(text).lower().strip()
    return text

# Áp dụng tiền xử lý
df['processed_question'] = df['question'].apply(preprocess_text)
df['processed_answer'] = df['answer'].apply(preprocess_text)

# Vectorize câu hỏi sử dụng TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_question'])

# Training model sử dụng Naive Bayes
model = MultinomialNB()
model.fit(X, df['processed_answer'])

# Lưu model và vectorizer
with open('models/chatbot_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)