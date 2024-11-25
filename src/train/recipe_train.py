import json
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def load_training_data():
    with open('data/train/recipe_qa.json', 'r') as f:
        qa_pairs = json.load(f)
    
    questions = [pair['question'] for pair in qa_pairs]
    labels = [pair['label'] for pair in qa_pairs]
    answers = [pair['answer'] for pair in qa_pairs]
    
    return questions, labels, answers

def train_model():
    # Load and prepare data
    questions, labels, answers = load_training_data()
    
    # Create pipeline with vectorizer and classifier
    model = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=5000)),
        ('classifier', MultinomialNB())
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        questions, labels, test_size=0.2, random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Save model as pkl
    with open('src/models/recipe_chatbot.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save answers dictionary
    answer_dict = {
        'questions': questions,
        'answers': answers,
        'labels': labels
    }
    
    with open('src/models/recipe_answers.pkl', 'wb') as f:
        pickle.dump(answer_dict, f)
    
    # Print accuracy
    print(f"Model accuracy: {model.score(X_test, y_test):.2f}")

if __name__ == "__main__":
    train_model()