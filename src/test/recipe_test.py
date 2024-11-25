import pickle
from sklearn.metrics.pairwise import cosine_similarity

def load_model():
    # Load model pipeline
    with open('src/models/recipe_chatbot.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load answer dictionary
    with open('src/models/recipe_answers.pkl', 'rb') as f:
        data = pickle.load(f)
    
    return model, data

def get_response(user_input, model, data):
    # Vectorize and predict using pipeline
    predicted_label = model.predict([user_input])[0]
    
    # Get vectorizer from pipeline
    vectorizer = model.named_steps['vectorizer']
    
    # Find similar questions
    user_vector = vectorizer.transform([user_input])
    questions_vector = vectorizer.transform(data['questions'])
    similarities = cosine_similarity(user_vector, questions_vector)[0]
    
    # Filter by predicted label
    label_indices = [i for i, label in enumerate(data['labels']) 
                    if label == predicted_label]
    
    # Get most similar question index
    label_similarities = [(i, similarities[i]) for i in label_indices]
    best_match_idx = max(label_similarities, key=lambda x: x[1])[0]
    
    return data['answers'][best_match_idx]

def chat():
    print("Recipe Chatbot initialized! Type 'quit' to exit.")
    model, data = load_model()
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            break
            
        try:
            response = get_response(user_input, model, data)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Bot: I'm sorry, I couldn't understand that. Could you rephrase your question?")

if __name__ == "__main__":
    chat()