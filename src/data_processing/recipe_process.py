import pandas as pd
import json
import re
import os

def clean_text(text):
    # Remove special characters and extra spaces
    text = re.sub(r'[^\w\s]', ' ', str(text))
    text = ' '.join(text.split())
    return text.lower()

def generate_qa_pairs(df):
    qa_pairs = []
    
    for _, row in df.iterrows():
        # Basic recipe information questions
        qa_pairs.extend([
            {
                "question": f"What are the ingredients for {row['title']}?",
                "answer": f"The ingredients for {row['title']} are: {row['ingredients']}",
                "label": "ingredients"
            },
            {
                "question": f"How do I make {row['title']}?",
                "answer": f"Here are the directions for {row['title']}: {row['directions']}",
                "label": "directions"
            },
            {
                "question": f"What is the cooking time for {row['title']}?",
                "answer": f"The total cooking time for {row['title']} is {row['total_time']}",
                "label": "cooking_time"
            },
            {
                "question": f"How many calories are in {row['title']}?",
                "answer": f"{row['title']} contains {row['calories']} calories per serving",
                "label": "nutrition"
            }
        ])
        
        # Generate variations of questions
        qa_pairs.extend([
            {
                "question": f"Can you tell me how to prepare {row['title']}?",
                "answer": f"Here are the directions for {row['title']}: {row['directions']}",
                "label": "directions"
            },
            {
                "question": f"What ingredients do I need for {row['title']}?",
                "answer": f"The ingredients for {row['title']} are: {row['ingredients']}",
                "label": "ingredients"
            }
        ])

    return qa_pairs

def process_recipe_data():
    # Create directories if they don't exist
    os.makedirs('data/train', exist_ok=True)
    
    # Read the CSV file with automatic delimiter detection
    df = pd.read_csv('data/raw/specialized/recipe.csv', 
                     sep=',',  # Using comma as delimiter
                     encoding='utf-8',
                     on_bad_lines='skip')  # Skip problematic lines
    
    # Generate QA pairs
    qa_pairs = generate_qa_pairs(df)
    
    # Save processed data
    with open('data/train/recipe_qa.json', 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=4, ensure_ascii=False)
    
    print(f"Processed {len(qa_pairs)} QA pairs from {len(df)} recipes")

if __name__ == "__main__":
    process_recipe_data()