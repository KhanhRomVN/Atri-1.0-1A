import json
import csv
import os
from pathlib import Path

def load_conversations(json_path):
    """Load conversations from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['conversations']

def save_to_csv(conversations, csv_path):
    """Save conversations to CSV file"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['input', 'output'])
        # Write conversations
        for conv in conversations:
            writer.writerow([conv['input'], conv['output']])

def main():
    # Define paths relative to project root
    project_root = Path(__file__).parent.parent
    input_path = project_root / 'data/raw/conversations.json'
    output_path = project_root / 'data/processed/train_data.csv'
    
    # Process data
    conversations = load_conversations(input_path)
    save_to_csv(conversations, output_path)
    print(f"Processed {len(conversations)} conversations")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()