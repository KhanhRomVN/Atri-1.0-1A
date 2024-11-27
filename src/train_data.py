import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, EncoderDecoderModel, AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import numpy as np
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

# Custom dataset class
class ChatbotDataset(Dataset):
    def __init__(self, inputs, outputs, tokenizer, max_length=128):
        self.inputs = inputs    
        self.outputs = outputs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = str(self.inputs[idx])
        output_text = str(self.outputs[idx])

        # Tokenize input and output
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        output_encoding = self.tokenizer(
            output_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'decoder_input_ids': output_encoding['input_ids'].squeeze(),
            'decoder_attention_mask': output_encoding['attention_mask'].squeeze(),
            'labels': output_encoding['input_ids'].squeeze()
        }

def evaluate(model, val_loader, device):
    """
    Evaluate the model on the validation set
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            decoder_attention_mask = batch['decoder_attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    model.train()
    return avg_loss

def train_model(
    batch_size=2,
    num_epochs=10,
    learning_rate=5e-5,
    max_length=128,
    model_name='bert-base-multilingual-cased',
    train_data_path='data/processed/train_data.csv',
    model_save_dir='models/saved_models',
    validation_split=0.1,
    early_stopping_patience=3
):
    """
    Train the chatbot model with evaluation and early stopping
    """
    # Create model directory
    os.makedirs(model_save_dir, exist_ok=True)

    # Load and prepare data
    logging.info("Loading training data...")
    data = pd.read_csv(train_data_path)
    
    # Initialize tokenizer and model
    logging.info(f"Initializing tokenizer and model using {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Initialize encoder-decoder model
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        model_name,
        model_name
    )

    # Set special tokens
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Set decoder config
    model.config.decoder.is_decoder = True
    model.config.decoder.add_cross_attention = True

    # Split data into train and validation sets
    train_data, val_data = train_test_split(
        data, 
        test_size=validation_split, 
        random_state=42
    )

    # Create datasets
    train_dataset = ChatbotDataset(
        inputs=train_data['input'].values,
        outputs=train_data['output'].values,
        tokenizer=tokenizer,
        max_length=max_length
    )

    val_dataset = ChatbotDataset(
        inputs=val_data['input'].values,
        outputs=val_data['output'].values,
        tokenizer=tokenizer,
        max_length=max_length
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )

    # Training settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training tracking variables
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = {
        'train_loss': [],
        'val_loss': []
    }

    # Training loop
    logging.info("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        # Training
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Move all batch tensors to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            decoder_attention_mask = batch['decoder_attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix({'training_loss': loss.item()})

        # Calculate average losses
        avg_train_loss = total_train_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device)

        # Update training history
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(val_loss)

        # Log progress
        logging.info(f'Epoch {epoch+1}/{num_epochs}:')
        logging.info(f'Average Training Loss: {avg_train_loss:.4f}')
        logging.info(f'Validation Loss: {val_loss:.4f}')

        # Save best model
        if val_loss < best_val_loss:
            logging.info(f'Validation loss decreased from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...')
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save model and tokenizer
            model.save_pretrained(f'{model_save_dir}/best_model')
            tokenizer.save_pretrained(f'{model_save_dir}/best_tokenizer')
            
            # Save training metadata
            metadata = {
                'model_type': model_name,
                'max_length': max_length,
                'num_epochs_completed': epoch + 1,
                'best_validation_loss': best_val_loss,
                'training_history': training_history,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            }
            
            with open(f'{model_save_dir}/training_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=4)
        else:
            patience_counter += 1
            logging.info(f'Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}')

        # Early stopping
        if patience_counter >= early_stopping_patience:
            logging.info(f'Early stopping triggered after {epoch+1} epochs')
            break

    logging.info("Training completed!")
    return model, tokenizer, training_history

def generate_response(model, tokenizer, input_text, max_length=128):
    """
    Generate a response using the trained model
    """
    # Prepare input text
    inputs = tokenizer(
        input_text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Generate response
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        num_beams=5,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    # Training configuration
    config = {
        'batch_size': 4,
        'num_epochs': 20,
        'learning_rate': 3e-5,
        'max_length': 128,
        'model_name': 'bert-base-multilingual-cased',
        'train_data_path': 'data/processed/train_data.csv',
        'model_save_dir': 'models/saved_models',
        'validation_split': 0.1,
        'early_stopping_patience': 3
    }

    # Train the model
    model, tokenizer, history = train_model(**config)