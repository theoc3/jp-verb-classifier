import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
import os

torch.manual_seed(33)
np.random.seed(33)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

df = pd.read_csv('data/data_2/sentences_with_segments.csv', low_memory=False)
print(f"Loaded {len(df)} processed sentences with individual segments")

data = []
for idx, row in df.iterrows():
    for i in range(1, 11):  # There are up to 10 verbs per sentence
        verb_col = f'verb{i}'
        type_col = f'verbType{i}'
        segment_col = f'segment{i}'
        
        if pd.notna(row[verb_col]) and pd.notna(row[type_col]) and pd.notna(row[segment_col]):
            data.append({
                'segment': row[segment_col],
                'verb': row[verb_col],
                'verb_type': row[type_col]
            })

verb_df = pd.DataFrame(data)
print(f"Extracted {len(verb_df)} verb-segment pairs")

# Path to save model
model_path = 'models/bilstm_model.pt'

# Define model class and dataset here (keeping these unchanged)
class VerbSegmentDataset(Dataset):
    def __init__(self, segments, labels, char_to_idx, label_to_idx, max_length=50):
        self.segments = segments
        self.labels = labels
        self.char_to_idx = char_to_idx
        self.label_to_idx = label_to_idx
        self.max_length = max_length
        
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = self.segments[idx]
        label = self.labels[idx]
        
        # characters to indices
        char_indices = [self.char_to_idx.get(char, 0) for char in segment[:self.max_length]]
        if len(char_indices) < self.max_length:
            char_indices += [0] * (self.max_length - len(char_indices))
            
        # characters to tensors
        segment_tensor = torch.tensor(char_indices, dtype=torch.long)
        label_tensor = torch.tensor(self.label_to_idx[label], dtype=torch.long)
        
        return segment_tensor, label_tensor

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=True, 
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text = [batch size, max_length]
        embedded = self.embedding(text)
        # embedded = [batch size, max_length, embedding_dim]
        
        output, (hidden, cell) = self.lstm(embedded)
        # output = [batch size, max_length, hidden_dim * 2]
        # hidden = [n_layers * 2, batch size, hidden_dim]
        
        # concatenate the final forward and backward hidden states
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        # hidden = [batch size, hidden_dim * 2]
        
        return self.fc(hidden)

if os.path.exists(model_path):
    print(f"Loading existing model from '{model_path}'")
    checkpoint = torch.load(model_path, map_location=device)
    
    char_to_idx = checkpoint['char_to_idx']
    label_to_idx = checkpoint['label_to_idx']
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    vocab_size = checkpoint['model_config']['vocab_size']
    num_classes = checkpoint['model_config']['output_dim']
    
    model = BiLSTMClassifier(
        vocab_size=checkpoint['model_config']['vocab_size'],
        embedding_dim=checkpoint['model_config']['embedding_dim'],
        hidden_dim=checkpoint['model_config']['hidden_dim'],
        output_dim=checkpoint['model_config']['output_dim'],
        n_layers=checkpoint['model_config']['n_layers'],
        dropout=checkpoint['model_config']['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
    
else:
    # remove classes that have only 1 member
    class_counts = verb_df['verb_type'].value_counts()
    single_classes = class_counts[class_counts == 1].index
    print(f"Removing {len(single_classes)} classes with only 1 example: {', '.join(single_classes)}")
    verb_df = verb_df[~verb_df['verb_type'].isin(single_classes)]
    print(f"Remaining examples: {len(verb_df)}")

    chars = set()
    for segment in verb_df['segment']:
        chars.update(segment)
    char_to_idx = {char: idx+1 for idx, char in enumerate(sorted(chars))} 
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    vocab_size = len(char_to_idx) + 1

    label_to_idx = {label: idx for idx, label in enumerate(sorted(verb_df['verb_type'].unique()))}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    num_classes = len(label_to_idx)

    print(f"Vocabulary size: {vocab_size}, Number of classes: {num_classes}")

    ### PARAMETERS 
    MAX_LENGTH = 50
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    N_LAYERS = 2
    DROPOUT = 0.5
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 10

    X_train, X_test, y_train, y_test = train_test_split(
        verb_df['segment'].values, verb_df['verb_type'].values, 
        test_size=0.2, random_state=42, stratify=verb_df['verb_type']
    )
    train_dataset = VerbSegmentDataset(X_train, y_train, char_to_idx, label_to_idx, MAX_LENGTH)
    test_dataset = VerbSegmentDataset(X_test, y_test, char_to_idx, label_to_idx, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    def train_model(model, train_loader, optimizer, criterion, device):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            segments, labels = batch
            segments, labels = segments.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(segments)
            
            loss = criterion(predictions, labels)
            acc = (predictions.argmax(1) == labels).float().mean()
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
        return epoch_loss / len(train_loader), epoch_acc / len(train_loader)

    def evaluate_model(model, test_loader, criterion, device):
        model.eval()
        epoch_loss = 0
        epoch_acc = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                segments, labels = batch
                segments, labels = segments.to(device), labels.to(device)
                
                predictions = model(segments)
                loss = criterion(predictions, labels)
                acc = (predictions.argmax(1) == labels).float().mean()
                
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                
                all_preds.extend(predictions.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return epoch_loss / len(test_loader), epoch_acc / len(test_loader), all_preds, all_labels

    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=num_classes,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Training BiLSTM model...")
    best_acc = 0
    metrics = []

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        eval_loss, eval_acc, all_preds, all_labels = evaluate_model(model, test_loader, criterion, device)
        
        metrics.append({
            'Epoch': epoch + 1,
            'Train Loss': train_loss,
            'Train Acc (%)': train_acc * 100,
            'Eval Loss': eval_loss,
            'Eval Acc (%)': eval_acc * 100
        })
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\tEval Loss: {eval_loss:.4f} | Eval Acc: {eval_acc*100:.2f}%")
        
        # save if model is better
        if eval_acc > best_acc:
            best_acc = eval_acc
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'char_to_idx': char_to_idx,
                'label_to_idx': label_to_idx,
                'model_config': {
                    'vocab_size': vocab_size,
                    'embedding_dim': EMBEDDING_DIM,
                    'hidden_dim': HIDDEN_DIM,
                    'output_dim': num_classes,
                    'n_layers': N_LAYERS,
                    'dropout': DROPOUT
                }
            }, model_path)
            print(f"Model saved with accuracy: {eval_acc*100:.2f}%")

    print(f"BiLSTM model saved to '{model_path}'")

    epochs_df = pd.DataFrame(metrics)
    epochs_df.to_csv('bilstm_epochs.csv', index=False)
    print("Training metrics saved to 'bilstm_epochs.csv'")

    # evaluate
    _, _, all_preds, all_labels = evaluate_model(model, test_loader, criterion, device)
    print("\nClassification Report:")
    y_pred = [idx_to_label[pred] for pred in all_preds]
    y_true = [idx_to_label[label] for label in all_labels]
    
    report = classification_report(y_true, y_pred)
    print(report)
    
    # Save evaluation report
    report_path = model_path.replace('.pt', '_evaluation.txt')
    with open(report_path, 'w') as f:
        f.write(f"Classification Report for BiLSTM Model:\n\n")
        f.write(report)
    print(f"Evaluation report saved to '{report_path}'")

##########################################################################################

def predict_verb_type(segment, model, char_to_idx, label_to_idx, idx_to_label, max_length=50, device=device):
    model.eval()
    
    # segment to index
    char_indices = [char_to_idx.get(char, 0) for char in segment[:max_length]]
    if len(char_indices) < max_length:
        char_indices += [0] * (max_length - len(char_indices))
    
    # segment to tensor
    segment_tensor = torch.tensor([char_indices], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output = model(segment_tensor)
        prediction = output.argmax(1).item()
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence = probabilities[0][prediction].item()
    
    return idx_to_label[prediction], confidence

### TEST PREDICTION
print("\nExample Predictions:")
test_df = pd.read_csv('test.csv')
test_segments = test_df['segment'].tolist()
for segment in test_segments:
    pred_type, confidence = predict_verb_type(segment, model, char_to_idx, label_to_idx, idx_to_label)
    print(f"Segment: {segment}")
    print(f"Predicted verb type: {pred_type} (confidence: {confidence:.2f})")