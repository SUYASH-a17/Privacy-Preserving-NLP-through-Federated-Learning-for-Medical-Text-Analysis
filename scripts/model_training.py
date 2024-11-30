# scripts/model_training.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from data_processing import load_config, load_data, preprocess_data

class TextDataset(Dataset):
    """
    Custom Dataset for loading text data.
    """
    def __init__(self, X, y):
        self.X = X.toarray()
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), self.y[idx]

class SimpleTextClassifier(nn.Module):
    """
    A simple feedforward neural network for text classification.
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(SimpleTextClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2, output_dim)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    """
    Train the model and validate after each epoch.
    """
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= total_train
        train_accuracy = train_correct / total_train
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= total_val
        val_accuracy = val_correct / total_val
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

def main():
    config = load_config('configs/config.yaml')
    device = torch.device(config['deployment']['device'] if torch.cuda.is_available() else 'cpu')
    
    # Load and preprocess data
    df = load_data(config['data']['raw_path'])
    X, y, vectorizer, label_encoder = preprocess_data(df)
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create DataLoaders
    train_dataset = TextDataset(X_train, y_train)
    val_dataset = TextDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Initialize model, criterion, optimizer
    input_dim = X.shape[1]
    hidden_dim1 = 256
    hidden_dim2 = 128
    output_dim = len(label_encoder.classes_)
    
    model = SimpleTextClassifier(input_dim, hidden_dim1, hidden_dim2, output_dim).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Train the model
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=config['training']['num_epochs']
    )
    
    # Save the model
    models_dir = config['models']['dir']
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, config['models']['model_file'])
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
