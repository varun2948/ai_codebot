import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import RobertaTokenizer, RobertaModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd


# ------------------ 1. Data Augmentation Functions ------------------ #
def augment_code(code):
    """Apply various code transformations to create augmented samples"""
    augmented_samples = []
    
    # 1. Variable name changes
    if '=' in code:
        vars = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b(?=\s*=)', code)
        if vars:
            for var in vars:
                new_var = var + '_new'
                augmented = code.replace(var, new_var)
                augmented_samples.append(augmented)
    
    # 2. Comment variations
    if '#' in code:
        augmented = code.replace('#', '# ')
        augmented_samples.append(augmented)
    
    # 3. Whitespace variations
    augmented = re.sub(r'\s+', ' ', code)
    augmented_samples.append(augmented)
    
    # 4. Line break variations
    if ';' in code:
        augmented = code.replace(';', ';\n')
        augmented_samples.append(augmented)
    
    return augmented_samples

# ------------------ 2. Model ------------------ #
class CodeReviewClassifier(nn.Module):
    def __init__(self, hidden_dim=768, num_labels=2, dropout_rate=0.2):
        super(CodeReviewClassifier, self).__init__()
        self.codebert = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


# ------------------ 3. Dataset ------------------ #
class CodeReviewDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128, augment=False):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code = str(self.data.iloc[idx]["code"])
        label = int(self.data.iloc[idx]["label"])
        
        # Apply data augmentation for training set
        if self.augment and label == 1:  # Only augment bad code samples
            augmented_samples = augment_code(code)
            if augmented_samples:
                code = random.choice(augmented_samples)

        encoding = self.tokenizer(
            code,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ------------------ 4. Load Data ------------------ #
df = pd.read_csv("code_review_dataset.csv")

# Calculate class weights for balanced sampling
class_counts = df['label'].value_counts()
total_samples = len(df)
class_weights = [total_samples / (2 * count) for count in class_counts]
sample_weights = [class_weights[label] for label in df['label']]

# Split data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Create samplers
train_sampler = WeightedRandomSampler(
    weights=[class_weights[label] for label in train_df['label']],
    num_samples=len(train_df),
    replacement=True
)

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
train_dataset = CodeReviewDataset(train_df, tokenizer, augment=True)
val_dataset = CodeReviewDataset(val_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=8)

# ------------------ 5. Training Setup ------------------ #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CodeReviewClassifier(dropout_rate=0.2).to(device)
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# Early stopping parameters
best_val_loss = float('inf')
patience = 3
patience_counter = 0
best_model_state = None

# ------------------ 6. Training Loop ------------------ #
EPOCHS = 20
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after epoch {epoch+1}")
            model.load_state_dict(best_model_state)  # Restore best model
            break

# Plot and save loss curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_curves.png')
plt.close()

# ------------------ 7. Evaluation ------------------ #
def evaluate_model(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    # Plot and save confusion matrix with improved styling
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Good Code', 'Bad Code'],
                yticklabels=['Good Code', 'Bad Code'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n🔍 Evaluation Results:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)


evaluate_model(model, val_loader, device)

# ------------------ 8. Save Model ------------------ #
torch.save(model.state_dict(), "codebert_review_model_balanced.pt")
print("✅ Model saved as codebert_review_model_balanced.pt")
