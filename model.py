
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import pandas as pd

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=10)

# Example: Load MIMIC-III dataset (replace with your actual loading and preprocessing)
# Assuming you have a CSV file with columns 'TEXT' (clinical notes) and 'LABEL' (disease label)
mimic_df = pd.read_csv("C:/Users/aadit/Downloads/archive (2)/mimic-iii-clinical-database-demo-1.4/DIAGNOSES_ICD.csv")

# Replace 'TEXT' and 'LABEL' with actual column names in your dataset
texts = mimic_df['row_id'].tolist()
labels = mimic_df['subject_id'].tolist()

# Tokenize and encode the text data
tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# Convert labels to PyTorch tensor
labels_tensor = torch.tensor(labels)

# Create TensorDataset
dataset = TensorDataset(tokenized_texts['input_ids'], tokenized_texts['attention_mask'], labels_tensor)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoader for training and validation
batch_size = 8  # Set your desired batch size
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define optimizer and learning rate scheduler
learning_rate = 2e-5  # Set your desired learning rate
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 3  # Set your desired number of epochs
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluation loop
model.eval()
val_loss = 0.0
correct_preds = 0
total_preds = 0

with torch.no_grad():
    for batch in val_dataloader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        val_loss += loss.item()
        _, predicted_labels = torch.max(logits, 1)
        correct_preds += (predicted_labels == labels).sum().item()
        total_preds += len(labels)

# Calculate validation accuracy
accuracy = correct_preds / total_preds
print(f"Validation Accuracy: {accuracy}")
