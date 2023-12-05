from transformers import BertTokenizer, BertForSequenceClassification
import torch

# pre-trained model (BERT)
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)

# list of tasks
tasks = [
    {"text": "having lunch with my wife", "label": 1},  # Social
    {"text": "celebrating anniversary", "label": 2},  # Events
    {"text": "completing task", "label": 3},  # Work
    {"text": "paying the bills", "label": 4},  # Finance
]

# Tokenize and encode the tasks
encoded_tasks = [tokenizer(task["text"], return_tensors="pt") for task in tasks]
labels = torch.tensor([task["label"] for task in tasks])

# Train the model
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
num_epochs = 3

for epoch in range(num_epochs):
    for encoded_task, label in zip(encoded_tasks, labels):
        outputs = model(**encoded_task, labels=label.unsqueeze(0))
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_model")

# testing the model

# Load the fine-tuned model
fine_tuned_model = BertForSequenceClassification.from_pretrained("fine_tuned_model")

# Test the model on a new task
new_task = "having dinner with my dog"
encoded_new_task = tokenizer(new_task, return_tensors="pt")
output = fine_tuned_model(**encoded_new_task)
predicted_category = torch.argmax(output.logits).item()

# Print the predicted category
print(f"Task: {new_task}\nPredicted Category: {predicted_category}")
