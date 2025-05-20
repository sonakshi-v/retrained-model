import torch
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine
import numpy as np
import csv

# Function to calculate cosine similarity
def get_similarity(model, tokenizer, text1, text2):
    # Ensure tokenizer has a padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Encode texts
    encoded_input1 = tokenizer(text1, return_tensors='pt', padding=True, truncation=True, max_length=128)
    encoded_input2 = tokenizer(text2, return_tensors='pt', padding=True, truncation=True, max_length=128)

    # Generate model output
    with torch.no_grad():
        output1 = model(**encoded_input1)
        output2 = model(**encoded_input2)

    # Mean pooling to get sentence embeddings
    sentence_embedding1 = torch.mean(output1.last_hidden_state, dim=1).squeeze()
    sentence_embedding2 = torch.mean(output2.last_hidden_state, dim=1).squeeze()

    # Return cosine similarity
    return 1 - cosine(sentence_embedding1.numpy(), sentence_embedding2.numpy())

# Manual implementation of TOPSIS
def topsis(data, weights, impacts):
    normalized = data / np.sqrt((data**2).sum(axis=0))
    weighted = normalized * weights
    ideal = np.max(weighted, axis=0)
    anti_ideal = np.min(weighted, axis=0)
    distance_to_ideal = np.sqrt(((weighted - ideal)**2).sum(axis=1))
    distance_to_anti_ideal = np.sqrt(((weighted - anti_ideal)**2).sum(axis=1))
    score = distance_to_anti_ideal / (distance_to_anti_ideal + distance_to_ideal)
    # Using argsort to sort indexes by score in descending order (since higher is better)
    return np.argsort(-score) + 1  # +1 to adjust index to rank (1-based)

# Sample texts
text1 = "The company's revenue has grown by 15% in the last quarter."
text2 = "There has been a 15% increase in the company's quarterly revenue."

# Model names and their Hugging Face identifiers
model_ids = {
    "bert-base-uncased": "BERT",
    "roberta-base": "RoBERTa",
    "sentence-transformers/bert-base-nli-mean-tokens": "Sentence-BERT",
    "gpt2": "GPT-2",  # As a proxy for GPT-3
    "distilbert-base-uncased": "DistilBERT"
}

# Initialize models and tokenizers
models = {name: (AutoModel.from_pretrained(id), AutoTokenizer.from_pretrained(id)) for id, name in model_ids.items()}

# Compute similarities
similarities = []
model_names = []
for model_id, (model, tokenizer) in models.items():
    similarity = get_similarity(model, tokenizer, text1, text2)
    similarities.append(similarity)
    model_names.append(model_id)
    print(f"{model_id} similarity: {similarity}")

# Prepare data for TOPSIS
data = np.array(similarities).reshape(-1, 1)
weights = np.array([1])  # Weights for the criterion
impacts = "max"  # We are looking to maximize the similarity

# Apply TOPSIS
rankings = topsis(data, weights, impacts)

# Map rankings to model names
sorted_models = sorted(zip(model_names, rankings), key=lambda x: x[1])

# Save results to CSV
with open('results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Model', 'Rank'])
    for model, rank in sorted_models:
        writer.writerow([model, int(rank)])