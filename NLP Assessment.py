#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Input:
# The input to the seq2seq model will be the architecture of a neural network
# created in PyTorch. This includes detailed information about its layers,
# configurations, and parameters.

# Output:
# - The output from the seq2seq model will be a concise textual description
#   that includes:
#   The input shape (e.g., [b,200,10]).
#   The output shape (e.g., [b,10]).


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

# Define the dataset class
class NeuralNetworkDataset(Dataset):
    def __init__(self, num_samples, max_layers, max_dim, max_summary_length=128):
        super(NeuralNetworkDataset, self).__init__()
        self.num_samples = num_samples
        self.max_layers = max_layers
        self.max_dim = max_dim
        self.max_summary_length = max_summary_length
        self.data = self.generate_data()

    def generate_data(self):
        data = []
        for _ in range(self.num_samples):
            # Generate the neural network architecture
            layers = []
            num_layers = random.randint(1, self.max_layers)
            prev_dim = random.randint(1, self.max_dim)
            for _ in range(num_layers):
                curr_dim = random.randint(1, self.max_dim)
                layers.append(np.array([prev_dim, curr_dim]))
                prev_dim = curr_dim
            layers = np.array(layers)

            # Generate a textual summary for the architecture
            input_shape = f"[b,{layers[0][0]}]"
            output_shape = f"[b,{layers[-1][1]}]"
            summary = f"The input shape is {input_shape} and the output shape is {output_shape}."

            # Padding the layers to have the same length
            layers_padded = np.zeros((self.max_layers, 2))
            layers_padded[:len(layers), :] = layers

            # Pad the summary to the maximum length
            summary_padded = torch.zeros(self.max_summary_length, dtype=torch.long)
            summary_encoded = torch.tensor([ord(char) for char in summary], dtype=torch.long)
            summary_padded[:len(summary_encoded)] = summary_encoded

            # Convert layers_padded to a tensor
            layers_tensor = torch.tensor(layers_padded, dtype=torch.float32)
            data.append((layers_tensor, summary_padded))

        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

# Define the sequence-to-sequence model
class Seq2SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(Seq2SeqModel, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, target_seq):
        # Encode the input sequence
        encoder_outputs, (hidden, cell) = self.encoder(input_seq)

        # Initialize decoder hidden and cell states from encoder
        decoder_hidden = hidden  # Shape: (num_layers, batch_size, hidden_size)
        decoder_cell = cell      

        # Initialize the first decoder input
        decoder_input = torch.zeros(input_seq.size(0), 1, self.fc.in_features, device=input_seq.device)

        # List to store decoder outputs
        outputs = []

        # Loop through each timestep in the target sequence
        for i in range(target_seq.size(1)):
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, (decoder_hidden, decoder_cell))
            output = self.fc(decoder_output.squeeze(1))
            outputs.append(output)
            decoder_input = output.unsqueeze(1)

        outputs = torch.stack(outputs, dim=1)
        return outputs

# Train the model
num_samples = 10000
max_layers = 10
max_dim = 100
dataset = NeuralNetworkDataset(num_samples, max_layers, max_dim)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

input_size = max_layers * 2  # Each layer has an input and output dimension
hidden_size = 128
output_size = 256  # Assuming max length of summary is 128 characters
num_layers = 2
dropout = 0.2

model = Seq2SeqModel(input_size, hidden_size, output_size, num_layers, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for i, (layers, summary) in enumerate(dataloader):

        
        layers = layers.view(layers.size(0), -1)  # Flatten the layers tensor

        # Forward pass
        outputs = model(layers, summary)

        # Compute the loss
        loss = criterion(outputs.view(-1, output_size), summary.view(-1))

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

# Evaluate the model
def evaluate_model(model, neural_networks):
    correct = 0
    total = 0
    for network in neural_networks:
        input_seq = torch.tensor(network, dtype=torch.float32).unsqueeze(0)  # Convert the network to a tensor
        outputs = model(input_seq, None)
        summary = ""
        for output in outputs.squeeze(0):
            _, predicted = torch.max(output, 0)
            summary += chr(predicted.item())

        expected_summary = f"The input shape is [b,{network[0][0]}] and the output shape is [b,{network[-1][1]}]."
        if summary == expected_summary:
            correct += 1
        total += 1

    accuracy = correct / total
    f1_score = 2 * accuracy / (1 + accuracy)
    return accuracy, f1_score

# Testing the model on random 5 new neural networks, will be checked on inputs provided
new_networks = [

]

accuracy, f1_score = evaluate_model(model, new_networks)
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1_score:.4f}")

