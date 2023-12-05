import torch
import os
import re

def load_avg_loss_values(directory):
    # Dictionary to store epoch and average loss values
    epoch_avg_loss_dict = {}

    # Regular expression to match filenames and extract epoch numbers
    filename_pattern = re.compile(r'loss_epoch_(\d+)\.pth')

    # List all files in the directory
    for filename in os.listdir(directory):
        match = filename_pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            file_path = os.path.join(directory, filename)
            data = torch.load(file_path)
            avg_loss = sum(data['loss']) / len(data['loss'])  # Calculate average loss
            epoch_avg_loss_dict[epoch] = avg_loss

    return epoch_avg_loss_dict

directory = '.'  # Replace with your directory path

# Load average loss values
avg_loss_dict = load_avg_loss_values(directory)

# Print average losses for each epoch
for epoch in sorted(avg_loss_dict.keys()):
    print(f"Epoch {epoch}: Average Loss = {avg_loss_dict[epoch]}")

