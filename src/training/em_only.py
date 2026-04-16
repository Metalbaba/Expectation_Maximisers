import torch
import pandas as pd
import numpy as np
import sys
import os

# Adjust path to import from models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.em import DawidSkeneEM  # Assumes you saved the EM class here

def main():
    # Hardware acceleration
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running Projected EM on device: {device}")

    # 1. Load Data
    votes_df = pd.read_csv("../../data/processed/simulated_noisy_votes.csv")
    truth_df = pd.read_csv("../../data/processed/ground_truth.csv")
    params_df = pd.read_csv("../../data/true_params/true_annotator_parameters.csv")

    num_prompts = truth_df['prompt_id'].nunique()
    num_annotators = params_df['annotator_id'].nunique()

    # Convert to PyTorch tensors
    prompts = torch.tensor(votes_df['prompt_id'].values, dtype=torch.long, device=device)
    annotators = torch.tensor(votes_df['annotator_id'].values, dtype=torch.long, device=device)
    votes = torch.tensor(votes_df['vote'].values, dtype=torch.float32, device=device)
    
    true_labels = truth_df['truth_is_A'].values
    true_accuracies = params_df['true_accuracy'].values

    # 2. Initialize and Train
    em = DawidSkeneEM(num_prompts=num_prompts, num_annotators=num_annotators, device=device)
    
    print("Starting EM Algorithm...")
    em.fit(prompts, annotators, votes, epochs=15)

    # 3. Validate Results
    # Gamma represents the model's confidence that A is the true winner
    predicted_probs = em.gamma.cpu().numpy()
    
    # Hard labeling: If confidence > 0.5, we predict A (1), else B (0)
    predicted_labels = (predicted_probs > 0.5).astype(int)
    
    accuracy = np.mean(predicted_labels == true_labels)
    
    # Calculate Mean Absolute Error for the worker parameters
    # Averaging our alpha and beta estimates to compare against the 1D true parameter
    estimated_accuracies = ((em.alpha + em.beta) / 2.0).cpu().numpy()
    param_mae = np.mean(np.abs(estimated_accuracies - true_accuracies))

    print("-" * 30)
    print(f"Final Model Labeling Accuracy: {accuracy * 100:.2f}%")
    print(f"Worker Parameter Estimation MAE: {param_mae:.4f}")
    print("-" * 30)
    
    # Sanity Check on an Adversary vs. Expert
    print("Parameter Sanity Check:")
    for i in [0, 15, 30, 48]: # Just sampling a few random workers
        print(f"Worker {i} -> True Acc: {true_accuracies[i]:.2f} | Est Acc: {estimated_accuracies[i]:.2f}")

if __name__ == "__main__":
    main()