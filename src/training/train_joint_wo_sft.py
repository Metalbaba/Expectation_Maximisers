import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.em import DawidSkeneEM
from models.dummy_llm import DummyCausalLM
from models.dpo import get_batch_logps, weighted_dpo_loss
from data.dataset import RlhfCrowdDataset

def get_llm_priors(policy_model, ref_model, dataloader, num_prompts, device, beta=0.1):
    """Calculates P(A is better) for all prompts using current LLM state."""
    policy_model.eval()
    priors = torch.full((num_prompts,), 0.5, device=device)
    
    with torch.no_grad():
        for batch in dataloader:
            p_ids = batch['prompt_id'].to(device)
            ids_A, ids_B = batch['input_ids_A'].to(device), batch['input_ids_B'].to(device)
            
            # Calculate sequence log-probabilities
            pi_A = get_batch_logps(policy_model(ids_A), ids_A)
            pi_B = get_batch_logps(policy_model(ids_B), ids_B)
            ref_A = get_batch_logps(ref_model(ids_A), ids_A)
            ref_B = get_batch_logps(ref_model(ids_B), ids_B)
            
            # DPO formulation for implicit reward
            logits = (pi_A - pi_B) - (ref_A - ref_B)
            
            # Convert reward difference to probability using sigmoid
            prob_A = torch.sigmoid(beta * logits)
            priors[p_ids] = prob_A
            
    return priors

def train_joint_real():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Starting True Joint Training on {device}...")

    # 1. Setup Dataset & Dataloader
    dataset = RlhfCrowdDataset(
        votes_csv="../../data/processed/simulated_noisy_votes.csv", 
        truth_csv="../../data/processed/ground_truth.csv"
    )
    # batch_size=3 is required because we simulated 3 workers per prompt. 
    # This keeps all votes for a single prompt in the same batch.
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
    
    num_prompts = len(dataset.truth_df)
    num_annotators = dataset.votes_df['annotator_id'].nunique()
    
    # 2. Setup Models (Vocab size is 50257 to match GPT-2 Tokenizer)
    policy_model = DummyCausalLM(vocab_size=50257).to(device)
    ref_model = DummyCausalLM(vocab_size=50257).to(device)
    ref_model.eval()
    optimizer = optim.AdamW(policy_model.parameters(), lr=5e-5)

    em = DawidSkeneEM(num_prompts=num_prompts, num_annotators=num_annotators, device=device)

    # Pre-load all votes into memory for the EM step
    all_prompts = torch.tensor(dataset.votes_df['prompt_id'].values, device=device)
    all_annotators = torch.tensor(dataset.votes_df['annotator_id'].values, device=device)
    all_votes = torch.tensor(dataset.votes_df['vote'].values, dtype=torch.float32, device=device)
    golden_truth = torch.tensor(dataset.truth_df['truth_is_A'].values, device=device)

    epochs = 30
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # STEP A: LLM acts as prior for EM
        llm_priors = get_llm_priors(policy_model, ref_model, dataloader, num_prompts, device)
        
        # STEP B: EM Algorithm updates annotator profiles and creates Trust Weights
        em.e_step(all_prompts, all_annotators, all_votes, llm_priors=llm_priors)
        em.m_step(all_prompts, all_annotators, all_votes)
        trust_weights_all = em.gamma.detach()
        
        # Check LLM Accuracy against the Golden Dataset
        # If LLM prior > 0.5, it predicts A. Compare to golden_truth.
        llm_predictions = (llm_priors > 0.5).int()
        accuracy = (llm_predictions == golden_truth).float().mean().item()
        print(f"Golden Dataset Accuracy: {accuracy * 100:.2f}%")

        # STEP C: Train the LLM using Weighted DPO
        policy_model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            ids_A = batch['input_ids_A'].to(device)
            ids_B = batch['input_ids_B'].to(device)
            p_ids = batch['prompt_id'].to(device)
            
            # We only need one trust weight per prompt (since all 3 items in the batch are the same prompt)
            batch_trust = trust_weights_all[p_ids[0]].unsqueeze(0)
            
            optimizer.zero_grad()
            
            pi_A = get_batch_logps(policy_model(ids_A), ids_A)
            pi_B = get_batch_logps(policy_model(ids_B), ids_B)
            with torch.no_grad():
                ref_A = get_batch_logps(ref_model(ids_A), ids_A)
                ref_B = get_batch_logps(ref_model(ids_B), ids_B)
            
            loss = weighted_dpo_loss(pi_A[0:1], pi_B[0:1], ref_A[0:1], ref_B[0:1], trust_weights_A=batch_trust)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Weighted DPO Loss: {total_loss / num_prompts:.4f}")

if __name__ == "__main__":
    train_joint_real()