import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.em import DawidSkeneEM
from models.sft_model import load_sft_model
from models.dpo import get_batch_logps, weighted_dpo_loss
from data.dataset import PreTokenizedRlhfDataset
from utils.device import get_device
from utils.metrics import calculate_golden_accuracy, get_implicit_reward_margin, extract_worker_accuracies

def get_llm_priors(policy_model, ref_model, dataloader, num_prompts, device, beta=0.1):
    policy_model.eval()
    priors = torch.full((num_prompts,), 0.5, device=device)
    
    with torch.no_grad():
        for batch in dataloader:
            p_ids = batch['prompt_id'].to(device)
            ids_A, ids_B = batch['input_ids_A'].to(device), batch['input_ids_B'].to(device)
            
            pi_A = get_batch_logps(policy_model(ids_A).logits, ids_A)
            pi_B = get_batch_logps(policy_model(ids_B).logits, ids_B)
            ref_A = get_batch_logps(ref_model(ids_A).logits, ids_A)
            ref_B = get_batch_logps(ref_model(ids_B).logits, ids_B)
            
            logits = (pi_A - pi_B) - (ref_A - ref_B)
            priors[p_ids] = torch.sigmoid(beta * logits)
            
    return priors

def train_joint_real(epochs=15, beta=0.1):
    device = get_device()
    print(f"Starting True Joint Training on {device}...")

    # 1. Setup Dataset & Dataloader
    dataset = PreTokenizedRlhfDataset()
    
    # HARDWARE OPTIMIZATION: 
    # num_workers=4 uses your CPU cores to prepare batches asynchronously.
    # pin_memory=True speeds up the transfer to unified memory/GPU.
    # dataloader = DataLoader(dataset, batch_size=3, shuffle=False, num_workers=4, pin_memory=True)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
    
    votes_df = pd.read_csv("../../data/processed/simulated_noisy_votes.csv")
    truth_df = pd.read_csv("../../data/processed/ground_truth.csv")
    
    num_prompts = len(truth_df)
    num_annotators = votes_df['annotator_id'].nunique()
    
    # 2. Setup Real SFT Models
    policy_model = load_sft_model("gpt2", device)
    ref_model = load_sft_model("gpt2", device)
    ref_model.eval() # Reference model is frozen forever
    optimizer = optim.AdamW(policy_model.parameters(), lr=1e-5) # Lower LR for real LLMs

    em = DawidSkeneEM(num_prompts=num_prompts, num_annotators=num_annotators, device=device)

    all_prompts = torch.tensor(votes_df['prompt_id'].values, device=device)
    all_annotators = torch.tensor(votes_df['annotator_id'].values, device=device)
    all_votes = torch.tensor(votes_df['vote'].values, dtype=torch.float32, device=device)
    golden_truth = torch.tensor(truth_df['truth_is_A'].values, device=device)

    # Dictionary to store metrics for the Jupyter Notebook
    history = {
        "accuracy": [],
        "dpo_loss": [],
        "reward_margin": [],
        "worker_alphas": []
    }

    # Tracking specific workers (You can adjust these IDs based on your true_params.csv)
    tracked_workers = [0, 15, 48] 

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # STEP A & B: LLM Prior & EM Update
        llm_priors = get_llm_priors(policy_model, ref_model, dataloader, num_prompts, device, beta)
        em.e_step(all_prompts, all_annotators, all_votes, llm_priors=llm_priors)
        em.m_step(all_prompts, all_annotators, all_votes)
        
        trust_weights_all = em.gamma.detach()
        
        # Calculate & Store Metrics
        acc = calculate_golden_accuracy(llm_priors, golden_truth)
        alphas = extract_worker_accuracies(em, tracked_workers)
        history["accuracy"].append(acc)
        history["worker_alphas"].append(alphas)
        
        print(f"Golden Dataset Accuracy: {acc * 100:.2f}%")

        # STEP C: Weighted DPO Training
        policy_model.train()
        epoch_loss = 0.0
        epoch_margin = 0.0
        
        for batch in dataloader:
            ids_A, ids_B = batch['input_ids_A'].to(device), batch['input_ids_B'].to(device)
            p_ids = batch['prompt_id'].to(device)
            batch_trust = trust_weights_all[p_ids[0]].unsqueeze(0)
            
            optimizer.zero_grad()
            
            # Note: HuggingFace models return an object. We need .logits
            pi_A = get_batch_logps(policy_model(ids_A).logits, ids_A)
            pi_B = get_batch_logps(policy_model(ids_B).logits, ids_B)
            with torch.no_grad():
                ref_A = get_batch_logps(ref_model(ids_A).logits, ids_A)
                ref_B = get_batch_logps(ref_model(ids_B).logits, ids_B)
            
            loss = weighted_dpo_loss(pi_A[0:1], pi_B[0:1], ref_A[0:1], ref_B[0:1], trust_weights_A=batch_trust, beta=beta)
            margin = get_implicit_reward_margin(pi_A[0:1], pi_B[0:1], ref_A[0:1], ref_B[0:1], beta=beta)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_margin += margin
            
        avg_loss = epoch_loss / num_prompts
        avg_margin = epoch_margin / num_prompts
        history["dpo_loss"].append(avg_loss)
        history["reward_margin"].append(avg_margin)
        
        print(f"Weighted DPO Loss: {avg_loss:.4f} | Reward Margin: {avg_margin:.4f}")

    return history

if __name__ == "__main__":
    train_joint_real()