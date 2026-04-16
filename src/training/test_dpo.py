import torch
import sys
import os

# Adjust path to import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.dummy_llm import DummyCausalLM
from models.dpo import get_batch_logps, weighted_dpo_loss

def test_pipeline():
    # Hardware acceleration for macOS Apple Silicon
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Testing on {device}...")

    vocab_size = 1000
    batch_size = 4
    seq_len = 15

    # 1. Initialize models (Active policy and frozen reference)
    policy_model = DummyCausalLM(vocab_size=vocab_size).to(device)
    ref_model = DummyCausalLM(vocab_size=vocab_size).to(device)
    ref_model.eval() # Reference model is always frozen in DPO

    # 2. Generate dummy data (token IDs)
    responses_A = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    responses_B = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    # Dummy trust weights from EM algorithm (e.g., model is 90% sure A is better for batch 1)
    trust_weights = torch.tensor([0.9, 0.1, 0.5, 0.99], device=device)

    # 3. Forward Pass
    logits_A = policy_model(responses_A)
    logits_B = policy_model(responses_B)
    
    with torch.no_grad():
        ref_logits_A = ref_model(responses_A)
        ref_logits_B = ref_model(responses_B)

    # 4. Extract sequence log-probabilities
    pi_logps_A = get_batch_logps(logits_A, responses_A)
    pi_logps_B = get_batch_logps(logits_B, responses_B)
    ref_logps_A = get_batch_logps(ref_logits_A, responses_A)
    ref_logps_B = get_batch_logps(ref_logits_B, responses_B)

    # 5. Calculate Weighted DPO Loss
    loss = weighted_dpo_loss(
        pi_logps_A, pi_logps_B, 
        ref_logps_A, ref_logps_B, 
        trust_weights_A=trust_weights
    )

    print(f"Forward pass successful. Loss: {loss.item():.4f}")
    
    # 6. Test Backward Pass
    loss.backward()
    print("Backward pass (gradient calculation) successful!")

if __name__ == "__main__":
    test_pipeline()