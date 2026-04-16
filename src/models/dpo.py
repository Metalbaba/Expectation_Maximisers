import torch
import torch.nn.functional as F

def get_batch_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes the log probabilities of the actual sequence of tokens.
    logits: (batch_size, seq_len, vocab_size)
    labels: (batch_size, seq_len)
    """
    # Shift logits and labels so token t predicts token t+1
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    
    # Calculate log-probs using cross entropy (no reduction)
    log_probs = -F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)), 
        shifted_labels.view(-1), 
        reduction='none'
    )
    # Sum log-probs over the sequence length for each item in the batch
    return log_probs.view(shifted_labels.shape[0], -1).sum(dim=-1)

def weighted_dpo_loss(
    pi_logps_A: torch.Tensor, 
    pi_logps_B: torch.Tensor, 
    ref_logps_A: torch.Tensor, 
    ref_logps_B: torch.Tensor, 
    trust_weights_A: torch.Tensor,
    beta: float = 0.1
):
    """
    pi_logps: The active LLM's log probabilities for responses
    ref_logps: The frozen reference LLM's log probabilities for responses
    trust_weights_A: The gamma_i from the EM algorithm (Probability that A is better)
    """
    # Calculate the implicit reward difference (A vs B) for both models
    pi_logratios = pi_logps_A - pi_logps_B
    ref_logratios = ref_logps_A - ref_logps_B
    
    # The core DPO preference math
    logits = pi_logratios - ref_logratios
    
    # Normally DPO assumes A is the winner. 
    # Loss if A is the true winner:
    loss_if_A = -F.logsigmoid(beta * logits)
    
    # Loss if B is the true winner (we invert the logits):
    loss_if_B = -F.logsigmoid(-beta * logits)
    
    # EXPECTATION MAXIMIZATION INJECTION:
    # We weight the losses by the EM algorithm's confidence
    expected_loss = (trust_weights_A * loss_if_A) + ((1.0 - trust_weights_A) * loss_if_B)
    
    # Return the mean loss for the batch to backpropagate
    return expected_loss.mean()