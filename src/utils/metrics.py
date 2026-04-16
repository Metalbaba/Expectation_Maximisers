import torch

def calculate_golden_accuracy(llm_priors: torch.Tensor, golden_truth: torch.Tensor):
    """Calculates if the LLM's belief matches the hidden Golden Dataset."""
    llm_predictions = (llm_priors > 0.5).int()
    accuracy = (llm_predictions == golden_truth).float().mean().item()
    return accuracy

def get_implicit_reward_margin(pi_logps_A, pi_logps_B, ref_logps_A, ref_logps_B, beta=0.1):
    """
    Calculates the average reward margin between the chosen and rejected response.
    A rising margin proves DPO is successfully teaching the model preference.
    """
    pi_logratios = pi_logps_A - pi_logps_B
    ref_logratios = ref_logps_A - ref_logps_B
    margin = beta * (pi_logratios - ref_logratios)
    return margin.mean().item()

def extract_worker_accuracies(em_model, worker_ids):
    """Extracts the current estimated accuracy (alpha) for specific workers to plot."""
    alphas = em_model.alpha.detach().cpu().numpy()
    return {w_id: alphas[w_id] for w_id in worker_ids}