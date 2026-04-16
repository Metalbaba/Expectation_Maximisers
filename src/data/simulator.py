import random
import numpy as np
import pandas as pd
from datasets import load_dataset

# ==========================================
# 1. KARGER'S (l, r)-REGULAR GRAPH SETTINGS
# ==========================================
NUM_TASKS = 1000
NUM_ANNOTATORS = 50
L_WORKERS_PER_TASK = 3
R_TASKS_PER_WORKER = 60
# Ensure m * l = n * r
assert NUM_TASKS * L_WORKERS_PER_TASK == NUM_ANNOTATORS * R_TASKS_PER_WORKER

annotator_types = (['expert'] * 10 + ['average'] * 25 + ['spammer'] * 10 + ['adversary'] * 5)
np.random.shuffle(annotator_types)

true_theta = {}
for i, a_type in enumerate(annotator_types):
    if a_type == 'expert': true_theta[i] = np.random.uniform(0.90, 0.99)
    elif a_type == 'average': true_theta[i] = np.random.uniform(0.65, 0.85)
    elif a_type == 'spammer': true_theta[i] = np.random.uniform(0.45, 0.55)
    elif a_type == 'adversary': true_theta[i] = np.random.uniform(0.05, 0.20)

print("Downloading dataset...")
dataset = load_dataset("Anthropic/hh-rlhf", split=f"train[:{NUM_TASKS}]")

# ==========================================
# 2. GENERATE CONFIGURATION MODEL BIPARTITE GRAPH
# ==========================================
# Create half-edges for tasks and workers
task_stubs = [i for i in range(NUM_TASKS) for _ in range(L_WORKERS_PER_TASK)]
worker_stubs = [j for j in range(NUM_ANNOTATORS) for _ in range(R_TASKS_PER_WORKER)]

# Shuffle worker stubs to create random pairings
random.shuffle(worker_stubs)
assignments = list(zip(task_stubs, worker_stubs))

# ==========================================
# 3. SIMULATE VOTING
# ==========================================
noisy_data = []
ground_truth_labels = []

for prompt_idx, row in enumerate(dataset):
    true_winner_text = row['chosen']
    true_loser_text = row['rejected']
    
    # Randomly assign ground truth to A (1) or B (0)
    truth_is_A = random.random() > 0.5
    ground_truth_labels.append({"prompt_id": prompt_idx, "truth_is_A": int(truth_is_A)})
    
    resp_A = true_winner_text if truth_is_A else true_loser_text
    resp_B = true_loser_text if truth_is_A else true_winner_text

    # Find all workers assigned to this task from our graph
    assigned_workers = [w for t, w in assignments if t == prompt_idx]

    for worker_id in assigned_workers:
        picked_true_winner = random.random() < true_theta[worker_id]
        chosen_resp = 1 if (picked_true_winner == truth_is_A) else 0 # 1 means voted A, 0 means voted B
        
        noisy_data.append({
            "prompt_id": prompt_idx,
            "annotator_id": worker_id,
            "vote": chosen_resp # Emulates PyTorch binary tensor input
        })

pd.DataFrame(noisy_data).to_csv("../../data/processed/simulated_noisy_votes.csv", index=False)
pd.DataFrame([{"annotator_id": k, "true_accuracy": v} for k, v in true_theta.items()]).to_csv("../../data/true_params/true_annotator_parameters.csv", index=False)
pd.DataFrame(ground_truth_labels).to_csv("../../data/processed/ground_truth.csv", index=False)
print("Graph generated and votes simulated!")