import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
import os

def pre_tokenize_dataset():
    print("Loading CSVs and HuggingFace Dataset...")
    votes_df = pd.read_csv("../../data/processed/simulated_noisy_votes.csv")
    truth_df = pd.read_csv("../../data/processed/ground_truth.csv")
    
    # We load the exact number of prompts we simulated
    num_prompts = len(truth_df)
    hf_dataset = load_dataset("Anthropic/hh-rlhf", split=f"train[:{num_prompts}]")
    
    # Using GPT-2 as our baseline SFT model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    max_length = 128
    
    processed_data = []
    
    print("Tokenizing text... (This uses your CPU heavily)")
    for idx, vote_row in votes_df.iterrows():
        prompt_id = int(vote_row['prompt_id'])
        annotator_id = int(vote_row['annotator_id'])
        vote = float(vote_row['vote'])

        hf_row = hf_dataset[prompt_id]
        truth_is_A = truth_df[truth_df['prompt_id'] == prompt_id]['truth_is_A'].values[0]

        text_A = hf_row['chosen'] if truth_is_A else hf_row['rejected']
        text_B = hf_row['rejected'] if truth_is_A else hf_row['chosen']

        # Tokenize
        tokens_A = tokenizer(text_A, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        tokens_B = tokenizer(text_B, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")

        processed_data.append({
            "prompt_id": prompt_id,
            "annotator_id": annotator_id,
            "vote": vote,
            "input_ids_A": tokens_A['input_ids'].squeeze(0),
            "input_ids_B": tokens_B['input_ids'].squeeze(0)
        })

    os.makedirs("../../data/tokenized", exist_ok=True)
    torch.save(processed_data, "../../data/tokenized/rlhf_tokens.pt")
    print("Saved extremely fast tensor dataset to data/tokenized/rlhf_tokens.pt!")

if __name__ == "__main__":
    pre_tokenize_dataset()