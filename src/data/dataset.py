import torch
from torch.utils.data import Dataset

class PreTokenizedRlhfDataset(Dataset):
    def __init__(self, pt_file_path="../../data/tokenized/rlhf_tokens.pt"):
        # Load the pre-processed list of dictionaries instantly
        print(f"Loading pre-tokenized dataset from {pt_file_path}...")
        self.data = torch.load(pt_file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "prompt_id": item["prompt_id"],
            "annotator_id": item["annotator_id"],
            "vote": item["vote"],
            "input_ids_A": item["input_ids_A"],
            "input_ids_B": item["input_ids_B"]
        }