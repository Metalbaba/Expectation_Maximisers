from transformers import AutoModelForCausalLM

def load_sft_model(model_name="gpt2", device="cpu"):
    """
    Loads a pre-trained model to act as our Supervised Fine-Tuned (SFT) base.
    """
    print(f"Loading {model_name} onto {device}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    return model