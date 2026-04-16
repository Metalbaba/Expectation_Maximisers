import torch
import torch.nn as nn

class DummyCausalLM(nn.Module):
    def __init__(self, vocab_size=1000, d_model=32, n_heads=2, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # A single transformer encoder layer acts as our LLM block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projects back to vocabulary size to get logits
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        """
        input_ids shape: (batch_size, seq_len)
        Returns logits shape: (batch_size, seq_len, vocab_size)
        """
        # 1. Convert token IDs to embeddings
        x = self.embedding(input_ids)
        
        # 2. Generate a causal mask to ensure autoregressive behavior 
        # (tokens can only attend to previous tokens)
        seq_len = input_ids.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(input_ids.device)
        
        # 3. Pass through transformer
        x = self.transformer(x, mask=causal_mask)
        
        # 4. Output logits over the vocabulary
        logits = self.lm_head(x)
        return logits