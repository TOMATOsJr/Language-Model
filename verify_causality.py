import torch
from nepali.training.model import TransformerLM
from nepali.training.config import ModelConfig

def verify_causality():
    cfg = ModelConfig()
    model = TransformerLM(
        vocab_size=cfg.vocab_size, dmodel=cfg.dmodel, num_layers=cfg.num_layers,
        num_heads=cfg.num_heads, dff=cfg.dff, dropout=cfg.dropout
    )
    model.eval()

    # Create random token sequence
    T = 50
    t = 25 # We will check logits at t=25
    tokens = torch.randint(0, cfg.vocab_size, (1, T))
    
    with torch.no_grad():
        logits1 = model(tokens)

    # Change token at t+1 (index 26)
    tokens_modified = tokens.clone()
    # ensure it changes
    tokens_modified[0, t+1] = (tokens[0, t+1] + 1) % cfg.vocab_size

    with torch.no_grad():
        logits2 = model(tokens_modified)

    # Check logits at position t (index 25)
    # The logits at t predict the token at t+1. 
    # They should depend ONLY on tokens <= t.
    diff = torch.abs(logits1[0, :t+1] - logits2[0, :t+1]).max().item()
    
    print(f"Max absolute difference in logits up to position {t}: {diff}")
    
    # Also check that logits at t+1 ARE different
    diff_after = torch.abs(logits1[0, t+1:] - logits2[0, t+1:]).max().item()
    print(f"Max absolute difference in logits after position {t}: {diff_after}")

    assert diff < 1e-5, f"Causality broken! Diff is {diff}"
    print("Causality successfully verified: Changing token at t+1 does not affect logits at position t.")

if __name__ == "__main__":
    verify_causality()
