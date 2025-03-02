import tiktoken
import torch
from utils import GPTModel

from pathlib import Path

finetuned_model_path = Path("weights/gpt2-medium355M-sft.pth")

BASE_CONFIG = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,  # Dropout rate
    "qkv_bias": True,  # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
model = GPTModel(BASE_CONFIG)

import torch

model.load_state_dict(
    torch.load(
        "weights/gpt2-medium355M-sft.pth",
        map_location=torch.device("cpu"),
        weights_only=True,
    )
)
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")

prompt = """Below is an instruction that describes a task. Write a response 
that appropriately completes the request.

### Instruction:
Convert the active sentence to passive: 'The chef cooks the meal every day.'
"""

from utils import generate, text_to_token_ids, token_ids_to_text


def extract_response(response_text, input_text):
    return response_text[len(input_text) :].replace("### Response:", "").strip()


torch.manual_seed(123)

token_ids = generate(
    model=model,
    idx=text_to_token_ids(prompt, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256,
)

response = token_ids_to_text(token_ids, tokenizer)
response = extract_response(response, prompt)
print(response)
