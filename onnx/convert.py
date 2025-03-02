import torch
import tiktoken
import onnx
import onnxruntime
from utils import GPTModel, text_to_token_ids, token_ids_to_text, generate

# Load model configuration
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# Load the model
model = GPTModel(BASE_CONFIG)
model.load_state_dict(
    torch.load(
        "weights/gpt2-medium355M-sft.pth",
        map_location=torch.device("cpu"),
        weights_only=True,
    )
)
model.eval()

# print(f"Model exported to {onnx_path}")

dummy_input = torch.randint(
    0, 50257, (1, BASE_CONFIG["context_length"])
)  # Example input [batch_size=1, seq_length=32]
torch.onnx.export(
    model,
    dummy_input,
    "gpt2_medium.onnx",
    input_names=["input_ids"],
    #   do_constant_folding=False,
    #   export_params=True,
    opset_version=17,
    #   dynamic_axes={"input_ids": {1: "seq_length"}, "logits": {1: "seq_length"}},
    output_names=["logits"],
)


import onnxruntime as ort

# Load the ONNX model
onnx_path = "gpt2_medium.onnx"
session = ort.InferenceSession(onnx_path)

# Load tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Define input text
prompt = """Below is an instruction that describes a task. Write a response 
that appropriately completes the request.

### Instruction:
Convert the active sentence to passive: 'The chef cooks the meal every day.'
"""

# Convert input to token IDs
input_ids = text_to_token_ids(prompt, tokenizer).numpy()
# input_ids = torch.tensor(text_to_token_ids(prompt, tokenizer), dtype=torch.int64).unsqueeze(0)  # Shape: [1, seq_len]
# input_ids = input_ids.numpy()  # Convert to NumPy (shape should be [1, seq_len])
print("Input Shape:", input_ids.shape)  # Should print (1, seq_len)

# Convert to tensor and pad if necessary
if input_ids.shape[1] < BASE_CONFIG["context_length"]:
    padded_input = torch.tensor(input_ids, dtype=torch.int64)
    padded_input = torch.nn.functional.pad(
        padded_input,
        (0, BASE_CONFIG["context_length"] - input_ids.shape[1]),
        value=50256,
    )  # Use EOS token for padding
else:
    padded_input = torch.tensor(
        input_ids[: BASE_CONFIG["context_length"]], dtype=torch.int64
    )  # Truncate if too long

print(input_ids, padded_input)

input_ids = padded_input.numpy()
# print("Input Shape:", padded_input.shape)

# Run inference
outputs = session.run(["logits"], {"input_ids": input_ids})

# Decode output tokens
response_tokens = torch.tensor(outputs[0]).argmax(dim=-1).tolist()
# print(response_tokens, type(response_tokens))
response = token_ids_to_text(torch.tensor(response_tokens), tokenizer)


# Extract response
def extract_response(response_text, input_text):
    return response_text[len(input_text) :].replace("### Response:", "").strip()


response = extract_response(response, prompt)
print(f"Response: {response}")
