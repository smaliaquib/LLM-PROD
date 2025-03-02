from fastapi import FastAPI, HTTPException
import tiktoken
import torch
from pathlib import Path
from pydantic import BaseModel

# Assuming utils.py and previous_chapters.py are in the same directory
from utils import GPTModel
from utils import generate, text_to_token_ids, token_ids_to_text

app = FastAPI()

# Define the path to the finetuned model
finetuned_model_path = Path("weights/gpt2-medium355M-sft.pth")

# Base configuration for the model
BASE_CONFIG = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,  # Dropout rate
    "qkv_bias": True,  # Query-key-value bias
}

# Model configurations
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Choose the model
CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# Initialize the model
model = GPTModel(BASE_CONFIG)
model.load_state_dict(
    torch.load(
        finetuned_model_path, map_location=torch.device("cpu"), weights_only=True
    )
)
model.eval()

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("gpt2")


# Define the request model
class TextGenerationRequest(BaseModel):
    prompt: str


# Define the response model
class TextGenerationResponse(BaseModel):
    generated_text: str


# Function to extract the response text
def extract_response(response_text, input_text):
    return response_text[len(input_text) :].replace("### Response:", "").strip()


# Define the endpoint
@app.post("/generate-text/", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    try:
        # Tokenize the input prompt
        token_ids = text_to_token_ids(request.prompt, tokenizer)

        # Generate text
        generated_token_ids = generate(
            model=model,
            idx=token_ids,
            max_new_tokens=35,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256,
        )

        # Convert token IDs to text
        generated_text = token_ids_to_text(generated_token_ids, tokenizer)

        # Extract the response
        response_text = extract_response(generated_text, request.prompt)

        return TextGenerationResponse(generated_text=response_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
